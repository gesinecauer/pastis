import sys
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.metrics.pairwise import paired_distances

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from pastis.optimization.utils_poisson import _setup_jax
    _setup_jax(debug_nan_inf=True)
    from jax import grad

    from utils import get_counts, set_counts_ambiguity
    from pastis.optimization import utils_poisson


def test_euclidean_distance():
    seed = 0
    nbeads = 100

    random_state = np.random.RandomState(seed=seed)
    struct = random_state.rand(nbeads, 3)
    row, col = (x.ravel() for x in np.indices((nbeads, nbeads)))

    euc_dis_correct = paired_distances(struct[row], struct[col])
    euc_dis_test = utils_poisson._euclidean_distance(struct, row=row, col=col)
    assert_array_almost_equal(euc_dis_correct, euc_dis_test)


@pytest.mark.parametrize("ambiguity", ["ua", "ambig", "pa"])
def test_find_beads_to_remove(ambiguity):
    lengths = np.array([40])
    ploidy = 2
    seed = 0
    alpha, beta = -3, 0.5
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.rand(lengths.sum() * ploidy, 3)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=struct_nan, random_state=random_state,
        use_poisson=True)

    struct_nan_correct = struct_nan[struct_nan < lengths.sum()]
    struct_nan_correct = np.append(
        struct_nan_correct, struct_nan_correct + lengths.sum())

    struct_nan_test = utils_poisson.find_beads_to_remove(
        counts, lengths=lengths, ploidy=ploidy, multiscale_factor=1)

    assert_array_equal(struct_nan_correct, struct_nan_test)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_struct_replace_nan(seed):
    lengths = np.array([10, 21, 4])
    ploidy = 2
    seed = 0
    num_nan_per_hmlg = 10

    coord0 = np.arange(lengths.sum() * ploidy, dtype=float).reshape(-1, 1)
    coord1 = coord2 = np.zeros_like(coord0)
    struct_interp_correct = np.concatenate([coord0, coord1, coord2], axis=1)
    struct = struct_interp_correct.copy()

    # Set specified beads to NaN
    random_state = np.random.RandomState(seed=seed)
    struct_nan = random_state.randint(
        low=0, high=lengths.sum(), size=num_nan_per_hmlg)
    struct_nan = struct_nan[struct_nan < lengths.sum()]
    struct_nan = np.append(struct_nan, struct_nan + lengths.sum())
    struct[struct_nan[struct_nan < struct.shape[0]], :] = np.nan

    struct_interp_test = utils_poisson._struct_replace_nan(
        struct, lengths=lengths, ploidy=ploidy, kind='linear')

    # Don't compare if whole molecule is NaN, or only 1 bead in mol is not NaN
    begin = end = 0
    for i in range(lengths.size * ploidy):
        end += np.tile(lengths, ploidy)[i]
        if np.isfinite(struct[begin:end, 0]).sum() < 2:
            struct_interp_correct[begin:end] = np.nan
            struct_interp_test[begin:end] = np.nan
        begin = end

    assert_array_equal(struct_interp_correct, struct_interp_test)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_jax_max(seed):
    random_state = np.random.RandomState(seed=seed)
    input1, input2 = random_state.uniform(low=-10, high=10, size=2)
    while input1 == input2:  # We don't care about grad if both values are equal
        input2 = random_state.uniform(low=-10, high=10)

    # Check objective
    obj = utils_poisson.jax_max(input1, input2)._value
    assert obj == np.maximum(input1, input2)

    # Check gradient
    diff_wrt_1st_arg = grad(
        utils_poisson.jax_max, argnums=0)(input1, input2)._value
    diff_wrt_2nd_arg = grad(
        utils_poisson.jax_max, argnums=1)(input1, input2)._value
    if input1 > input2:
        assert diff_wrt_1st_arg == 1
        assert diff_wrt_2nd_arg == 0
    else:
        assert diff_wrt_1st_arg == 0
        assert diff_wrt_2nd_arg == 1


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_jax_min(seed):
    random_state = np.random.RandomState(seed=seed)
    input1, input2 = random_state.uniform(low=-10, high=10, size=2)
    while input1 == input2:  # We don't care about grad if both values are equal
        input2 = random_state.uniform(low=-10, high=10)

    # Check objective
    obj = utils_poisson.jax_min(input1, input2)._value
    assert obj == np.minimum(input1, input2)

    # Check gradient
    diff_wrt_1st_arg = grad(
        utils_poisson.jax_min, argnums=0)(input1, input2)._value
    diff_wrt_2nd_arg = grad(
        utils_poisson.jax_min, argnums=1)(input1, input2)._value
    if input1 < input2:
        assert diff_wrt_1st_arg == 1
        assert diff_wrt_2nd_arg == 0
    else:
        assert diff_wrt_1st_arg == 0
        assert diff_wrt_2nd_arg == 1


def test_subset_chrom_of_struct_and_bias():
    ploidy = 2
    seed = 0
    lengths_full = np.array([35, 44, 16, 52, 73, 44])
    chrom_full = np.array(['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6'])
    chrom_subset = np.array(['chr6', 'chr1', 'chr2'])

    random_state = np.random.RandomState(seed=seed)
    n = lengths_full.sum()
    struct_full = random_state.rand(n * ploidy, 3)
    bias_full = 0.1 + random_state.rand(n)

    lengths_subset_correct = []
    chrom_subset_correct = []
    struct_subset_correct = []
    bias_subset_correct = []
    begin = end = 0
    for i in range(lengths_full.size):
        end += lengths_full[i]
        if chrom_full[i] in chrom_subset:
            lengths_subset_correct.append(lengths_full[i])
            chrom_subset_correct.append(chrom_full[i])
            struct_subset_correct.append(struct_full[begin:end, :])
            bias_subset_correct.append(bias_full[begin:end])
        begin = end
    lengths_subset_correct = np.array(lengths_subset_correct)
    chrom_subset_correct = np.array(chrom_subset_correct)
    bias_subset_correct = np.concatenate(bias_subset_correct)

    # If diploid, add the other homolog to struct_subset_correct
    if ploidy == 2:
        begin = end = 0
        for i in range(lengths_full.size):
            end += lengths_full[i]
            if chrom_full[i] in chrom_subset:
                struct_subset_correct.append(
                    struct_full[(begin + n):(end + n), :])
            begin = end
    struct_subset_correct = np.concatenate(struct_subset_correct)

    tmp = utils_poisson.subset_chrom_of_data(
        ploidy=ploidy, lengths_full=lengths_full, chrom_full=chrom_full,
        chrom_subset=chrom_subset, bias=bias_full, structures=struct_full)
    lengths_subset_test, chrom_subset_test, data_subset = tmp
    struct_subset_test = data_subset['struct']
    bias_subset_test = data_subset['bias']

    assert_array_equal(lengths_subset_correct, lengths_subset_test)
    assert_array_equal(chrom_subset_correct, chrom_subset_test)
    assert_array_equal(struct_subset_correct, struct_subset_test)
    assert_array_equal(bias_subset_correct, bias_subset_test)


@pytest.mark.parametrize("ambiguity", ["ua", "ambig", "pa"])
def test_subset_chrom_of_counts(ambiguity):
    ploidy = 2
    seed = 0
    lengths_full = np.array([35, 44, 16, 52, 73, 44])
    chrom_full = np.array(['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6'])
    chrom_subset = np.array(['chr6', 'chr1', 'chr2'])
    val = 10000  # Value for counts bins that will be excluded by function

    # Get subset of lengths & chrom names
    lengths_subset_correct = []
    chrom_subset_correct = []
    begin = end = 0
    for i in range(lengths_full.size):
        end += lengths_full[i]
        if chrom_full[i] in chrom_subset:
            lengths_subset_correct.append(lengths_full[i])
            chrom_subset_correct.append(chrom_full[i])
        begin = end
    lengths_subset_correct = np.array(lengths_subset_correct)
    chrom_subset_correct = np.array(chrom_subset_correct)

    random_state = np.random.RandomState(seed=seed)
    nbeads = lengths_full.sum() * ploidy
    counts = random_state.uniform(low=0.0, high=1.0, size=(nbeads, nbeads))
    begin1 = end1 = 0
    for i in range(lengths_full.size * ploidy):
        end1 += np.tile(lengths_full, ploidy)[i]
        chrom1 = np.tile(chrom_full, ploidy)[i]

        begin2 = end2 = 0
        for j in range(lengths_full.size * ploidy):
            end2 += np.tile(lengths_full, ploidy)[j]
            chrom2 = np.tile(chrom_full, ploidy)[j]
            if not (chrom1 in chrom_subset and chrom2 in chrom_subset):
                counts[begin1:end1, begin2:end2] = val
            begin2 = end2

        begin1 = end1
    counts_full = set_counts_ambiguity(
        counts, lengths=lengths_full, ploidy=ploidy, ambiguity=ambiguity)

    nbeads_subset = lengths_subset_correct.sum() * ploidy
    counts_subset_correct = set_counts_ambiguity(
        counts[counts != val].reshape(nbeads_subset, nbeads_subset),
        lengths=lengths_subset_correct, ploidy=ploidy,
        ambiguity=ambiguity).toarray()

    tmp = utils_poisson.subset_chrom_of_data(
        ploidy=ploidy, lengths_full=lengths_full, chrom_full=chrom_full,
        chrom_subset=chrom_subset, counts=counts_full)
    lengths_subset_test, chrom_subset_test, data_subset = tmp
    counts_subset_test = data_subset['counts'][0].toarray()

    assert_array_equal(lengths_subset_correct, lengths_subset_test)
    assert_array_equal(chrom_subset_correct, chrom_subset_test)
    assert_array_equal(counts_subset_correct, counts_subset_test)


@pytest.mark.parametrize("ambiguity", ["ua", "ambig", "pa"])
def test_intermol_counts(ambiguity):
    lengths = np.array([21, 34, 16])
    ploidy = 2
    seed = 0
    val = 10000  # Value for counts bins that will be excluded by function

    random_state = np.random.RandomState(seed=seed)
    nbeads = lengths.sum() * ploidy
    counts = random_state.uniform(low=0.0, high=1.0, size=(nbeads, nbeads))
    begin = end = 0
    for l in np.tile(lengths, ploidy):
        end += l
        counts[begin:end, begin:end] = val  # Set intra-molecular counts to val
        begin = end
    counts = set_counts_ambiguity(
        counts, lengths=lengths, ploidy=ploidy, ambiguity=ambiguity).toarray()
    counts[(counts != 0) & (counts % val == 0)] = val
    counts[counts > val] = 0

    counts_inter_correct = counts.copy()
    counts_inter_correct[counts_inter_correct == val] = 0

    counts_inter_test = utils_poisson._intermol_counts(
        counts, lengths_at_res=lengths, ploidy=ploidy).toarray()

    assert_array_equal(counts_inter_correct, counts_inter_test)


@pytest.mark.parametrize("ambiguity", ["ua", "ambig", "pa"])
def test_counts_near_diag(ambiguity):
    lengths = np.array([21, 34, 16])
    ploidy = 2
    val = 10000  # Value for counts bins that will be excluded by function

    nbeads = lengths.sum() * ploidy
    counts_diag1 = np.arange(1, nbeads)
    btwn_molecules = np.cumsum(np.tile(lengths, ploidy))[:-1] - 1
    counts_diag1[btwn_molecules] = val
    counts = np.full((nbeads, nbeads), val)
    row = np.arange(nbeads - 1)
    col = row + 1
    counts[row, col] = counts_diag1
    counts[col, row] = counts_diag1
    counts = set_counts_ambiguity(
        counts, lengths=lengths, ploidy=ploidy, ambiguity=ambiguity).toarray()
    counts[counts > val] -= val
    counts[counts > val] -= val
    counts[counts > val] -= val

    counts_nghbr_correct = counts.copy()
    counts_nghbr_correct[counts_nghbr_correct == val] = 0

    counts_nghbr_test = utils_poisson._counts_near_diag(
        counts, lengths_at_res=lengths, ploidy=ploidy, nbins=1).toarray()

    assert_array_equal(counts_nghbr_correct, counts_nghbr_test)
