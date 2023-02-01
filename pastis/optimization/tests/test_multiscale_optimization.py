import sys
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from utils import get_counts, decrease_struct_res_correct
    from utils import decrease_counts_res_correct
    from utils import remove_struct_nan_from_counts

    from pastis.optimization import multiscale_optimization
    from pastis.optimization.counts import preprocess_counts


def get_struct_index_correct(multiscale_factor, lengths, ploidy):
    """Return full-res struct index grouped by the corresponding low-res bead.
    """

    lengths_lowres = multiscale_optimization.decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor)

    idx = np.arange(lengths_lowres.sum() * ploidy)
    idx = np.repeat(
        np.indices([multiscale_factor]), idx.shape[0]) + np.tile(
        idx * multiscale_factor, multiscale_factor)
    idx = idx.reshape(multiscale_factor, -1)

    # Figure out which rows / cols are out of bounds
    bins = np.tile(lengths, ploidy).cumsum()
    for i in range(lengths.shape[0] * ploidy):
        idx_binned = np.digitize(idx, bins)
        bad_idx = np.invert(np.equal(idx_binned, idx_binned.min(axis=0)))
        idx_mask = idx_binned.min(axis=0) == i
        vals = np.unique(
            idx[:, idx_mask][bad_idx[:, idx_mask]])
        for val in np.flip(vals, axis=0):
            idx[idx > val] -= 1
    bad_idx += idx >= lengths.sum() * ploidy

    # If a bin spills over chromosome / homolog boundaries, set it to whatever
    # It will get ignored later
    idx[bad_idx] = 0

    return idx, bad_idx


@pytest.mark.parametrize("multiscale_factor", [1, 2, 4, 8])
def test_decrease_lengths_res(multiscale_factor):
    lengths_lowres_correct = np.array([1, 2, 3, 4, 5])
    lengths_fullres = lengths_lowres_correct * multiscale_factor
    lengths_lowres_test = multiscale_optimization.decrease_lengths_res(
        lengths_fullres, multiscale_factor=multiscale_factor)
    assert_array_equal(lengths_lowres_correct, lengths_lowres_test)


def test_increase_struct_res():
    lengths = np.array([10, 21])
    multiscale_factor = 2
    ploidy = 2
    struct_nan_lowres = np.array([0, 4])

    coord0 = np.arange(lengths.sum() * ploidy, dtype=float).reshape(-1, 1)
    coord1 = coord2 = np.zeros_like(coord0)
    struct_highres_correct = np.concatenate([coord0, coord1, coord2], axis=1)

    struct_lowres = decrease_struct_res_correct(
        struct_highres_correct, multiscale_factor=multiscale_factor,
        lengths=lengths, ploidy=ploidy)

    # Set specified beads to NaN
    lengths_lowres = multiscale_optimization.decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor)
    struct_nan_lowres = struct_nan_lowres[
        struct_nan_lowres < lengths_lowres.sum()]
    struct_nan_lowres = np.append(
        struct_nan_lowres, struct_nan_lowres + lengths_lowres.sum())
    struct_lowres[struct_nan_lowres[
        struct_nan_lowres < struct_lowres.shape[0]], :] = np.nan

    struct_highres_test = multiscale_optimization.increase_struct_res(
        struct_lowres, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)
    assert_allclose(
        struct_highres_correct, struct_highres_test)


@pytest.mark.parametrize("ambiguity,multiscale_factor", [
    ('ua', 1), ('ambig', 1), ('pa', 1), ('ua', 2), ('ambig', 2), ('pa', 2),
    ('ua', 4), ('ambig', 4), ('pa', 4), ('ua', 8), ('ambig', 8), ('pa', 8)])
def test_decrease_counts_res(ambiguity, multiscale_factor):
    lengths = np.array([10, 21])
    ploidy = 2
    seed = 0
    alpha, beta = -3, 1
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.uniform(size=(lengths.sum() * ploidy, 3))
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=struct_nan, random_state=random_state,
        use_poisson=False)

    counts_lowres_correct = decrease_counts_res_correct(
        counts, multiscale_factor=multiscale_factor, lengths=lengths).toarray()
    counts_lowres_test = multiscale_optimization.decrease_counts_res(
        counts, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy).toarray()

    assert_array_equal(
        counts_lowres_correct != 0, counts_lowres_test != 0)
    assert_allclose(counts_lowres_correct, counts_lowres_test)


@pytest.mark.parametrize("multiscale_factor", [1, 2, 4, 8])
def test_get_struct_index(multiscale_factor):
    lengths = np.array([10, 21])
    ploidy = 2

    idx_correct, bad_idx_correct = get_struct_index_correct(
        multiscale_factor=multiscale_factor, lengths=lengths, ploidy=ploidy)
    idx_test, bad_idx_test = multiscale_optimization._get_struct_index(
        multiscale_factor=multiscale_factor, lengths=lengths, ploidy=ploidy)

    assert_array_equal(idx_correct, idx_test)
    assert_array_equal(bad_idx_correct, bad_idx_test)


@pytest.mark.parametrize("multiscale_factor", [1, 2, 4, 8])
def test_decrease_struct_res(multiscale_factor):
    lengths = np.array([10, 21])
    ploidy = 2
    seed = 0
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])

    random_state = np.random.RandomState(seed=seed)
    struct = random_state.uniform(size=(lengths.sum() * ploidy, 3))

    # Set specified beads to NaN
    struct_nan = struct_nan[struct_nan < lengths.sum()]
    struct_nan = np.append(struct_nan, struct_nan + lengths.sum())
    struct[struct_nan[struct_nan < struct.shape[0]], :] = np.nan

    struct_lowres_correct = decrease_struct_res_correct(
        struct, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)
    struct_lowres_test = multiscale_optimization.decrease_struct_res(
        struct, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)

    assert_allclose(
        struct_lowres_correct, struct_lowres_test)


@pytest.mark.parametrize("multiscale_factor", [2, 4, 8])
def test_get_epsilon_from_struct(multiscale_factor):
    lengths = np.array([10, 21])
    seed = 0
    ploidy = 2
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])

    random_state = np.random.RandomState(seed=seed)
    coord0 = random_state.uniform(size=(lengths.sum() * ploidy, 1))
    coord1 = coord2 = np.zeros_like(coord0)
    struct = np.concatenate([coord0, coord1, coord2], axis=1)

    # Set specified beads to NaN
    struct_nan = struct_nan[struct_nan < lengths.sum()]
    struct_nan = np.append(struct_nan, struct_nan + lengths.sum())
    struct[struct_nan[struct_nan < struct.shape[0]], :] = np.nan
    coord0[struct_nan[struct_nan < coord0.shape[0]], :] = np.nan

    multiscale_var_correct = []
    begin = end = 0
    for l in np.tile(lengths, ploidy):
        end += l
        for i in range(begin, end, multiscale_factor):
            slice = coord0[i:min(end, i + multiscale_factor)]
            beads_in_group = np.invert(np.isnan(slice)).sum()
            if beads_in_group < 1:
                var = np.nan
            else:
                var = np.var(slice[~np.isnan(slice)])
            multiscale_var_correct.append(var)
        begin = end
    multiscale_var_correct = np.array(multiscale_var_correct)
    epsilon_correct = np.sqrt(multiscale_var_correct * 2 / 3)

    epsilon_test = multiscale_optimization.get_epsilon_from_struct(
        struct, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, replace_nan=False, verbose=False)

    assert_allclose(np.mean(epsilon_correct), epsilon_test)


@pytest.mark.parametrize("min_beads", [5, 10, 11, 100, 101, 200])
def test__choose_max_multiscale_factor(min_beads):
    lengths = np.array([101])
    multiscale_factor = multiscale_optimization._choose_max_multiscale_factor(
        lengths, min_beads=min_beads)

    lengths_lowres = multiscale_optimization.decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor)
    lengths_lowres_toosmall = multiscale_optimization.decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor * 2)

    assert (min_beads <= lengths_lowres.min()) or (min_beads > lengths.min())
    assert min_beads > lengths_lowres_toosmall.min()


@pytest.mark.parametrize(
    "multiscale_factor,use_zero_counts",
    [(2, False), (4, False), (8, False), (2, True), (4, True), (8, True)])
def test_fullres_per_lowres_dis(multiscale_factor, use_zero_counts):
    lengths = np.array([101])
    ploidy = 2
    seed = 0
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()

    # Make counts
    counts = random_state.randint(low=0, high=10, size=(n, n))
    counts = np.triu(counts, 1)
    counts = remove_struct_nan_from_counts(
        counts, lengths=lengths, struct_nan=struct_nan)

    fullres_struct_nan_correct = struct_nan[struct_nan < lengths.sum()]
    if ploidy == 2:
        fullres_struct_nan_correct = np.append(struct_nan, struct_nan + n)

    counts, _, fullres_struct_nan_test = preprocess_counts(
        counts=counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, beta=1, bias=None, verbose=False,
        exclude_zeros=False)

    assert_array_equal(fullres_struct_nan_correct, fullres_struct_nan_test)

    fullres_per_lowres_bead = multiscale_optimization._count_fullres_per_lowres_bead(
        multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy, fullres_struct_nan=fullres_struct_nan_test)

    if use_zero_counts:
        counts_bins = counts[0].bins_zero
        if counts_bins is None:
            return
    else:
        counts_bins = counts[0].bins_nonzero

    correct = fullres_per_lowres_bead[counts_bins.row] * \
        fullres_per_lowres_bead[counts_bins.col]

    test = counts_bins.fullres_per_lowres_dis

    mask = correct != test
    row = counts_bins.row[mask]
    col = counts_bins.col[mask]
    tmp = np.stack([row, col, correct[mask], test[mask]], axis=1)
    print(fullres_per_lowres_bead)
    print(tmp)

    assert_array_equal(correct, test)


@pytest.mark.parametrize("multiscale_factor", [2, 4, 8])
def test_decrease_bias_res(multiscale_factor):
    lengths = np.array([100, 45, 21])
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])

    # Create full-res bias, set struct_nan to 0
    bias_fullres = np.arange(lengths.sum(), dtype=float)
    bias_fullres[struct_nan[struct_nan < lengths.sum()]] = 0

    # Get correct low-res bias
    tmp = np.tile(bias_fullres.reshape(-1, 1), (1, 3))
    tmp[tmp == 0] = np.nan
    bias_lowres_correct = decrease_struct_res_correct(
        tmp, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=1)[:, 0].ravel()
    bias_lowres_correct[np.isnan(bias_lowres_correct)] = 0

    bias_lowres_test = multiscale_optimization.decrease_bias_res(
        bias_fullres, multiscale_factor=multiscale_factor, lengths=lengths)
    assert_array_equal(bias_lowres_correct, bias_lowres_test)
