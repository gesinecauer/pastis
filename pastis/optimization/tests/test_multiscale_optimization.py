import sys
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import sparse

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from utils import get_counts, decrease_struct_res_correct
    from utils import decrease_counts_res_correct

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
    lengths_lowres_true = np.array([1, 2, 3, 4, 5])
    lengths_fullres = lengths_lowres_true * multiscale_factor
    lengths_lowres = multiscale_optimization.decrease_lengths_res(
        lengths_fullres, multiscale_factor=multiscale_factor)
    assert_array_equal(lengths_lowres_true, lengths_lowres)


def test_increase_struct_res():
    lengths = np.array([10, 21])
    multiscale_factor = 2
    ploidy = 1
    struct_nan_lowres = np.array([0, 4])

    nbeads = lengths.sum() * ploidy
    coord0 = np.arange(nbeads, dtype=float).reshape(-1, 1)
    coord1 = coord2 = np.zeros_like(coord0)

    struct_highres_true = np.concatenate(
        [coord0, coord1, coord2], axis=1)

    struct_lowres = decrease_struct_res_correct(
        struct_highres_true, multiscale_factor=multiscale_factor,
        lengths=lengths, ploidy=ploidy)
    struct_lowres[struct_nan_lowres] = np.nan

    struct_highres = multiscale_optimization.increase_struct_res(
        struct_lowres, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)
    assert_array_almost_equal(
        struct_highres_true, struct_highres)


# def test_increase_struct_res_gaussian():  # TODO remove
#     lengths = np.array([10, 21])
#     ploidy = 1
#     current_multiscale_factor = 2
#     rescale_by = 2
#     seed = 0
#     final_multiscale_factor = int(current_multiscale_factor / rescale_by)

#     lengths_current = multiscale_optimization.decrease_lengths_res(
#         lengths=lengths, multiscale_factor=current_multiscale_factor)
#     lengths_final = multiscale_optimization.decrease_lengths_res(
#         lengths=lengths, multiscale_factor=final_multiscale_factor)

#     random_state = np.random.RandomState(seed=seed)
#     struct_current = random_state.rand(lengths_current.sum() * ploidy, 3)

#     struct_highres = multiscale_optimization.increase_struct_res_gaussian(
#         struct_current, current_multiscale_factor=current_multiscale_factor,
#         final_multiscale_factor=final_multiscale_factor, lengths=lengths,
#         ploidy=ploidy, std_dev=0.000001)

#     struct_lowres_new = multiscale_optimization.decrease_struct_res(
#         struct_highres, multiscale_factor=rescale_by, lengths=lengths_final,
#         ploidy=ploidy)
#     mask = np.invert(np.isnan(struct_current[:, 0]))
#     assert_array_almost_equal(
#         struct_current[mask], struct_lowres_new[mask], decimal=5)


@pytest.mark.parametrize("ambiguity,multiscale_factor", [
    ('ua', 1), ('ambig', 1), ('pa', 1), ('ua', 2), ('ambig', 2), ('pa', 2),
    ('ua', 4), ('ambig', 4), ('pa', 4), ('ua', 8), ('ambig', 8), ('pa', 8)])
def test_decrease_counts_res(ambiguity, multiscale_factor):
    lengths = np.array([10, 21])
    ploidy = 2
    seed = 42
    alpha, beta = -3, 1
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])
    struct_nan = np.append(struct_nan, struct_nan + lengths.sum())

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.rand(lengths.sum() * ploidy, 3)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=struct_nan, random_state=random_state,
        use_poisson=False)

    counts_lowres_true = decrease_counts_res_correct(
        counts, multiscale_factor=multiscale_factor, lengths=lengths).toarray()
    counts_lowres = multiscale_optimization.decrease_counts_res(
        counts, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy).toarray()

    assert_array_almost_equal(counts_lowres_true != 0, counts_lowres != 0)
    assert_array_almost_equal(counts_lowres_true, counts_lowres)


@pytest.mark.parametrize("multiscale_factor", [1, 2, 4, 8])
def test_get_struct_index(multiscale_factor):
    lengths = np.array([10, 21])
    ploidy = 2

    idx_true, bad_idx_true = get_struct_index_correct(
        multiscale_factor=multiscale_factor, lengths=lengths, ploidy=ploidy)
    idx, bad_idx = multiscale_optimization._get_struct_index(
        multiscale_factor=multiscale_factor, lengths=lengths, ploidy=ploidy)

    assert_array_equal(idx_true, idx)
    assert_array_equal(bad_idx_true, bad_idx)


@pytest.mark.parametrize("multiscale_factor", [1, 2, 4, 8])
def test_decrease_struct_res(multiscale_factor):
    lengths = np.array([10, 21])
    ploidy = 1
    seed = 42
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])

    nbeads = lengths.sum() * ploidy
    random_state = np.random.RandomState(seed=seed)
    struct = random_state.rand(nbeads, 3)
    struct[struct_nan] = np.nan

    struct_lowres_true = decrease_struct_res_correct(
        struct, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)
    struct_lowres = multiscale_optimization.decrease_struct_res(
        struct, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)

    assert_array_almost_equal(
        struct_lowres_true, struct_lowres)


@pytest.mark.parametrize("multiscale_factor", [2, 4, 8])
def test_get_multiscale_epsilon_from_struct(multiscale_factor):
    lengths = np.array([10, 21])
    seed = 42
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])

    nbeads = lengths.sum()
    random_state = np.random.RandomState(seed=seed)
    coord0 = random_state.rand(nbeads, 1)
    coord1 = coord2 = np.zeros_like(coord0)
    struct = np.concatenate([coord0, coord1, coord2], axis=1)
    coord0[struct_nan] = np.nan
    struct[struct_nan] = np.nan

    multiscale_var_true = []
    begin = end = 0
    for l in lengths:
        end += l
        for i in range(begin, end, multiscale_factor):
            slice = coord0[i:min(end, i + multiscale_factor)]
            beads_in_group = np.invert(np.isnan(slice)).sum()
            if beads_in_group < 1:
                var = np.nan
            else:
                var = np.var(slice[~np.isnan(slice)])
            multiscale_var_true.append(var)
        begin = end
    multiscale_var_true = np.array(multiscale_var_true)
    epsilon_true = np.sqrt(multiscale_var_true * 2 / 3)
    # multiscale_var_true[np.isnan(multiscale_var_true)] = np.nanmedian(
    #     multiscale_var_true)

    epsilon_infer = multiscale_optimization.get_multiscale_epsilon_from_struct(
        struct, lengths=lengths, multiscale_factor=multiscale_factor,
        replace_nan=False, verbose=False)

    assert_array_almost_equal(epsilon_true, epsilon_infer)


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
    lengths = np.array([100])
    ploidy = 2
    seed = 42
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()

    # Make counts
    counts_raw = random_state.randint(0, 10, size=(n, n))
    counts_raw = np.triu(counts_raw, 1)
    if struct_nan is not None:
        counts_raw[struct_nan, :] = 0
        counts_raw[:, struct_nan] = 0
    counts_raw = sparse.coo_matrix(counts_raw)

    fullres_struct_nan_true = struct_nan
    if ploidy == 2:
        fullres_struct_nan_true = np.append(struct_nan, struct_nan + n)

    counts, _, fullres_struct_nan = preprocess_counts(
        counts_raw=counts_raw, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, beta=1., verbose=False,
        exclude_zeros=False)

    assert_array_equal(fullres_struct_nan_true, fullres_struct_nan)

    fullres_per_lowres_bead = multiscale_optimization._count_fullres_per_lowres_bead(
        multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy, fullres_struct_nan=fullres_struct_nan)

    if use_zero_counts:
        counts_bins = counts[0].bins_zero
        if counts_bins is None:
            return
    else:
        counts_bins = counts[0].bins_nonzero

    correct = fullres_per_lowres_bead[counts_bins.row] * \
        fullres_per_lowres_bead[counts_bins.col]

    current = counts_bins.fullres_per_lowres_dis

    mask = correct != current
    row = counts_bins.row[mask]
    col = counts_bins.col[mask]
    tmp = np.stack([row, col, correct[mask], current[mask]], axis=1)
    print(fullres_per_lowres_bead)
    print(tmp)

    assert_array_equal(correct, current)


# def test_infer_multiscale_variances_ambig():  # TODO remove
#     lengths = np.array([160])
#     ploidy = 2
#     seed = 42
#     alpha, beta = -3, 1
#     multiscale_rounds = 4

#     multiscale_factor = 2 ** (multiscale_rounds - 1)
#     random_state = np.random.RandomState(seed=seed)
#     n = lengths.sum()
#     struct_true = random_state.rand(n * ploidy, 3)
#     counts = get_counts(
#         struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
#         ambiguity="ambig", struct_nan=None, random_state=random_state,
#         use_poisson=False)

#     multiscale_var_true = multiscale_optimization.get_multiscale_variances_from_struct(
#         struct_true, lengths=lengths, multiscale_factor=multiscale_factor)

#     struct_draft_fullres, _, _, _, draft_converged = pastis_algorithms._infer_draft(
#         counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
#         seed=seed, normalize=False, filter_threshold=0, beta=beta,
#         multiscale_rounds=multiscale_rounds, use_multiscale_variance=True,
#         hsc_lambda=0., est_hmlg_sep=None,
#         callback_freq={'print': None, 'history': None, 'save': None})

#     assert draft_converged

#     multiscale_variances_infer = multiscale_optimization.get_multiscale_variances_from_struct(
#         struct_draft_fullres, lengths=lengths,
#         multiscale_factor=multiscale_factor)

#     median_true = np.median(multiscale_var_true)
#     median_infer = np.median(multiscale_variances_infer)
#     assert_array_almost_equal(median_true, median_infer, decimal=1)
