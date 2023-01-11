import sys
import pytest
import numpy as np
from scipy import sparse
from sklearn.metrics import euclidean_distances
from numpy.testing import assert_array_almost_equal, assert_array_equal

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from utils import get_counts
    import pastis.optimization.counts as counts_py

    from topsy.utils.debug import print_array_non0  # TODO remove


def ambiguate_counts_correct(counts, lengths, ploidy):
    lengths = np.array(lengths, copy=False, ndmin=1, dtype=int)
    n = lengths.sum()
    if not isinstance(counts, list):
        counts = [counts]

    if len(counts) == 1 and (ploidy == 1 or counts[0].shape == (n, n)):
        return counts[0]

    counts_ambig = np.zeros((n, n))
    for c in counts:
        c = c.toarray()
        if c.shape[0] > c.shape[1]:
            c_ambig = np.sum(
                [c[:n, :], c[n:, :], c[:n, :].T, c[n:, :].T], axis=0)
        elif c.shape[0] < c.shape[1]:
            c_ambig = np.sum(
                [c[:, :n].T, c[:, n:].T, c[:, :n], c[:, n:]], axis=0)
        elif c.shape[0] == n:
            c_ambig = c
        else:
            c_ambig = np.sum(
                [c[:n, :n], c[:n, n:], c[:n, n:].T, c[n:, n:]], axis=0)
        counts_ambig += c_ambig

    return sparse.coo_matrix(np.triu(counts_ambig, 1))


def compare_counts_bins_objects(bins, bins_true):
    if bins_true is None and bins is None:
        return

    if bins_true is None:  # TODO remove
        print(bins.row)
        print(bins.col)
    if bins is None:
        print(bins_true.row)
        print(bins_true.col)

    assert bins_true is not None
    assert bins is not None

    print(bins_true.row)
    print(bins_true.col); print()
    print(bins.row)
    print(bins.col); print('\n')
    assert_array_equal(bins_true.row, bins.row)
    assert_array_equal(bins_true.col, bins.col)

    # print_array_non0(bins_true.data == bins.data)
    # print(bins_true.data[:, 0])
    # print(bins.data[:, 0])

    if bins_true.data is not None and bins.data is not None:
        assert_array_almost_equal(
            bins_true.data.sum(axis=0), bins.data.sum(axis=0))
        assert_array_almost_equal(bins_true.data, bins.data)
    else:
        assert bins_true.data is None
        assert bins.data is None

    assert bins_true.multiscale_factor == bins.multiscale_factor
    if bins_true.multiscale_factor > 1:
        if bins_true.mask is not None and bins.mask is not None:
            if bins_true.mask.shape == bins.mask.shape:
                where_diff = np.where(bins_true.mask != bins.mask)
                if where_diff[0].size > 0:
                    print('row', bins_true.row[np.unique(where_diff[0])])
                    print('col', bins_true.col[np.unique(where_diff[1])])
                    print_array_non0(bins_true.mask[:10, :10]); print()
                    print_array_non0(bins.mask[:10, :10]); print('\ndiff:')
                    print_array_non0((bins_true.mask != bins.mask)[:10, :10])
            else:
                print(f"{bins_true.mask.shape=}, {bins.mask.shape=}... {bins_true.data.shape=}")

            assert_array_equal(bins_true.mask, bins.mask)
        else:
            assert bins_true.mask is None
            assert bins.mask is None

    assert bins_true == bins


def compare_counts_objects(counts, counts_true):
    # Compare counts ndarrays
    counts_true_arr = counts_true.tocoo().toarray()
    counts_arr = counts.tocoo().toarray()
    counts_true_non0 = np.invert(
        np.isnan(counts_true_arr)) & (counts_true_arr != 0)
    counts_non0 = np.invert(np.isnan(counts_arr)) & (counts_arr != 0)
    assert_array_equal(counts_true_non0, counts_non0)
    assert_array_almost_equal(counts_true_arr, counts_arr)

    # Compare CountsBins objects
    print("COMPARING NONZERO BINS")
    compare_counts_bins_objects(
        counts.bins_nonzero, bins_true=counts_true.bins_nonzero)
    print("COMPARING ZERO BINS")
    compare_counts_bins_objects(
        counts.bins_zero, bins_true=counts_true.bins_zero)

    # Compare other attributes
    assert counts_true == counts


@pytest.mark.parametrize(
    "multiscale_factor,beta", [
        (1, 1), (1, 0.1), (1, 0.01),
        (2, 1), (2, 0.1), (2, 0.01),
        (4, 1), (4, 0.1), (4, 0.01),
        (8, 1), (8, 0.1), (8, 0.01)])
def test_add_counts_haploid(multiscale_factor, beta):
    lengths = np.array([10, 21])
    ploidy = 1
    seed = 42
    alpha = -3
    struct_nan1 = np.array([0, 1, 2, 3, 12, 15, 25])
    struct_nan2 = np.array([0, 1, 3, 12, 25])

    random_state = np.random.RandomState(seed=seed)
    struct_true1 = random_state.rand(lengths.sum() * ploidy, 3)
    counts1 = get_counts(
        struct_true1, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=None, struct_nan=struct_nan1, random_state=random_state,
        use_poisson=True).toarray()
    struct_true2 = random_state.rand(lengths.sum() * ploidy, 3)
    counts2 = get_counts(
        struct_true2, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=None, struct_nan=struct_nan2, random_state=random_state,
        use_poisson=True).toarray()

    # "True" summed counts: sum before converting to CountsMatrix
    true_counts_sum_object = counts_py._format_counts(
        counts=counts1 + counts2, lengths=lengths, ploidy=ploidy,
        beta=beta * 2, exclude_zeros=False,
        multiscale_factor=multiscale_factor)[0]

    # Sum after converting to CountsMatrix
    counts_object1 = counts_py._format_counts(
        counts=counts1, lengths=lengths, ploidy=ploidy, beta=beta,
        exclude_zeros=False, multiscale_factor=multiscale_factor)[0]
    counts_object2 = counts_py._format_counts(
        counts=counts2, lengths=lengths, ploidy=ploidy, beta=beta,
        exclude_zeros=False, multiscale_factor=multiscale_factor)[0]
    counts_sum_object = sum([counts_object1, counts_object2])

    compare_counts_objects(
        counts_sum_object, counts_true=true_counts_sum_object)


@pytest.mark.parametrize(
    "ambiguity,multiscale_factor,beta", [
        ('ambig', 1, 1), ('pa', 1, 1), ('ua', 1, 1),
        ('ambig', 2, 1), ('pa', 2, 1), ('ua', 2, 1),
        ('ambig', 4, 1), ('pa', 4, 1), ('ua', 4, 1),
        ('ambig', 8, 1), ('pa', 8, 1), ('ua', 8, 1),
        ('ambig', 1, 0.1), ('pa', 1, 0.1), ('ua', 1, 0.1),
        ('ambig', 2, 0.1), ('pa', 2, 0.1), ('ua', 2, 0.1),
        ('ambig', 4, 0.1), ('pa', 4, 0.1), ('ua', 4, 0.1),
        ('ambig', 8, 0.1), ('pa', 8, 0.1), ('ua', 8, 0.1),
        ('ambig', 1, 0.01), ('pa', 1, 0.01), ('ua', 1, 0.01),
        ('ambig', 2, 0.01), ('pa', 2, 0.01), ('ua', 2, 0.01),
        ('ambig', 4, 0.01), ('pa', 4, 0.01), ('ua', 4, 0.01),
        ('ambig', 8, 0.01), ('pa', 8, 0.01), ('ua', 8, 0.01)])
def test_ambiguate_counts(ambiguity, multiscale_factor, beta):
    lengths = np.array([10, 21])
    ploidy = 2
    seed = 42
    alpha = -3
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])  # NaN in both hmlgs
    struct_nan = np.append(struct_nan, struct_nan + lengths.sum())  # Both hmlgs
    struct_nan = np.append(struct_nan, [4, 5, 6, 7])  # NaN in one hmlg only

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.rand(lengths.sum() * ploidy, 3)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=struct_nan, random_state=random_state,
        use_poisson=True)

    # "True" ambiguated counts: ambiguate before converting to CountsMatrix
    true_counts_ambig_arr_fullres = ambiguate_counts_correct(
        counts, lengths=lengths, ploidy=ploidy)
    beta_ambig = counts_py._ambiguate_beta(
        beta, counts=counts, lengths=lengths, ploidy=ploidy)
    true_counts_ambig_object = counts_py._format_counts(
        counts=true_counts_ambig_arr_fullres, lengths=lengths, ploidy=ploidy,
        beta=beta_ambig, exclude_zeros=False,
        multiscale_factor=multiscale_factor)[0]
    print_array_non0(true_counts_ambig_object.tocoo().toarray())

    # Ambiguate after converting to CountsMatrix
    counts_objects = counts_py._format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta,
        exclude_zeros=False, multiscale_factor=multiscale_factor)
    counts_ambig_object = counts_py.ambiguate_counts(
        counts_objects, lengths=lengths, ploidy=ploidy)

    compare_counts_objects(
        counts_ambig_object, counts_true=true_counts_ambig_object)


def test_3d_indices_haploid():
    lengths = np.array([20])
    ploidy = 1
    seed = 42
    alpha, beta = -3, 1

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    struct_true = random_state.rand(n * ploidy, 3)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity="ua", struct_nan=None, random_state=random_state,
        use_poisson=False)

    row3d, col3d = counts_py._counts_indices_to_3d_indices(
        counts, lengths_at_res=lengths, ploidy=ploidy,
        exclude_zeros=True)

    assert np.array_equal(counts.row, row3d)
    assert np.array_equal(counts.col, col3d)


def test_3d_indices_diploid_unambig():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    alpha, beta = -3, 1

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    struct_true = random_state.rand(n * ploidy, 3)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity="ua", struct_nan=None, random_state=random_state,
        use_poisson=False)

    row3d, col3d = counts_py._counts_indices_to_3d_indices(
        counts, lengths_at_res=lengths, ploidy=ploidy,
        exclude_zeros=True)

    assert np.array_equal(counts.row, row3d)
    assert np.array_equal(counts.col, col3d)


def test_3d_indices_diploid_ambig():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    alpha, beta = -3, 1

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    struct_true = random_state.rand(n * ploidy, 3)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity="ambig", struct_nan=None, random_state=random_state,
        use_poisson=False)

    row3d, col3d = counts_py._counts_indices_to_3d_indices(
        counts, lengths_at_res=lengths, ploidy=ploidy,
        exclude_zeros=True)

    row3d_true = np.concatenate([np.tile(counts.row, 2), np.tile(counts.row, 2) + n])
    col3d_true = np.tile(np.concatenate([counts.col, counts.col + n]), 2)
    assert np.array_equal(row3d_true, row3d)
    assert np.array_equal(col3d_true, col3d)


def test_3d_indices_diploid_partially_ambig():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    alpha, beta = -3, 1

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    struct_true = random_state.rand(n * ploidy, 3)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity="pa", struct_nan=None, random_state=random_state,
        use_poisson=False)

    row3d, col3d = counts_py._counts_indices_to_3d_indices(
        counts, lengths_at_res=lengths, ploidy=ploidy,
        exclude_zeros=True)

    row3d_true = np.tile(counts.row, 2)
    col3d_true = np.concatenate([counts.col, counts.col + n])
    assert np.array_equal(row3d_true, row3d)
    assert np.array_equal(col3d_true, col3d)
