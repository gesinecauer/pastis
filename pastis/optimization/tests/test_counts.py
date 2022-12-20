import sys
import pytest
import numpy as np
from sklearn.metrics import euclidean_distances
from scipy import sparse
from numpy.testing import assert_array_almost_equal, assert_array_equal

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from utils import get_counts
    import pastis.optimization.counts as counts_py
    from pastis.optimization.multiscale_optimization import decrease_counts_res

    from topsy.utils.debug import print_array_non0  # TODO remove


def compare_counts_bins_objects(bins, bins_true):
    if bins_true is None and bins is None:
        return
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
        where_diff = np.where(bins_true.mask != bins.mask)
        print('row', bins_true.row[np.unique(where_diff[1])])
        print('col', bins_true.col[np.unique(where_diff[1])])
        print_array_non0(bins_true.mask[:10, :10]); print()
        print_array_non0(bins.mask[:10, :10]); print('\ndiff:')
        print_array_non0((bins_true.mask != bins.mask)[:10, :10])

        assert_array_equal(bins_true.mask, bins.mask)

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


def test_add_counts_haploid():
    pass


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
    true_counts_ambig_arr_fullres = counts_py.ambiguate_counts(
        counts, lengths=lengths, ploidy=ploidy, exclude_zeros=True)
    true_counts_ambig_arr = decrease_counts_res(
        true_counts_ambig_arr_fullres, multiscale_factor=multiscale_factor,
        lengths=lengths, ploidy=ploidy).toarray()
    beta_ambig = counts_py._ambiguate_beta(
        beta, counts=counts, lengths=lengths, ploidy=ploidy)
    true_counts_ambig_object = counts_py._format_counts(
        counts=true_counts_ambig_arr_fullres, lengths=lengths, ploidy=ploidy,
        beta=beta_ambig, exclude_zeros=False,
        multiscale_factor=multiscale_factor)[0]

    # Ambiguate after converting to CountsMatrix
    counts_objects = counts_py._format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta,
        exclude_zeros=False, multiscale_factor=multiscale_factor)
    counts_ambig_object = counts_py.ambiguate_counts(
        counts_objects, lengths=lengths, ploidy=ploidy, exclude_zeros=False)
    counts_ambig_arr = counts_ambig_object.tocoo().toarray()

    # print_array_non0(counts); print()
    # print_array_non0(true_counts_ambig_arr); print()
    # print_array_non0(counts_ambig_arr); print('\n')
    # print_array_non0(true_counts_ambig_object['ambig'].data); print()
    # print_array_non0(counts_ambig_object['ambig'].data); print('\n')
    # print_array_non0(counts_ambig_object['ambig0'].data); print('\n')

    compare_counts_objects(
        counts_ambig_object, counts_true=true_counts_ambig_object)

