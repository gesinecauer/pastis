import sys
import pytest
import numpy as np
from sklearn.metrics import euclidean_distances
from scipy import sparse
from numpy.testing import assert_array_almost_equal, assert_array_equal

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    import pastis.optimization.counts as counts_py
    from pastis.optimization.multiscale_optimization import decrease_counts_res

    from topsy.utils.debug import print_array_non0  # TODO remove


@pytest.mark.parametrize(
    "ambiguity,multiscale_factor",
    [('ambig', 1), ('pa', 1), ('ua', 1), ('ambig', 2), ('pa', 2), ('ua', 2),
     ('ambig', 4), ('pa', 4), ('ua', 4), ('ambig', 8), ('pa', 8), ('ua', 8)])
def test_ambiguate_counts(ambiguity, multiscale_factor):
    if ambiguity not in ('ambig', 'pa', 'ua'):
        raise ValueError(f"Ambiguity not understood: {ambiguity}")
    lengths = np.array([10, 21])
    ploidy = 2
    seed = 42
    alpha, beta = -3, 0.1
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    struct_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(struct_true)
    dis[dis == 0] = np.inf

    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    if ambiguity == 'ambig':
        counts = counts[:n, :n] + counts[
            n:, n:] + counts[:n, n:] + counts[n:, :n]
        counts = np.triu(counts, 1)
    elif ambiguity == 'pa':
        counts = counts[:, :n] + counts[:, n:]
        np.fill_diagonal(counts[:n, :], 0)
        np.fill_diagonal(counts[n:, :], 0)
    elif ambiguity == 'ua':
        counts = np.triu(counts, 1)
    if ploidy == 2:
        struct_nan_tmp = np.concatenate([struct_nan, struct_nan + n])
    else:
        struct_nan_tmp = struct_nan
    counts[struct_nan_tmp[struct_nan_tmp < counts.shape[0]], :] = 0
    counts[:, struct_nan_tmp[struct_nan_tmp < counts.shape[1]]] = 0
    counts = random_state.poisson(counts)
    counts = sparse.coo_matrix(counts)

    true_counts_ambig_arr_fullres = counts_py.ambiguate_counts(
        counts, lengths=lengths, ploidy=ploidy, exclude_zeros=True)
    true_counts_ambig_arr = decrease_counts_res(
        true_counts_ambig_arr_fullres, multiscale_factor=multiscale_factor,
        lengths=lengths, ploidy=ploidy).toarray()
    true_counts_ambig_object = {c.name: c for c in counts_py._format_counts(
        counts=true_counts_ambig_arr_fullres, lengths=lengths, ploidy=ploidy,
        beta=beta, exclude_zeros=False, multiscale_factor=multiscale_factor)}

    counts_ambig_object = [c.ambiguate() for c in counts_py._format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta,
        exclude_zeros=False, multiscale_factor=multiscale_factor)]
    counts_ambig_object = {
        c.name: c for c in counts_ambig_object if c is not None}
    counts_ambig_arr = [
        c.tocoo().toarray() for c in counts_ambig_object.values(
        ) if c.sum() > 0][0]

    # print_array_non0(true_counts_ambig_arr); print()
    # print_array_non0(counts_ambig_arr); print('\n')

    # print_array_non0(true_counts_ambig_object['ambig'].data); print()
    # print_array_non0(counts_ambig_object['ambig'].data); print('\n')
    # print_array_non0(counts_ambig_object['ambig0'].data); print('\n')

    true_counts_ambig_non0 = np.invert(
        np.isnan(true_counts_ambig_arr)) & (true_counts_ambig_arr != 0)
    counts_ambig_non0 = np.invert(
        np.isnan(counts_ambig_arr)) & (counts_ambig_arr != 0)
    assert_array_equal(true_counts_ambig_non0, counts_ambig_non0)

    assert_array_almost_equal(true_counts_ambig_arr, counts_ambig_arr)

    assert true_counts_ambig_object.keys() == counts_ambig_object.keys()
    for key in true_counts_ambig_object.keys():
        print(key)
        print(true_counts_ambig_object[key].row)
        print(true_counts_ambig_object[key].col); print()
        print(counts_ambig_object[key].row)
        print(counts_ambig_object[key].col); print('\n')
        assert_array_equal(
            true_counts_ambig_object[key].row, counts_ambig_object[key].row)
        assert_array_equal(
            true_counts_ambig_object[key].col, counts_ambig_object[key].col)

        # print_array_non0(true_counts_ambig_object[key].data == counts_ambig_object[key].data)
        # print(true_counts_ambig_object[key].data[:, 0])
        # print(counts_ambig_object[key].data[:, 0])
        assert_array_almost_equal(
            true_counts_ambig_object[key].data.sum(axis=0),
            counts_ambig_object[key].data.sum(axis=0))

        assert_array_almost_equal(
            true_counts_ambig_object[key].data, counts_ambig_object[key].data)
        assert true_counts_ambig_object[key] == counts_ambig_object[key]
