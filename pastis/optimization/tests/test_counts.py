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


def test_add_counts_haploid():
    pass


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
    struct_nan = np.append(struct_nan, struct_nan + lengths.sum())
    struct_nan = np.append(struct_nan, 4)  # Test asymmetry in struct_nan

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.rand(lengths.sum() * ploidy, 3)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=struct_nan, random_state=random_state,
        use_poisson=True)

    true_counts_ambig_arr_fullres = counts_py.ambiguate_counts(
        counts, lengths=lengths, ploidy=ploidy, exclude_zeros=True)
    true_counts_ambig_arr = decrease_counts_res(
        true_counts_ambig_arr_fullres, multiscale_factor=multiscale_factor,
        lengths=lengths, ploidy=ploidy).toarray()
    beta_ambig = counts_py._ambiguate_beta(
        beta, counts=counts, lengths=lengths, ploidy=ploidy)
    true_counts_ambig_objects = {c.name: c for c in counts_py._format_counts(
        counts=true_counts_ambig_arr_fullres, lengths=lengths, ploidy=ploidy,
        beta=beta_ambig, exclude_zeros=False,
        multiscale_factor=multiscale_factor)}

    # counts_ambig_objects = [c.ambiguate() for c in counts_py._format_counts(
    #     counts=counts, lengths=lengths, ploidy=ploidy, beta=beta,
    #     exclude_zeros=False, multiscale_factor=multiscale_factor)]
    # counts_ambig_objects = {
    #     c.name: c for c in counts_ambig_objects if c is not None}

    counts_objects = counts_py._format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta,
        exclude_zeros=False, multiscale_factor=multiscale_factor)
    counts_ambig_objects = {c.name: c for c in counts_py.ambiguate_counts(
        counts_objects, lengths=lengths, ploidy=ploidy, exclude_zeros=False)}
    counts_ambig_arr = [
        c.tocoo().toarray() for c in counts_ambig_objects.values(
        ) if c.sum() > 0][0]

    # print_array_non0(true_counts_ambig_arr); print()
    # print_array_non0(counts_ambig_arr); print('\n')
    # print_array_non0(true_counts_ambig_objects['ambig'].data); print()
    # print_array_non0(counts_ambig_objects['ambig'].data); print('\n')
    # print_array_non0(counts_ambig_objects['ambig0'].data); print('\n')

    true_counts_ambig_non0 = np.invert(
        np.isnan(true_counts_ambig_arr)) & (true_counts_ambig_arr != 0)
    counts_ambig_non0 = np.invert(
        np.isnan(counts_ambig_arr)) & (counts_ambig_arr != 0)
    assert_array_equal(true_counts_ambig_non0, counts_ambig_non0)

    assert_array_almost_equal(true_counts_ambig_arr, counts_ambig_arr)

    assert true_counts_ambig_objects.keys() == counts_ambig_objects.keys()
    for key in true_counts_ambig_objects.keys():
        print(key)
        print(true_counts_ambig_objects[key].row)
        print(true_counts_ambig_objects[key].col); print()
        print(counts_ambig_objects[key].row)
        print(counts_ambig_objects[key].col); print('\n')
        assert_array_equal(
            true_counts_ambig_objects[key].row, counts_ambig_objects[key].row)
        assert_array_equal(
            true_counts_ambig_objects[key].col, counts_ambig_objects[key].col)

        # print_array_non0(true_counts_ambig_objects[key].data == counts_ambig_objects[key].data)
        # print(true_counts_ambig_objects[key].data[:, 0])
        # print(counts_ambig_objects[key].data[:, 0])
        assert_array_almost_equal(
            true_counts_ambig_objects[key].data.sum(axis=0),
            counts_ambig_objects[key].data.sum(axis=0))

        assert_array_almost_equal(
            true_counts_ambig_objects[key].data, counts_ambig_objects[key].data)
        assert true_counts_ambig_objects[key] == counts_ambig_objects[key]
