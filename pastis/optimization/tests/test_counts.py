import sys
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from utils import get_counts, ambiguate_counts_correct, get_struct_randwalk

    from pastis.optimization.utils_poisson import _setup_jax
    _setup_jax(traceback=True)

    import pastis.optimization.counts as counts_py
    from pastis.optimization.estimate_alpha_beta import _estimate_beta
    from pastis.optimization.utils_poisson import _dict_is_equal

    from topsy.utils.debug import print_array_non0  # TODO remove


def idx_isin_correct(idx1, idx2):
    """Whether each (row, col) pair in idx1 (row1, col1) is in idx2 (row2, col2)
    """

    if isinstance(idx1, (list, tuple)):
        idx1 = np.stack(idx1, axis=1)
    if isinstance(idx2, (list, tuple)):
        idx2 = np.stack(idx2, axis=1)
    return (idx1 == idx2[:, None]).all(axis=2).any(axis=0)


def compare_counts_bins_objects(bins, bins_correct):
    if bins_correct is None and bins is None:
        return

    if bins_correct is None:  # TODO remove
        print(bins.row)
        print(bins.col)
    if bins is None:
        print(bins_correct.row)
        print(bins_correct.col)

    assert bins_correct is not None
    assert bins is not None

    print(bins_correct.row)
    print(bins_correct.col); print()
    print(bins.row)
    print(bins.col); print('\n')
    assert_array_equal(bins_correct.row, bins.row)
    assert_array_equal(bins_correct.col, bins.col)

    # print_array_non0(bins_correct.data == bins.data)
    # print(bins_correct.data[:, 0])
    # print(bins.data[:, 0])

    if bins_correct.data is not None and bins.data is not None:
        assert_allclose(
            bins_correct.data.sum(axis=0), bins.data.sum(axis=0))
        assert_allclose(bins_correct.data, bins.data)
    else:
        assert bins_correct.data is None
        assert bins.data is None

    assert bins_correct.multiscale_factor == bins.multiscale_factor
    if bins_correct.multiscale_factor > 1:
        if bins_correct.mask is not None and bins.mask is not None:
            if bins_correct.mask.shape == bins.mask.shape:
                where_diff = np.where(bins_correct.mask != bins.mask)
                if where_diff[0].size > 0:
                    print('row', bins_correct.row[np.unique(where_diff[0])])
                    print('col', bins_correct.col[np.unique(where_diff[1])])
                    print_array_non0(bins_correct.mask[:10, :10]); print()
                    print_array_non0(bins.mask[:10, :10]); print('\ndiff:')
                    print_array_non0((bins_correct.mask != bins.mask)[:10, :10])
            else:
                print(f"{bins_correct.mask.shape=}, {bins.mask.shape=}... {bins_correct.data.shape=}")

            assert_array_equal(bins_correct.mask, bins.mask)
        else:
            assert bins_correct.mask is None
            assert bins.mask is None

    # Compare other attributes
    assert _dict_is_equal(bins_correct.__dict__, bins.__dict__, verbose=True)
    assert bins_correct == bins


def compare_counts_objects(counts, counts_correct):
    # Compare counts ndarrays
    counts_correct_arr = counts_correct.tocoo().toarray()
    counts_arr = counts.tocoo().toarray()
    counts_correct_non0 = np.invert(
        np.isnan(counts_correct_arr)) & (counts_correct_arr != 0)
    counts_non0 = np.invert(np.isnan(counts_arr)) & (counts_arr != 0)
    assert_array_equal(counts_correct_non0, counts_non0)
    assert_allclose(counts_correct_arr, counts_arr)

    # Compare CountsBins objects
    print("COMPARING NONZERO BINS")
    compare_counts_bins_objects(
        counts.bins_nonzero, bins_correct=counts_correct.bins_nonzero)
    print("COMPARING ZERO BINS")
    compare_counts_bins_objects(
        counts.bins_zero, bins_correct=counts_correct.bins_zero)

    # Compare other attributes
    assert _dict_is_equal(
        counts_correct.__dict__, counts.__dict__, verbose=True)
    assert counts_correct == counts


@pytest.mark.parametrize(
    "multiscale_factor,beta", [
        (1, 1), (1, 0.1), (1, 0.01),
        (2, 1), (2, 0.1), (2, 0.01),
        (4, 1), (4, 0.1), (4, 0.01),
        (8, 1), (8, 0.1), (8, 0.01)])
def test_add_counts_haploid(multiscale_factor, beta):
    lengths = np.array([10, 21])
    ploidy = 1
    seed = 0
    alpha = -3
    struct_nan1 = np.array([0, 1, 2, 3, 12, 15, 25])
    struct_nan2 = np.array([0, 1, 3, 12, 25])

    random_state = np.random.RandomState(seed=seed)
    struct_true1 = random_state.uniform(size=(lengths.sum() * ploidy, 3))
    counts1 = get_counts(
        struct_true1, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=None, struct_nan=struct_nan1, random_state=random_state,
        use_poisson=True).toarray()
    struct_true2 = random_state.uniform(size=(lengths.sum() * ploidy, 3))
    counts2 = get_counts(
        struct_true2, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=None, struct_nan=struct_nan2, random_state=random_state,
        use_poisson=True).toarray()

    # "True" summed counts: sum before converting to CountsMatrix
    true_counts_sum_object = counts_py._format_counts(
        counts=counts1 + counts2, lengths=lengths, ploidy=ploidy,
        beta=beta * 2, bias=None, exclude_zeros=False,
        multiscale_factor=multiscale_factor)[0]

    # Sum after converting to CountsMatrix
    counts_object1 = counts_py._format_counts(
        counts=counts1, lengths=lengths, ploidy=ploidy, beta=beta, bias=None,
        exclude_zeros=False, multiscale_factor=multiscale_factor)[0]
    counts_object2 = counts_py._format_counts(
        counts=counts2, lengths=lengths, ploidy=ploidy, beta=beta, bias=None,
        exclude_zeros=False, multiscale_factor=multiscale_factor)[0]
    counts_sum_object = sum([counts_object1, counts_object2])

    compare_counts_objects(
        counts_sum_object, counts_correct=true_counts_sum_object)


@pytest.mark.parametrize("ambiguity,multiscale_factor,beta", [
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
    seed = 0
    alpha = -3
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.uniform(size=(lengths.sum() * ploidy, 3))
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
        beta=beta_ambig, bias=None, exclude_zeros=False,
        multiscale_factor=multiscale_factor)[0]
    print_array_non0(true_counts_ambig_object.tocoo().toarray())

    # Ambiguate after converting to CountsMatrix
    counts_objects = counts_py._format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta, bias=None,
        exclude_zeros=False, multiscale_factor=multiscale_factor)
    counts_ambig_object = counts_py.ambiguate_counts(
        counts_objects, lengths=lengths, ploidy=ploidy)

    compare_counts_objects(
        counts_ambig_object, counts_correct=true_counts_ambig_object)


def test_3d_indices_haploid():
    lengths = np.array([20])
    ploidy = 1
    seed = 0
    alpha, beta = -3, 1
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.uniform(size=(lengths.sum() * ploidy, 3))
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity="ua", struct_nan=struct_nan, random_state=random_state,
        use_poisson=True)

    row3d, col3d = counts_py._counts_indices_to_3d_indices(
        counts, lengths_at_res=lengths, ploidy=ploidy,
        exclude_zeros=True)

    assert_array_equal(counts.row, row3d)
    assert_array_equal(counts.col, col3d)


def test_3d_indices_diploid_unambig():
    lengths = np.array([20])
    ploidy = 2
    seed = 0
    alpha, beta = -3, 1
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.uniform(size=(lengths.sum() * ploidy, 3))
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity="ua", struct_nan=struct_nan, random_state=random_state,
        use_poisson=True)

    row3d, col3d = counts_py._counts_indices_to_3d_indices(
        counts, lengths_at_res=lengths, ploidy=ploidy,
        exclude_zeros=True)

    assert_array_equal(counts.row, row3d)
    assert_array_equal(counts.col, col3d)


def test_3d_indices_diploid_ambig():
    lengths = np.array([20])
    ploidy = 2
    seed = 0
    alpha, beta = -3, 1
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.uniform(size=(lengths.sum() * ploidy, 3))
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity="ambig", struct_nan=struct_nan, random_state=random_state,
        use_poisson=True)

    row3d_test, col3d_test = counts_py._counts_indices_to_3d_indices(
        counts, lengths_at_res=lengths, ploidy=ploidy,
        exclude_zeros=True)

    n = lengths.sum()
    row3d_correct = np.concatenate(
        [np.tile(counts.row, 2), np.tile(counts.row, 2) + n])
    col3d_correct = np.tile(np.concatenate([counts.col, counts.col + n]), 2)
    assert_array_equal(row3d_correct, row3d_test)
    assert_array_equal(col3d_correct, col3d_test)


def test_3d_indices_diploid_partially_ambig():
    lengths = np.array([20])
    ploidy = 2
    seed = 0
    alpha, beta = -3, 1
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.uniform(size=(lengths.sum() * ploidy, 3))
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity="pa", struct_nan=struct_nan, random_state=random_state,
        use_poisson=True)

    row3d_test, col3d_test = counts_py._counts_indices_to_3d_indices(
        counts, lengths_at_res=lengths, ploidy=ploidy,
        exclude_zeros=True)

    n = lengths.sum()
    row3d_correct = np.tile(counts.row, 2)
    col3d_correct = np.concatenate([counts.col, counts.col + n])
    assert_array_equal(row3d_correct, row3d_test)
    assert_array_equal(col3d_correct, col3d_test)


@pytest.mark.parametrize("ambiguities,use_bias", [
    ("ua", False), ("ambig", False), ("pa", False),
    ("ua", True), ("ambig", True), ("pa", True),
    (["ua", "ambig"], False), (["ua", "ambig"], True),
    (["ua", "ambig", "pa"], False), (["ua", "ambig", "pa"], True)])
def test_ambiguate_beta(ambiguities, use_bias):
    lengths = np.array([20])
    ploidy = 2
    seed = 0
    alpha, beta_true = -3, 1
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.uniform(size=(lengths.sum() * ploidy, 3))
    if use_bias:
        bias = 0.1 + random_state.uniform(size=lengths.sum())
    else:
        bias = None
    if isinstance(ambiguities, str):
        ambiguities = [ambiguities]
    beta = [beta_true / len(ambiguities) for x in ambiguities]
    counts = []
    for i in range(len(ambiguities)):
        counts.append(get_counts(
            struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha,
            beta=beta[i], ambiguity=ambiguities[i], struct_nan=struct_nan,
            random_state=random_state, use_poisson=False, bias=bias))

    # Test _ambiguate_beta
    counts_ambig = ambiguate_counts_correct(
        counts, lengths=lengths, ploidy=ploidy)
    counts_ambig_objects, _, _ = counts_py.preprocess_counts(
        counts_ambig, lengths=lengths, ploidy=ploidy, beta=beta_true, bias=bias,
        verbose=False)
    beta_ambig_correct = _estimate_beta(
        struct_true, counts=counts_ambig_objects, alpha=alpha, lengths=lengths,
        ploidy=ploidy, bias=bias)[0]._value
    beta_ambig_test = counts_py._ambiguate_beta(
        beta, counts=counts, lengths=lengths, ploidy=ploidy)
    assert_allclose(beta_ambig_correct, beta_ambig_test)

    # Test _disambiguate_beta
    beta_disambig = counts_py._disambiguate_beta(
        beta_ambig_correct, counts=counts, lengths=lengths, ploidy=ploidy,
        bias=bias)
    assert_allclose(beta, beta_disambig)

    # Confirm that beta MLE is working
    counts_objects, _, _ = counts_py.preprocess_counts(
        counts, lengths=lengths, ploidy=ploidy, beta=beta, bias=bias,
        verbose=False)
    beta_mle = _estimate_beta(
        struct_true, counts=counts_objects, alpha=alpha, lengths=lengths,
        ploidy=ploidy, bias=bias)._value
    assert_allclose(beta, beta_mle)


@pytest.mark.parametrize("use_bias", [False, True])
def test_set_initial_beta(use_bias):
    lengths = np.array([40])
    ploidy = 2
    seed = 0
    alpha, beta = -3, 1
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])
    true_interhmlg_dis = 10

    random_state = np.random.RandomState(seed=seed)
    if use_bias:
        bias = 0.1 + random_state.uniform(size=lengths.sum())
    else:
        bias = None

    # Create 2 structures where distances between nghbr beads are always == 1
    struct_true1 = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state,
        true_interhmlg_dis=true_interhmlg_dis, noise=0)
    counts1 = get_counts(
        struct_true1, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        struct_nan=struct_nan, random_state=random_state, use_poisson=False,
        bias=bias, ambiguity="ambig")  # Counts get ambiguated later anyways
    struct_true2 = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state,
        true_interhmlg_dis=true_interhmlg_dis, noise=0)
    counts2 = get_counts(
        struct_true2, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        struct_nan=struct_nan, random_state=random_state, use_poisson=False,
        bias=bias, ambiguity="ambig")  # Counts get ambiguated later anyways

    # Get betas using only the distances between neighboring beads
    _, beta_nghbr1 = counts_py._set_initial_beta(
        counts1, lengths=lengths, ploidy=ploidy, bias=bias, exclude_zeros=False,
        neighboring_beads_only=True)
    _, beta_nghbr2 = counts_py._set_initial_beta(
        counts2, lengths=lengths, ploidy=ploidy, bias=bias, exclude_zeros=False,
        neighboring_beads_only=True)
    assert_allclose(beta_nghbr1, beta_nghbr2, rtol=1e-3)  # Approx eq


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_idx_isin(seed):
    high = 100
    num_total = 1000
    num_idx1 = 600
    num_idx2 = 600

    random_state = np.random.RandomState(seed=seed)
    arr = random_state.randint(low=0, high=high, size=(num_total, 2))
    idx1 = arr[random_state.choice(num_total, size=num_idx1, replace=False)]
    idx2 = arr[random_state.choice(num_total, size=num_idx2, replace=True)]

    correct = idx_isin_correct(idx1, idx2)
    test = counts_py._idx_isin(idx1, idx2)

    print(f'\nidx1... {idx1.shape=}'); print(idx1)
    print(f'\nidx2... {idx2.shape=}'); print(idx2)
    print(f'\ncorrect... {correct.shape}... {(~correct).sum()=}'); print(correct)
    print(f'\ntest... {test.shape}... {(~test).sum()=}'); print(test)
    print('\nidx1[test != correct]')
    print(np.stack([idx1[:, 0], idx1[:, 1], correct, test], axis=1)[test != correct])

    assert_array_equal(correct, test)
