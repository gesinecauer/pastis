import sys
import pytest
import numpy as np
from scipy.special import rel_entr
from numpy.testing import assert_allclose

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from utils import get_counts, get_struct_randwalk
    from utils import decrease_struct_res_correct, get_true_data_interchrom

    from pastis.optimization.utils_poisson import _setup_jax
    _setup_jax(traceback=True, debug_nan_inf=True)

    from pastis.optimization import constraints
    from pastis.optimization.counts import preprocess_counts
    from pastis.optimization.multiscale_optimization import get_epsilon_from_struct
    from pastis.optimization.multiscale_optimization import decrease_lengths_res


@pytest.mark.parametrize("multiscale_factor", [1, 2, 4, 8])
def test_neighboring_bead_indices(multiscale_factor):
    lengths = np.array([30])
    ploidy = 2

    lengths_lowres = decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor)
    row_nghbr = np.arange(lengths_lowres.sum() * ploidy - 1)
    row_nghbr = row_nghbr[~np.isin(
        row_nghbr, np.cumsum(np.tile(lengths_lowres, ploidy)) - 1)]

    row_nghbr_true = constraints._neighboring_bead_indices(
        lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor)

    print('row_nghbr\n', row_nghbr)
    print('row_nghbr_true\n', row_nghbr_true)
    print(row_nghbr[~np.isin(row_nghbr, row_nghbr_true)])
    print(row_nghbr_true[~np.isin(row_nghbr_true, row_nghbr)])

    assert np.array_equal(row_nghbr_true, row_nghbr)


@pytest.mark.parametrize("multiscale_factor", [1, 2, 4])
def test_constraint_bcc2019(multiscale_factor):
    lengths = np.array([20])
    ploidy = 2
    alpha, beta = -3, 1
    multiscale_reform = True

    n = lengths.sum()
    struct_true = np.concatenate(
        [np.arange(n * ploidy).reshape(-1, 1), np.zeros((n * ploidy, 1)),
            np.zeros((n * ploidy, 1))], axis=1)
    bias = None
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity="ua", struct_nan=None, random_state=None,
        use_poisson=False, bias=bias)

    counts, _, fullres_struct_nan = preprocess_counts(
        counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, beta=beta, bias=bias,
        multiscale_reform=multiscale_reform, verbose=False)

    struct_true_lowres = decrease_struct_res_correct(
        struct_true, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)

    constraint = constraints.prep_constraints(
        lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor,
        multiscale_reform=multiscale_reform, bcc_lambda=1, hsc_lambda=0,
        bcc_version='2019', fullres_struct_nan=fullres_struct_nan,
        verbose=False)[0]
    constraint.check()
    obj = constraint.apply(
        struct_true_lowres, alpha=alpha, beta=beta, epsilon=None, counts=counts,
        bias=bias)._value
    assert np.isclose(obj, 0)


@pytest.mark.parametrize("multiscale_factor", [1, 2, 4, 8])
def test_constraint_hsc2019(multiscale_factor):
    lengths = np.array([30])
    ploidy = 2
    seed = 0
    true_interhmlg_dis = 15
    alpha, beta = -3, 1
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])
    est_hmlg_sep = true_interhmlg_dis  # Using true value for convenience
    multiscale_reform = True

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state,
        true_interhmlg_dis=true_interhmlg_dis)
    bias = None
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity="ua", struct_nan=struct_nan, random_state=random_state,
        use_poisson=False, bias=bias)

    counts, _, fullres_struct_nan = preprocess_counts(
        counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, beta=beta, bias=bias,
        multiscale_reform=multiscale_reform, verbose=False)

    struct_true_lowres = decrease_struct_res_correct(
        struct_true, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)

    constraint = constraints.prep_constraints(
        lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor,
        multiscale_reform=multiscale_reform, bcc_lambda=0, hsc_lambda=1,
        hsc_version='2019', est_hmlg_sep=est_hmlg_sep, hsc_perc_diff=None,
        fullres_struct_nan=fullres_struct_nan, verbose=False)[0]
    constraint.check()
    obj = constraint.apply(
        struct_true_lowres, alpha=alpha, beta=beta, epsilon=None, counts=counts,
        bias=bias)._value
    assert np.isclose(obj, 0)


@pytest.mark.parametrize("ambiguity,multiscale_factor,multiscale_reform", [
    ('ua', 1, True), ('ambig', 1, True), ('pa', 1, True),
    ('ua', 2, True), ('ambig', 2, True), ('pa', 2, True),
    ('ua', 4, True), ('ambig', 4, True), ('pa', 4, True),
    ('ua', 8, True), ('ambig', 8, True), ('pa', 8, True),
    ('ua', 2, False), ('ambig', 2, False), ('pa', 2, False),
    ('ua', 4, False), ('ambig', 4, False), ('pa', 4, False),
    ('ua', 8, False), ('ambig', 8, False), ('pa', 8, False)])
def test_constraint_bcc2022(ambiguity, multiscale_factor, multiscale_reform):
    lengths = np.array([41])
    ploidy = 2
    seed = 0
    alpha, beta = -3, 1e3
    struct_nan = None
    true_interhmlg_dis = 15

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state,
        true_interhmlg_dis=true_interhmlg_dis)
    bias = None
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=struct_nan, random_state=None,
        use_poisson=True, bias=bias)

    counts, _, fullres_struct_nan = preprocess_counts(
        counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, beta=beta, bias=bias,
        multiscale_reform=multiscale_reform, verbose=False)

    struct_true_lowres = decrease_struct_res_correct(
        struct_true, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)
    if multiscale_factor == 1 or (not multiscale_reform):
        epsilon_true = None
    else:
        epsilon_true, _, _ = get_epsilon_from_struct(
            struct_true, lengths=lengths, ploidy=ploidy,
            multiscale_factor=multiscale_factor, verbose=False)

    constraint = constraints.prep_constraints(
        lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor,
        multiscale_reform=multiscale_reform, bcc_lambda=1, hsc_lambda=0,
        bcc_version='2022', data_interchrom=None,
        fullres_struct_nan=fullres_struct_nan, verbose=False)[0]
    constraint.check()
    obj = constraint.apply(
        struct_true_lowres, alpha=alpha, beta=beta, epsilon=epsilon_true,
        counts=counts, bias=bias)._value

    nbins_nghbr = constraint._var['row_nghbr'].size
    assert obj < (-1e4 / nbins_nghbr)


@pytest.mark.parametrize("p,q", [
    ([0.10, 0.40, 0.50], [0.80, 0.15, 0.05]),
    ([0.80, 0.15, 0.05], [0.10, 0.40, 0.50]),
    (np.arange(5) / 10, np.arange(5) / 10)])
def test_kl_divergence(p, q):
    p = np.array(p, ndmin=1, copy=False)
    q = np.array(q, ndmin=1, copy=False)

    kl_correct = rel_entr(p, q).sum()
    kl_test = constraints._kl_divergence(p, np.log(q), mean=False)
    assert_allclose(kl_correct, kl_test)


@pytest.mark.parametrize("ambiguity,multiscale_factor,multiscale_reform", [
    ('ua', 1, True), ('ambig', 1, True), ('pa', 1, True),
    ('ua', 2, True), ('ambig', 2, True), ('pa', 2, True),
    ('ua', 4, True), ('ambig', 4, True), ('pa', 4, True),
    ('ua', 8, True), ('ambig', 8, True), ('pa', 8, True),
    ('ua', 2, False), ('ambig', 2, False), ('pa', 2, False),
    ('ua', 4, False), ('ambig', 4, False), ('pa', 4, False),
    ('ua', 8, False), ('ambig', 8, False), ('pa', 8, False)])
def test_constraint_hsc2022(ambiguity, multiscale_factor, multiscale_reform):
    lengths = np.array([41])
    ploidy = 2
    seed = 0
    true_interhmlg_dis = 15
    alpha, beta = -3, 1e3
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])
    use_poisson = True  # Must be true for hsc2022

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state,
        true_interhmlg_dis=true_interhmlg_dis)
    bias = None
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=struct_nan, random_state=random_state,
        use_poisson=use_poisson, bias=bias)

    data_interchrom = get_true_data_interchrom(
        struct_true=struct_true, ploidy=ploidy, lengths=lengths,
        ambiguity=ambiguity, struct_nan=struct_nan, alpha=alpha, beta=beta,
        bias=bias, random_state=random_state, use_poisson=use_poisson,
        multiscale_rounds=np.log2(multiscale_factor) + 1,
        multiscale_reform=multiscale_reform)

    counts, _, fullres_struct_nan = preprocess_counts(
        counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, beta=beta, bias=bias,
        multiscale_reform=multiscale_reform, verbose=False)

    struct_true_lowres = decrease_struct_res_correct(
        struct_true, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)
    if multiscale_factor == 1 or (not multiscale_reform):
        epsilon_true = None
    else:
        epsilon_true, _, _ = get_epsilon_from_struct(
            struct_true, lengths=lengths, ploidy=ploidy,
            multiscale_factor=multiscale_factor, verbose=False)

    constraint = constraints.prep_constraints(
        lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor,
        multiscale_reform=multiscale_reform, bcc_lambda=0, hsc_lambda=1,
        hsc_version='2022', data_interchrom=data_interchrom,
        fullres_struct_nan=fullres_struct_nan, verbose=False)[0]
    constraint.check()
    obj = constraint.apply(
        struct_true_lowres, alpha=alpha, beta=beta, epsilon=epsilon_true,
        counts=counts, bias=bias)._value
    print(f"{obj=:g}")

    if multiscale_factor > 1 and (not multiscale_reform):  # Multires naive
        # The variance of the estimated distribution is too low in the naive
        # setting, so the KL divergence will be higher, depending on resolution
        assert obj < np.square(multiscale_factor) / 10
    else:  # Singleres or negbinom multires model
        assert obj < 0.1
