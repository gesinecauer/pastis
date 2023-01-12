import sys
import pytest
import numpy as np
from scipy.special import rel_entr
from numpy.testing import assert_array_almost_equal

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from utils import get_counts, get_struct_randwalk
    from utils import decrease_struct_res_correct, get_true_data_interchrom

    from pastis.optimization.utils_poisson import _setup_jax
    _setup_jax(debug_nan_inf=True)

    from pastis.optimization import constraints
    from pastis.optimization.counts import preprocess_counts
    from pastis.optimization.multiscale_optimization import get_epsilon_from_struct


@pytest.mark.parametrize("multiscale_factor", [1, 2, 4])
def test_constraint_bcc2019(multiscale_factor):
    lengths = np.array([20])
    ploidy = 2
    alpha, beta = -3, 1

    n = lengths.sum()
    struct_true = np.concatenate(
        [np.arange(n * ploidy).reshape(-1, 1), np.zeros((n * ploidy, 1)),
            np.zeros((n * ploidy, 1))], axis=1)
    counts_raw = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity="ua", struct_nan=None, random_state=None,
        use_poisson=False, bias=None)

    counts, _, fullres_struct_nan = preprocess_counts(
        counts_raw, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, beta=beta, verbose=False)

    struct_true_lowres = decrease_struct_res_correct(
        struct_true, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)

    constraint = constraints.prep_constraints(
        lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor,
        bcc_lambda=1, hsc_lambda=0, bcc_version='2019',
        fullres_struct_nan=fullres_struct_nan, verbose=False)[0]
    constraint.check()
    obj = constraint.apply(
        struct_true_lowres, alpha=alpha, epsilon=None, counts=counts,
        bias=None)._value
    assert np.isclose(obj, 0)


@pytest.mark.parametrize("multiscale_factor", [1, 2, 4, 8])
def test_constraint_hsc2019(multiscale_factor):
    lengths = np.array([30])
    ploidy = 2
    seed = 42
    true_interhmlg_dis = 15
    alpha, beta = -3, 1
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])
    est_hmlg_sep = true_interhmlg_dis  # Using true value for convenience

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state,
        true_interhmlg_dis=true_interhmlg_dis)
    counts_raw = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity="ua", struct_nan=struct_nan, random_state=random_state,
        use_poisson=False, bias=None)

    counts, _, fullres_struct_nan = preprocess_counts(
        counts_raw, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, beta=beta, verbose=False)

    struct_true_lowres = decrease_struct_res_correct(
        struct_true, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)

    constraint = constraints.prep_constraints(
        lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor,
        bcc_lambda=0, hsc_lambda=1, hsc_version='2019',
        est_hmlg_sep=est_hmlg_sep, hsc_perc_diff=None,
        fullres_struct_nan=fullres_struct_nan, verbose=False)[0]
    constraint.check()
    obj = constraint.apply(
        struct_true_lowres, alpha=alpha, epsilon=None, counts=counts,
        bias=None)._value
    assert np.isclose(obj, 0)


@pytest.mark.parametrize("ambiguity,multiscale_factor", [
    ('ua', 1), ('ambig', 1), ('pa', 1), ('ua', 2), ('ambig', 2), ('pa', 2),
    ('ua', 4), ('ambig', 4), ('pa', 4), ('ua', 8), ('ambig', 8), ('pa', 8)])
def test_constraint_bcc2022(ambiguity, multiscale_factor):
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    alpha, beta = -3, 1e3
    true_interhmlg_dis = 15

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state,
        true_interhmlg_dis=true_interhmlg_dis)
    counts_raw = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=None, random_state=None,
        use_poisson=True, bias=None)

    counts, _, fullres_struct_nan = preprocess_counts(
        counts_raw, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, beta=beta, verbose=False)

    struct_true_lowres = decrease_struct_res_correct(
        struct_true, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)
    if multiscale_factor == 1:
        epsilon_true = None
    else:
        epsilon_true = np.mean(get_epsilon_from_struct(
            struct_true, lengths=lengths, multiscale_factor=multiscale_factor,
            verbose=False))

    constraint = constraints.prep_constraints(
        lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor,
        bcc_lambda=1, hsc_lambda=0, bcc_version='2022', data_interchrom=None,
        fullres_struct_nan=fullres_struct_nan, verbose=False)[0]
    constraint.check()
    obj = constraint.apply(
        struct_true_lowres, alpha=alpha, epsilon=epsilon_true, counts=counts,
        bias=None)._value

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
    kl_test = constraints._kl_divergence(p, np.log(q))
    assert_array_almost_equal(kl_correct, kl_test)


@pytest.mark.parametrize("ambiguity,multiscale_factor", [
    ('ua', 1), ('ambig', 1), ('pa', 1), ('ua', 2), ('ambig', 2), ('pa', 2),
    ('ua', 4), ('ambig', 4), ('pa', 4), ('ua', 8), ('ambig', 8), ('pa', 8)])
def test_constraint_hsc2022(ambiguity, multiscale_factor):
    lengths = np.array([30])
    ploidy = 2
    seed = 42
    true_interhmlg_dis = 15
    alpha, beta = -3, 1
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])
    use_poisson = True  # Must be true for hsc2022

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state,
        true_interhmlg_dis=true_interhmlg_dis)
    counts_raw = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=struct_nan, random_state=random_state,
        use_poisson=use_poisson, bias=None)

    # # For convenience, we are using unambig inter-hmlg counts as data_interchrom
    # counts_unambig = get_counts(
    #     struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
    #     ambiguity="ua", struct_nan=struct_nan, random_state=random_state,
    #     use_poisson=use_poisson, bias=None)
    # data_interchrom = constraints.get_counts_interchrom(
    #     counts_unambig, lengths=np.tile(lengths, 2), ploidy=1,
    #     filter_threshold=0, normalize=False, bias=None, verbose=False)

    data_interchrom = get_true_data_interchrom(
        struct_true=struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha,
        beta=beta, random_state=random_state, use_poisson=use_poisson)

    counts, _, fullres_struct_nan = preprocess_counts(
        counts_raw, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, beta=beta, verbose=False)

    struct_true_lowres = decrease_struct_res_correct(
        struct_true, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)
    if multiscale_factor == 1:
        epsilon_true = None
    else:
        epsilon_true = np.mean(get_epsilon_from_struct(
            struct_true, lengths=lengths, multiscale_factor=multiscale_factor,
            verbose=False))

    constraint = constraints.prep_constraints(
        lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor,
        bcc_lambda=0, hsc_lambda=1, hsc_version='2022',
        data_interchrom=data_interchrom, fullres_struct_nan=fullres_struct_nan,
        verbose=False)[0]
    constraint.check()
    obj = constraint.apply(
        struct_true_lowres, alpha=alpha, epsilon=epsilon_true, counts=counts,
        bias=None)._value
    print(f"{obj=:g}")
    assert obj < 1e-3
