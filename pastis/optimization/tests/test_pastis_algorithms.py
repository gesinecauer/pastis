import sys
import pytest
import numpy as np
from sklearn.metrics import euclidean_distances
from numpy.testing import assert_array_almost_equal

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from utils import get_counts, get_struct_randwalk, get_true_data_interchrom

    from pastis.optimization.utils_poisson import _setup_jax
    _setup_jax(debug_nan_inf=True)

    from pastis.optimization import pastis_algorithms
    from pastis.optimization.constraints import _inter_homolog_dis


def test_haploid():
    lengths = np.array([20])
    ploidy = 1
    seed = 42
    bcc_lambda = 0
    hsc_lambda = 0
    est_hmlg_sep = None
    alpha, beta = -3, 1

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    struct_true = random_state.rand(n * ploidy, 3)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity='ua', struct_nan=None, random_state=random_state,
        use_poisson=False)

    struct_, infer_param = pastis_algorithms.pastis_poisson(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, bcc_lambda=bcc_lambda,
        hsc_lambda=hsc_lambda, est_hmlg_sep=est_hmlg_sep, print_freq=None,
        history_freq=None, save_freq=None)

    assert infer_param['converged']


@pytest.mark.parametrize(
    "ambiguity", ["ua", "ambig", "pa"])
def test_diploid(ambiguity):
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    bcc_lambda = 0
    hsc_lambda = 0
    est_hmlg_sep = None
    alpha, beta = -3, 1

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    struct_true = random_state.rand(n * ploidy, 3)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=None, random_state=random_state,
        use_poisson=False)

    struct_, infer_param = pastis_algorithms.pastis_poisson(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, bcc_lambda=bcc_lambda,
        hsc_lambda=hsc_lambda, est_hmlg_sep=est_hmlg_sep, print_freq=None,
        history_freq=None, save_freq=None)

    assert infer_param['converged']
    # TODO couldn't we at least compare the distance error or something? these tests are too simplistic!


def test_diploid_combo():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    bcc_lambda = 0
    hsc_lambda = 0
    est_hmlg_sep = None
    alpha, beta = -3, 1
    ratio_ambig, ratio_pa, ratio_ua = 0.2, 0.7, 0.1

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    struct_true = random_state.rand(n * ploidy, 3)
    ambig_counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha,
        beta=beta * ratio_ambig, ambiguity="ambig", struct_nan=None,
        random_state=random_state, use_poisson=False)
    pa_counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha,
        beta=beta * ratio_pa, ambiguity="pa", struct_nan=None,
        random_state=random_state, use_poisson=False)
    ua_counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha,
        beta=beta * ratio_ua, ambiguity="ua", struct_nan=None,
        random_state=random_state, use_poisson=False)
    counts = [ambig_counts, pa_counts, ua_counts]

    struct_, infer_param = pastis_algorithms.pastis_poisson(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, bcc_lambda=bcc_lambda,
        hsc_lambda=hsc_lambda, est_hmlg_sep=est_hmlg_sep,
        print_freq=None, history_freq=None, save_freq=None)

    assert infer_param['converged']

    # Make sure estimated betas are appropriate given nreads per counts matrix
    infer_beta = np.array(infer_param['beta'])
    sim_ratio = np.array([ratio_ambig, ratio_pa, ratio_ua])
    assert_array_almost_equal(
        infer_beta / infer_beta.sum(), sim_ratio / sim_ratio.sum(), decimal=2)


def test_constraint_bcc2019():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    bcc_lambda = 1  # Update this as needed
    hsc_lambda = 0
    est_hmlg_sep = None
    alpha, beta = -3, 1

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    struct_true = random_state.rand(n * ploidy, 3)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity="ua", struct_nan=None, random_state=random_state,
        use_poisson=False)

    struct_, infer_param = pastis_algorithms.pastis_poisson(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, bcc_lambda=bcc_lambda,
        hsc_lambda=hsc_lambda, est_hmlg_sep=est_hmlg_sep,
        print_freq=None, history_freq=None, save_freq=None,
        bcc_version="2019", hsc_version="2019")

    assert infer_param['converged']

    nghbr_dis_true = np.diagonal(euclidean_distances(struct_true), 1)
    nghbr_dis_ = np.diagonal(euclidean_distances(struct_), 1)
    assert_array_almost_equal(
        nghbr_dis_true.mean(), nghbr_dis_.mean(), decimal=0)


def test_constraint_hsc2019_infer_hmlg_sep():
    lengths = np.array([40])
    ploidy = 2
    seed = 42
    bcc_lambda = 0
    hsc_lambda = 1e4  # Update this as needed
    true_interhmlg_dis = 15
    est_hmlg_sep = None  # Want to make sure can be inferred from unambig
    alpha, beta = -3, 1

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state,
        true_interhmlg_dis=true_interhmlg_dis)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity="ua", struct_nan=None, random_state=random_state,
        use_poisson=False)

    struct_, infer_param = pastis_algorithms.pastis_poisson(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, beta=beta,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda, est_hmlg_sep=est_hmlg_sep,
        print_freq=None, history_freq=None, save_freq=None,
        struct_true=struct_true, bcc_version="2019", hsc_version="2019",
        multiscale_reform=True, use_multiscale_variance=False)

    assert infer_param['converged']

    infer_est_hmlg_sep = infer_param['est_hmlg_sep']
    interhmlg_dis = _inter_homolog_dis(struct_, lengths=lengths)
    print(f"hmlg sep: true={true_interhmlg_dis}, infer={infer_est_hmlg_sep}, "
          f"final={interhmlg_dis}")

    # Make sure inference of est_hmlg_sep yields an acceptable result
    assert_array_almost_equal(true_interhmlg_dis, infer_est_hmlg_sep, decimal=0)

    # Make sure inferred homologs are separated >= inferred est_hmlg_sep
    assert interhmlg_dis >= infer_est_hmlg_sep - 1e-6


@pytest.mark.parametrize("ambiguity,multiscale_factor", [
    ('ua', 1), ('ambig', 1), ('pa', 1), ('ua', 2), ('ambig', 2), ('pa', 2),
    ('ua', 4), ('ambig', 4), ('pa', 4), ('ua', 8), ('ambig', 8), ('pa', 8)])
def test_constraint_hsc2019(ambiguity, multiscale_factor):
    lengths = np.array([40])
    ploidy = 2
    seed = 42
    bcc_lambda = 1  # Included to improve stability, update as needed
    hsc_lambda = 1e4  # Update this as needed
    true_interhmlg_dis = 15
    est_hmlg_sep = true_interhmlg_dis  # Using true value for convenience
    alpha, beta = -3, 1

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state,
        true_interhmlg_dis=true_interhmlg_dis)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=None, random_state=random_state,
        use_poisson=False)

    struct_, infer_param = pastis_algorithms.infer_at_alpha(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, beta=beta,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda, est_hmlg_sep=est_hmlg_sep,
        callback_freq={'print': 0, 'history': 0, 'save': 0},
        struct_true=struct_true, multiscale_factor=multiscale_factor,
        bcc_version="2019", hsc_version="2019", multiscale_reform=True,
        use_multiscale_variance=False)

    assert infer_param['converged']

    interhmlg_dis = _inter_homolog_dis(struct_, lengths=lengths)
    print(f"hmlg sep: true={true_interhmlg_dis}, final={interhmlg_dis}")

    # Make sure inferred homologs are separated >= inferred est_hmlg_sep
    assert interhmlg_dis >= est_hmlg_sep - 1e-6


@pytest.mark.parametrize("ambiguity,multiscale_factor", [
    ('ua', 1), ('ambig', 1), ('pa', 1), ('ua', 2), ('ambig', 2), ('pa', 2),
    ('ua', 4), ('ambig', 4), ('pa', 4), ('ua', 8), ('ambig', 8), ('pa', 8)])
def test_constraint_bcc2022(ambiguity, multiscale_factor):
    lengths = np.array([30])
    ploidy = 2
    seed = 42
    alpha, beta = -3, 1e3
    true_interhmlg_dis = 15
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])
    bcc_lambda = 10  # Update this as needed
    hsc_lambda = 1e3  # Included to improve stability, update as needed
    use_poisson = True

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state,
        true_interhmlg_dis=true_interhmlg_dis)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=struct_nan, random_state=None,
        use_poisson=use_poisson, bias=None)

    data_interchrom = get_true_data_interchrom(
        struct_true=struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha,
        beta=beta, random_state=random_state, use_poisson=use_poisson)

    struct_, infer_param = pastis_algorithms.infer_at_alpha(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, beta=beta,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda,
        callback_freq={'print': 0, 'history': 0, 'save': 0},
        struct_true=struct_true, multiscale_factor=multiscale_factor,
        bcc_version="2022", hsc_version="2022", data_interchrom=data_interchrom,
        multiscale_reform=True, use_multiscale_variance=False)

    assert infer_param['converged']

    nghbr_dis_true = np.diagonal(euclidean_distances(struct_true), 1)
    nghbr_dis_ = np.diagonal(euclidean_distances(struct_), 1)
    assert_array_almost_equal(
        nghbr_dis_true.mean(), nghbr_dis_.mean(), decimal=0)
    assert_array_almost_equal(
        nghbr_dis_true.var(), nghbr_dis_.var(), decimal=-1)


@pytest.mark.parametrize("ambiguity,multiscale_factor", [
    ('ua', 1), ('ambig', 1), ('pa', 1), ('ua', 2), ('ambig', 2), ('pa', 2),
    ('ua', 4), ('ambig', 4), ('pa', 4), ('ua', 8), ('ambig', 8), ('pa', 8)])
def test_constraint_hsc2022(ambiguity, multiscale_factor):
    lengths = np.array([30])
    ploidy = 2
    seed = 42
    true_interhmlg_dis = 10
    alpha, beta = -3, 1
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])
    bcc_lambda = 1
    hsc_lambda = 1e6  # Update this as needed
    use_poisson = True  # Must be true for hsc2022

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state,
        true_interhmlg_dis=true_interhmlg_dis)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=struct_nan, random_state=random_state,
        use_poisson=use_poisson, bias=None)

    data_interchrom = get_true_data_interchrom(
        struct_true=struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha,
        beta=beta, random_state=random_state, use_poisson=use_poisson)

    struct_, infer_param = pastis_algorithms.infer_at_alpha(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, beta=beta,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda,
        callback_freq={'print': 0, 'history': 0, 'save': 0},
        struct_true=struct_true, multiscale_factor=multiscale_factor,
        bcc_version="2022", hsc_version="2022", data_interchrom=data_interchrom,
        multiscale_reform=True, use_multiscale_variance=False)

    assert infer_param['converged']

    interhmlg_dis = _inter_homolog_dis(struct_, lengths=lengths)
    print(f"hmlg sep: true={true_interhmlg_dis}, final={interhmlg_dis}")

    # Make sure inference of est_hmlg_sep yields an acceptable result
    assert_array_almost_equal(true_interhmlg_dis, interhmlg_dis, decimal=0)
