import sys
import pytest
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, paired_distances
from numpy.testing import assert_allclose

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from pastis.optimization.utils_poisson import _setup_jax
    _setup_jax(debug_nan_inf=False)  # Set to True if needed for debugging

    from utils import get_counts, get_struct_randwalk, get_true_data_interchrom
    from utils import decrease_struct_res_correct
    from pastis.optimization import pastis_algorithms
    from pastis.optimization.utils_poisson import distance_between_homologs
    from pastis.optimization.constraints import _neighboring_bead_indices
    from pastis.optimization.multiscale_optimization import decrease_lengths_res


def compare_nghbr_dis(lengths, ploidy, multiscale_factor, struct_true,
                      struct_infer, mean_rtol=0.4, mean_atol=0, var_rtol=1,
                      var_atol=0.5):
    """Make sure distances between neighboring beads are accurate."""
    row_nghbr = _neighboring_bead_indices(
        lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor)
    row_nghbr = row_nghbr[np.isfinite(struct_infer[row_nghbr, 0]) & np.isfinite(
        struct_infer[row_nghbr + 1, 0])]
    struct_true_lowres = decrease_struct_res_correct(
        struct_true, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)
    nghbr_dis_correct = paired_distances(
        struct_true_lowres[row_nghbr], struct_true_lowres[row_nghbr + 1])
    nghbr_dis_ = paired_distances(
        struct_infer[row_nghbr], struct_infer[row_nghbr + 1])
    print(f"{np.median(nghbr_dis_correct)=:g}    {np.median(nghbr_dis_)=:g}")
    print(f"{nghbr_dis_correct.mean()=:g}    {nghbr_dis_.mean()=:g}")
    print(f"{nghbr_dis_correct.var()=:g}    {nghbr_dis_.var()=:g}")
    assert_allclose(
        nghbr_dis_correct.mean(), nghbr_dis_.mean(), rtol=mean_rtol,
        atol=mean_atol)
    assert_allclose(
        nghbr_dis_correct.var(), nghbr_dis_.var(), rtol=var_rtol, atol=var_atol)


def compare_hmlg_dis(lengths, ploidy, multiscale_factor, struct_true,
                     struct_infer, mean_rtol=0.25, mean_atol=0, var_rtol=1,
                     var_atol=1):
    """Check individual inter-homolog distances between all loci"""
    struct_true_lowres = decrease_struct_res_correct(
        struct_true, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)
    n = decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor).sum()
    mask = np.isfinite(struct_infer[:n, 0]) & np.isfinite(struct_infer[n:, 0])
    dis_interhmlg_correct = euclidean_distances(
        struct_true_lowres[:n][mask], struct_true_lowres[n:][mask])
    dis_interhmlg_ = euclidean_distances(
        struct_infer[:n][mask], struct_infer[n:][mask])
    print(f"{np.median(dis_interhmlg_correct)=:g}    {np.median(dis_interhmlg_)=:g}")
    print(f"{dis_interhmlg_correct.mean()=:g}    {dis_interhmlg_.mean()=:g}")
    print(f"{dis_interhmlg_correct.var()=:g}    {dis_interhmlg_.var()=:g}")
    assert_allclose(
        dis_interhmlg_correct.mean(), dis_interhmlg_.mean(), rtol=mean_rtol,
        atol=mean_atol)
    assert_allclose(
        dis_interhmlg_correct.var(), dis_interhmlg_.var(), rtol=var_rtol,
        atol=var_atol)


@pytest.mark.parametrize("multiscale_factor,multiscale_reform", [
    (1, True), (2, True), (4, True), (8, True), (2, False), (4, False),
    (8, False)])
def test_haploid(multiscale_factor, multiscale_reform):
    lengths = np.array([40])
    ploidy = 1
    seed = 0
    bcc_lambda = 0
    hsc_lambda = 0
    alpha, beta = -3, 1e3
    init = 'true'  # For convenience/speed

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity='ua', struct_nan=None, random_state=random_state,
        use_poisson=False)

    struct_, infer_param = pastis_algorithms.infer_at_alpha(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, beta=beta,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda,
        callback_freq={'print_freq': 0, 'log_freq': 0, 'save_freq': 0},
        struct_true=struct_true, multiscale_factor=multiscale_factor,
        multiscale_reform=multiscale_reform, verbose=False, init=init)

    assert infer_param['converged']


@pytest.mark.parametrize("multiscale_factor,multiscale_reform", [
    (1, True), (2, True), (4, True), (8, True), (2, False), (4, False),
    (8, False)])
def test_haploid_biased(multiscale_factor, multiscale_reform):
    lengths = np.array([40])
    ploidy = 1
    seed = 0
    bcc_lambda = 0
    hsc_lambda = 0
    alpha, beta = -3, 1e3
    init = 'true'  # For convenience/speed

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state)
    bias = 0.1 + random_state.uniform(size=lengths.sum())
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity='ua', struct_nan=None, random_state=random_state,
        use_poisson=False, bias=bias)

    struct_, infer_param = pastis_algorithms.infer_at_alpha(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, beta=beta,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda,
        callback_freq={'print_freq': 0, 'log_freq': 0, 'save_freq': 0},
        struct_true=struct_true, multiscale_factor=multiscale_factor,
        multiscale_reform=multiscale_reform, verbose=False, init=init, bias=bias)

    assert infer_param['converged']


@pytest.mark.parametrize("multiscale_rounds,multiscale_reform", [
    (1, True), (2, True), (2, False)])
def test_haploid_run_pastis(multiscale_rounds, multiscale_reform):
    lengths = np.array([40])
    ploidy = 1
    seed = 0
    bcc_lambda = 0
    hsc_lambda = 0
    alpha, beta = -3, 1e3
    init = 'true'  # For convenience/speed

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity='ua', struct_nan=None, random_state=random_state,
        use_poisson=False)

    struct_, infer_param = pastis_algorithms.pastis_poisson(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, beta=beta,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda,
        print_freq=None, log_freq=None, save_freq=None,
        struct_true=struct_true, multiscale_rounds=multiscale_rounds,
        multiscale_reform=multiscale_reform, verbose=False, init=init)

    assert infer_param['converged']


@pytest.mark.parametrize("multiscale_rounds,multiscale_reform", [
    (1, True), (2, True), (2, False)])
def test_haploid_infer_alpha(multiscale_rounds, multiscale_reform):
    lengths = np.array([40])
    ploidy = 1
    seed = 0
    bcc_lambda = 0
    hsc_lambda = 0
    alpha, beta = -3, 1e3
    init = 'true'  # For convenience/speed

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity='ua', struct_nan=None, random_state=random_state,
        use_poisson=False)

    struct_, infer_param = pastis_algorithms.pastis_poisson(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=None,
        seed=seed, normalize=False, filter_threshold=0, beta=beta,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda,
        print_freq=None, log_freq=None, save_freq=None,
        struct_true=struct_true, multiscale_rounds=multiscale_rounds,
        multiscale_reform=multiscale_reform, verbose=False, init=init)

    assert infer_param['converged']

    print(f"{alpha=:g}    {infer_param['alpha']=:g}")
    assert_allclose(alpha, infer_param['alpha'], rtol=0.1)


@pytest.mark.parametrize("ambiguity,multiscale_factor", [
    ('ua', 1), ('ambig', 1), ('pa', 1), ('ua', 2), ('ambig', 2), ('pa', 2),
    ('ua', 4), ('ambig', 4), ('pa', 4), ('ua', 8), ('ambig', 8), ('pa', 8)])
def test_diploid(ambiguity, multiscale_factor):
    lengths = np.array([30])
    ploidy = 2
    seed = 0
    bcc_lambda = 0
    hsc_lambda = 0
    alpha, beta = -3, 1e3
    true_interhmlg_dis = 10
    init = 'true'  # For convenience/speed

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state,
        true_interhmlg_dis=true_interhmlg_dis)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=None, random_state=random_state,
        use_poisson=True)

    struct_, infer_param = pastis_algorithms.infer_at_alpha(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, beta=beta,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda,
        callback_freq={'print_freq': 0, 'log_freq': 0, 'save_freq': 0},
        struct_true=struct_true, multiscale_factor=multiscale_factor,
        multiscale_reform=True, verbose=False, init=init)

    assert infer_param['converged']


def test_diploid_combo():
    lengths = np.array([20])
    ploidy = 2
    seed = 0
    bcc_lambda = 0
    hsc_lambda = 0
    alpha, beta = -3, 1e3
    ratio_ambig, ratio_pa, ratio_ua = 0.2, 0.7, 0.1
    init = 'true'  # For convenience/speed

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.uniform(size=(lengths.sum() * ploidy, 3))
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
        hsc_lambda=hsc_lambda, est_hmlg_sep=None,
        print_freq=None, log_freq=None, save_freq=None, verbose=False,
        struct_true=struct_true, init=init, beta=None)  # Test _set_initial_beta

    assert infer_param['converged']

    # Make sure estimated betas are appropriate given nreads per counts matrix
    infer_beta = np.array(infer_param['beta'])
    sim_ratio = np.array([ratio_ambig, ratio_pa, ratio_ua])
    assert_allclose(
        infer_beta / infer_beta.sum(), sim_ratio / sim_ratio.sum(), rtol=0.01)


def test_constraint_bcc2019():
    lengths = np.array([30])
    ploidy = 2
    seed = 0
    bcc_lambda = 1  # Update this as needed
    hsc_lambda = 0
    est_hmlg_sep = None
    alpha, beta = -3, 1e3
    true_interhmlg_dis = 10
    multiscale_factor = 1
    null = True  # If True, only optimize constraints, not main obj
    init = None

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state,
        true_interhmlg_dis=true_interhmlg_dis)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity="ua", struct_nan=None, random_state=random_state,
        use_poisson=False)

    print(f"{bcc_lambda=:g}    {hsc_lambda=:g}    {true_interhmlg_dis=:g}")
    struct_, infer_param = pastis_algorithms.pastis_poisson(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, bcc_lambda=bcc_lambda,
        hsc_lambda=hsc_lambda, est_hmlg_sep=est_hmlg_sep,
        print_freq=None, log_freq=None, save_freq=None,
        bcc_version="2019", hsc_version="2019", verbose=False, init=init,
        null=null, struct_true=struct_true, beta=beta)

    assert infer_param['converged']

    compare_nghbr_dis(
        lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor,
        struct_true=struct_true, struct_infer=struct_, mean_rtol=0.25,
        var_rtol=0.75)


def test_constraint_hsc2019_infer_hmlg_sep():
    lengths = np.array([40])
    ploidy = 2
    seed = 0
    bcc_lambda = 0
    hsc_lambda = 1e4  # Update this as needed
    true_interhmlg_dis = 15
    est_hmlg_sep = None  # Want to make sure can be inferred from unambig
    alpha, beta = -3, 1e3
    null = True  # If True, only optimize constraints, not main obj
    init = None

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state,
        true_interhmlg_dis=true_interhmlg_dis)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity="ua", struct_nan=None, random_state=random_state,
        use_poisson=False)

    print(f"{bcc_lambda=:g}    {hsc_lambda=:g}    {true_interhmlg_dis=:g}")
    struct_, infer_param = pastis_algorithms.pastis_poisson(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, beta=beta,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda, est_hmlg_sep=est_hmlg_sep,
        print_freq=None, log_freq=None, save_freq=None,
        struct_true=struct_true, bcc_version="2019", hsc_version="2019",
        multiscale_reform=True, verbose=False, init=init, null=null)

    assert infer_param['converged']

    infer_est_hmlg_sep = infer_param['est_hmlg_sep']
    interhmlg_dis = distance_between_homologs(
        struct_, lengths=lengths, multiscale_factor=1)
    print(f"hmlg sep: true={true_interhmlg_dis}, infer={infer_est_hmlg_sep}, "
          f"final={interhmlg_dis}")

    # Make sure inference of est_hmlg_sep yields an acceptable result
    assert_allclose(true_interhmlg_dis, infer_est_hmlg_sep, rtol=0.25)

    # Make sure inferred homologs are separated >= inferred est_hmlg_sep
    assert interhmlg_dis >= infer_est_hmlg_sep - 1e-6


@pytest.mark.parametrize("ambiguity,multiscale_factor", [
    ('ua', 1), ('ambig', 1), ('pa', 1), ('ua', 2), ('ambig', 2), ('pa', 2),
    ('ua', 4), ('ambig', 4), ('pa', 4), ('ua', 8), ('ambig', 8), ('pa', 8)])
def test_constraint_hsc2019(ambiguity, multiscale_factor):
    lengths = np.array([40])
    ploidy = 2
    seed = 0
    bcc_lambda = 1  # Can include to improve stability, update as needed
    hsc_lambda = 1e4  # Update this as needed
    true_interhmlg_dis = 15
    est_hmlg_sep = true_interhmlg_dis  # Using true value for convenience
    alpha, beta = -3, 1e3
    null = True  # If True, only optimize constraints, not main obj
    init = None

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state,
        true_interhmlg_dis=true_interhmlg_dis)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=None, random_state=random_state,
        use_poisson=False)

    print(f"{bcc_lambda=:g}    {hsc_lambda=:g}    {true_interhmlg_dis=:g}")
    struct_, infer_param = pastis_algorithms.infer_at_alpha(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, beta=beta,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda, est_hmlg_sep=est_hmlg_sep,
        callback_freq={'print_freq': 0, 'log_freq': 0, 'save_freq': 0},
        struct_true=struct_true, multiscale_factor=multiscale_factor,
        bcc_version="2019", hsc_version="2019", multiscale_reform=True,
        verbose=False, init=init, null=null)

    assert infer_param['converged']

    # Make sure inferred homologs are separated >= inferred est_hmlg_sep
    interhmlg_dis = distance_between_homologs(
        struct_, lengths=lengths, multiscale_factor=multiscale_factor)
    print(f"hmlg sep: true={true_interhmlg_dis},   infer={interhmlg_dis}")
    assert interhmlg_dis >= est_hmlg_sep - 1e-6


@pytest.mark.parametrize("ambiguity,multiscale_factor", [
    ('ua', 1), ('ambig', 1), ('pa', 1), ('ua', 2), ('ambig', 2), ('pa', 2),
    ('ua', 4), ('ambig', 4), ('pa', 4), ('ua', 8), ('ambig', 8), ('pa', 8)])
def test_constraint_bcc2022(ambiguity, multiscale_factor):
    lengths = np.array([40])
    ploidy = 2
    seed = 0
    alpha, beta = -3, 1e3
    true_interhmlg_dis = 15
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])
    use_poisson = True  # Must be true if including hsc2022
    bcc_lambda = 0.1  # Update this as needed
    hsc_lambda = 10  # Can include to improve stability, update as needed
    null = True  # If True, only optimize constraints, not main obj
    init = None

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
        multiscale_reform=True)

    print(f"{bcc_lambda=:g}    {hsc_lambda=:g}    {true_interhmlg_dis=:g}")
    struct_, infer_param = pastis_algorithms.infer_at_alpha(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, beta=beta,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda,
        callback_freq={'print_freq': 0, 'log_freq': 0, 'save_freq': 0},
        struct_true=struct_true, multiscale_factor=multiscale_factor,
        bcc_version="2022", hsc_version="2022", data_interchrom=data_interchrom,
        multiscale_reform=True, verbose=False, init=init, null=null, bias=bias)

    assert infer_param['converged']

    compare_nghbr_dis(
        lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor,
        struct_true=struct_true, struct_infer=struct_)


@pytest.mark.parametrize("ambiguity,multiscale_factor", [
    ('ua', 1), ('ambig', 1), ('pa', 1), ('ua', 2), ('ambig', 2), ('pa', 2),
    ('ua', 4), ('ambig', 4), ('pa', 4), ('ua', 8), ('ambig', 8), ('pa', 8)])
def test_constraint_bcc2022_biased(ambiguity, multiscale_factor):
    lengths = np.array([40])
    ploidy = 2
    seed = 0
    alpha, beta = -3, 1e3
    true_interhmlg_dis = 15
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])
    use_poisson = True  # Must be true if including hsc2022
    bcc_lambda = 0.1  # Update this as needed
    hsc_lambda = 10  # Can include to improve stability, update as needed
    null = True  # If True, only optimize constraints, not main obj
    init = None

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state,
        true_interhmlg_dis=true_interhmlg_dis)
    bias = 0.1 + random_state.uniform(size=lengths.sum())
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=struct_nan, random_state=random_state,
        use_poisson=use_poisson, bias=bias)

    data_interchrom = get_true_data_interchrom(
        struct_true=struct_true, ploidy=ploidy, lengths=lengths,
        ambiguity=ambiguity, struct_nan=struct_nan, alpha=alpha, beta=beta,
        bias=bias, random_state=random_state, use_poisson=use_poisson,
        multiscale_rounds=np.log2(multiscale_factor) + 1,
        multiscale_reform=True)

    print(f"{bcc_lambda=:g}    {hsc_lambda=:g}    {true_interhmlg_dis=:g}")
    struct_, infer_param = pastis_algorithms.infer_at_alpha(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, beta=beta,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda,
        callback_freq={'print_freq': 0, 'log_freq': 0, 'save_freq': 0},
        struct_true=struct_true, multiscale_factor=multiscale_factor,
        bcc_version="2022", hsc_version="2022", data_interchrom=data_interchrom,
        multiscale_reform=True, verbose=False, init=init, null=null, bias=bias)

    assert infer_param['converged']

    compare_nghbr_dis(
        lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor,
        struct_true=struct_true, struct_infer=struct_)


@pytest.mark.parametrize("ambiguity,multiscale_factor", [
    ('ua', 1), ('ambig', 1), ('pa', 1), ('ua', 2), ('ambig', 2), ('pa', 2),
    ('ua', 4), ('ambig', 4), ('pa', 4), ('ua', 8), ('ambig', 8), ('pa', 8)])
def test_constraint_hsc2022(ambiguity, multiscale_factor):
    lengths = np.array([40])
    ploidy = 2
    seed = 0
    alpha, beta = -3, 1e3
    true_interhmlg_dis = 15
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])
    use_poisson = True  # Must be true for hsc2022
    bcc_lambda = 0.1  # Can include to improve stability, update as needed
    hsc_lambda = 10  # Update this as needed
    null = True  # If True, only optimize constraints, not main obj
    init = None

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
        multiscale_reform=True)

    print(f"{bcc_lambda=:g}    {hsc_lambda=:g}    {true_interhmlg_dis=:g}")
    struct_, infer_param = pastis_algorithms.infer_at_alpha(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, beta=beta,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda,
        callback_freq={'print_freq': 0, 'log_freq': 0, 'save_freq': 0},
        struct_true=struct_true, multiscale_factor=multiscale_factor,
        bcc_version="2022", hsc_version="2022", data_interchrom=data_interchrom,
        multiscale_reform=True, verbose=False, init=init, null=null, bias=bias)

    assert infer_param['converged']

    # Check individual inter-homolog distances between all loci
    compare_hmlg_dis(
        lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor,
        struct_true=struct_true, struct_infer=struct_)

    # Make sure distance between homolog centers of mass isn't way off
    interhmlg_dis = distance_between_homologs(
        struct_, lengths=lengths, multiscale_factor=multiscale_factor)
    print(f"hmlg sep: true={true_interhmlg_dis},   infer={interhmlg_dis}")
    assert_allclose(true_interhmlg_dis, interhmlg_dis, rtol=0.35)


@pytest.mark.parametrize("ambiguity,multiscale_factor", [
    ('ua', 1), ('ambig', 1), ('pa', 1), ('ua', 2), ('ambig', 2), ('pa', 2),
    ('ua', 4), ('ambig', 4), ('pa', 4), ('ua', 8), ('ambig', 8), ('pa', 8)])
def test_constraint_hsc2022_biased(ambiguity, multiscale_factor):
    lengths = np.array([40])
    ploidy = 2
    seed = 0
    alpha, beta = -3, 1e3
    true_interhmlg_dis = 15
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])
    use_poisson = True  # Must be true for hsc2022
    bcc_lambda = 0.1  # Can include to improve stability, update as needed
    hsc_lambda = 10  # Update this as needed
    null = True  # If True, only optimize constraints, not main obj
    init = None

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state,
        true_interhmlg_dis=true_interhmlg_dis)
    bias = 0.1 + random_state.uniform(size=lengths.sum())
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=struct_nan, random_state=random_state,
        use_poisson=use_poisson, bias=bias)

    data_interchrom = get_true_data_interchrom(
        struct_true=struct_true, ploidy=ploidy, lengths=lengths,
        ambiguity=ambiguity, struct_nan=struct_nan, alpha=alpha, beta=beta,
        bias=bias, random_state=random_state, use_poisson=use_poisson,
        multiscale_rounds=np.log2(multiscale_factor) + 1,
        multiscale_reform=True)

    print(f"{bcc_lambda=:g}    {hsc_lambda=:g}    {true_interhmlg_dis=:g}")
    struct_, infer_param = pastis_algorithms.infer_at_alpha(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, beta=beta,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda,
        callback_freq={'print_freq': 0, 'log_freq': 0, 'save_freq': 0},
        struct_true=struct_true, multiscale_factor=multiscale_factor,
        bcc_version="2022", hsc_version="2022", data_interchrom=data_interchrom,
        multiscale_reform=True, verbose=False, init=init, null=null, bias=bias)

    assert infer_param['converged']

    # Check individual inter-homolog distances between all loci
    compare_hmlg_dis(
        lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor,
        struct_true=struct_true, struct_infer=struct_)

    # Make sure distance between homolog centers of mass isn't way off
    interhmlg_dis = distance_between_homologs(
        struct_, lengths=lengths, multiscale_factor=multiscale_factor)
    print(f"hmlg sep: true={true_interhmlg_dis},   infer={interhmlg_dis}")
    assert_allclose(true_interhmlg_dis, interhmlg_dis,
                    rtol=0.55)  # XXX higher than when bias=None...
