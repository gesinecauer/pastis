import sys
import pytest
import numpy as np
from sklearn.metrics import euclidean_distances
from numpy.testing import assert_array_almost_equal

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from utils import get_counts, get_struct_randwalk

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


def test_diploid_constraint_bcc2019():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    bcc_lambda = 1  # TODO update
    hsc_lambda = 0
    est_hmlg_sep = None
    alpha, beta = -3, 1
    ambiguity = "ua"

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
        hsc_lambda=hsc_lambda, est_hmlg_sep=est_hmlg_sep,
        print_freq=None, history_freq=None, save_freq=None)

    assert infer_param['converged']


def test_diploid_constraint_hsc2019_unambig():
    lengths = np.array([40])
    ploidy = 2
    seed = 42
    bcc_lambda = 0
    hsc_lambda = 1e4  # TODO update
    true_interhmlg_dis = np.array([15.])
    est_hmlg_sep = None
    alpha, beta = -3, 1
    ambiguity = "ua"

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state,
        true_interhmlg_dis=true_interhmlg_dis)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=None, random_state=random_state,
        use_poisson=False)

    struct_, infer_param = pastis_algorithms.pastis_poisson(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, beta=beta,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda, est_hmlg_sep=est_hmlg_sep,
        print_freq=None, history_freq=None, save_freq=None,
        struct_true=struct_true, multiscale_reform=True,
        use_multiscale_variance=False)

    assert infer_param['converged']

    infer_est_hmlg_sep = infer_param['est_hmlg_sep']
    interhmlg_dis = _inter_homolog_dis(struct_, lengths=lengths)

    print(f"hmlg sep: true={true_interhmlg_dis}, infer={infer_est_hmlg_sep}, "
          f"final={interhmlg_dis}")

    # Make sure inference of est_hmlg_sep yields an acceptable result
    assert_array_almost_equal(true_interhmlg_dis, infer_est_hmlg_sep, decimal=0)

    # Make sure inferred homologs are separated >= inferred est_hmlg_sep
    assert interhmlg_dis >= infer_est_hmlg_sep - 1e-6


@pytest.mark.parametrize("multiscale_factor", [1, 2, 4, 8])
def test_diploid_constraint_hsc2019_multiscale_unambig(multiscale_factor):
    lengths = np.array([40])
    ploidy = 2
    seed = 42

    bcc_lambda = 1  # FIXME !!! set to 0 !!! (do I really need to tho?)
    hsc_lambda = 1e4  # TODO update
    true_interhmlg_dis = np.array([1])  # FIXME !!! should be 15 or 5
    est_hmlg_sep = true_interhmlg_dis * 0.9  # FIXME shouldn't be *1
    alpha, beta = -3, 1
    ambiguity = "ua"

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
        multiscale_reform=True,
        use_multiscale_variance=False,
        init='mds')

    assert infer_param['converged']

    interhmlg_dis = _inter_homolog_dis(struct_, lengths=lengths)

    print(f"hmlg sep: true={true_interhmlg_dis}, est_hmlg_sep={est_hmlg_sep}, final={interhmlg_dis}")

    # Make sure inferred homologs are separated >= inferred est_hmlg_sep
    assert interhmlg_dis >= est_hmlg_sep - 1e-6

    # Make sure separation of inferred homologs is roughly accurate
    assert_array_almost_equal(true_interhmlg_dis, interhmlg_dis, decimal=0)


@pytest.mark.parametrize("multiscale_factor", [1, 2, 4, 8])
def test_diploid_constraint_hsc2019_multiscale_ambig(multiscale_factor):
    lengths = np.array([40])
    ploidy = 2
    seed = 42

    bcc_lambda = 1  # FIXME !!! set to 0 !!! (do I really need to tho?)
    hsc_lambda = 1e7  # TODO update
    true_interhmlg_dis = np.array([0.7])  # FIXME !!! should be 15 or 5
    est_hmlg_sep = true_interhmlg_dis * 1  # FIXME shouldn't be *1
    alpha, beta = -3, 1
    ambiguity = "ambig"

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
        multiscale_reform=True, use_multiscale_variance=False)

    assert infer_param['converged']

    interhmlg_dis = _inter_homolog_dis(struct_, lengths=lengths)

    print(f"hmlg sep: true={true_interhmlg_dis}, est_hmlg_sep={est_hmlg_sep}, final={interhmlg_dis}")

    # Make sure inferred homologs are separated >= inferred est_hmlg_sep
    assert interhmlg_dis >= est_hmlg_sep - 1e-4

    # Make sure separation of inferred homologs is roughly accurate
    assert_array_almost_equal(true_interhmlg_dis, interhmlg_dis, decimal=0)


# @pytest.mark.parametrize("multiscale_factor", [1, 2, 4, 8])
# def test_diploid_constraint_hsc2019_multiscale2_ambig(multiscale_factor):
#     from topsy.datasets.samples_generator import make_3d_genome  # FIXME

#     return ## FIXME

#     lengths = np.array([5])  # FIXME !!! 40
#     ploidy = 2
#     seed = 42

#     bcc_lambda = 1  # FIXME !!! set to 0 !!! (do I really need to tho?)
#     hsc_lambda = 10  # TODO update
#     true_interhmlg_dis = np.array([0.7])  # FIXME !!! should be 15 or 5
#     est_hmlg_sep = true_interhmlg_dis * 1  # FIXME shouldn't be *1
#     alpha, beta = -3, 1
#     ambiguity = "ambig"

#     random_state = np.random.RandomState(seed=seed)
#     n = lengths.sum()

#     # Create structure, without any beads overlapping
#     struct_true = make_3d_genome(
#         lengths, ploidy=ploidy, random_state=random_state,
#         distance_btwn_chrom=None, better_sim=True, verbose=False)

#     # Center homologs, so that distance between barycenters is 0
#     begin = end = 0
#     for i in range(len(lengths)):
#         end += lengths[i]
#         struct_true[n:][begin:end] -= struct_true[n:][begin:end].mean(axis=0)
#         struct_true[:n][begin:end] -= struct_true[:n][begin:end].mean(axis=0)
#         begin = end

#     # Separate homologs
#     begin = end = 0
#     for i in range(len(lengths)):
#         end += lengths[i]
#         struct_true[begin:end, 0] += true_interhmlg_dis[i]
#         begin = end

#     dis = euclidean_distances(struct_true)

#     # Debugging stuff
#     print(f'dis.min()='
#           f'{dis[np.triu_indices(dis.shape[0], 1)].min():.3g}')
#     begin = end = 0
#     for i in range(len(lengths)):
#         end += lengths[i]
#         hmlg1 = struct_true[n:][begin:end]
#         hmlg2 = struct_true[:n][begin:end]
#         h1_rad = np.sqrt((np.square(hmlg1 - hmlg1.mean(axis=0))).sum(axis=1))
#         h2_rad = np.sqrt((np.square(hmlg2 - hmlg2.mean(axis=0))).sum(axis=1))
#         print(f'{i}: radiuses max: {h1_rad.max():.3g} & {h2_rad.max():.3g}')
#         hmlg_sep = np.sqrt((np.square(hmlg1 - hmlg2)).sum(axis=1))
#         print(f'{i}: sep: min={hmlg_sep.min():.3g}, max={hmlg_sep.max():.3g}')
#         begin = end

#     counts = get_counts(
#         struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
#         ambiguity=ambiguity, struct_nan=None, random_state=random_state,
#         use_poisson=False)

#     struct_, infer_param = pastis_algorithms.infer_at_alpha(
#         counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
#         seed=seed, normalize=False, filter_threshold=0, beta=beta,
#         bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda, est_hmlg_sep=est_hmlg_sep,
#         callback_freq={'print': 0, 'history': 0, 'save': 0},
#         struct_true=struct_true, multiscale_factor=multiscale_factor,
#         multiscale_reform=True,
#         use_multiscale_variance=False)

#     assert infer_param['converged']

#     interhmlg_dis = _inter_homolog_dis(struct_, lengths=lengths)

#     print(f"hmlg sep: true={true_interhmlg_dis}, est_hmlg_sep={est_hmlg_sep}, final={interhmlg_dis}")

#     # Make sure inferred homologs are separated >= inferred est_hmlg_sep
#     assert interhmlg_dis >= est_hmlg_sep - 1e-6

#     # Make sure separation of inferred homologs is roughly accurate
#     assert_array_almost_equal(true_interhmlg_dis, interhmlg_dis, decimal=0)
