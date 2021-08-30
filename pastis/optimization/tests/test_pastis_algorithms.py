import sys
import pytest
import numpy as np
from sklearn.metrics import euclidean_distances
from numpy.testing import assert_array_almost_equal, assert_allclose
from scipy import sparse

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from pastis.optimization import pastis_algorithms
    from pastis.optimization.constraints import _inter_homolog_dis


def test_pastis_poisson_haploid():
    lengths = np.array([20])
    ploidy = 1
    seed = 42
    bcc_lambda = 0
    hsc_lambda = 0
    hsc_r = None
    alpha, beta = -3., 1.

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    struct_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(struct_true)
    dis[dis == 0] = np.inf
    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = np.triu(counts, 1)
    counts = sparse.coo_matrix(counts)

    struct_, infer_var = pastis_algorithms.pastis_poisson(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, bcc_lambda=bcc_lambda,
        hsc_lambda=hsc_lambda, hsc_r=hsc_r, print_freq=None, history_freq=None,
        save_freq=None)

    assert infer_var['converged']


def test_pastis_poisson_diploid_unambig():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    bcc_lambda = 0
    hsc_lambda = 0
    hsc_r = None
    alpha, beta = -3., 1.

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    struct_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(struct_true)
    dis[dis == 0] = np.inf
    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = np.triu(counts, 1)
    counts = sparse.coo_matrix(counts)

    struct_, infer_var = pastis_algorithms.pastis_poisson(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, bcc_lambda=bcc_lambda,
        hsc_lambda=hsc_lambda, hsc_r=hsc_r, print_freq=None, history_freq=None,
        save_freq=None)

    assert infer_var['converged']


def test_pastis_poisson_diploid_ambig():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    bcc_lambda = 0
    hsc_lambda = 0
    hsc_r = None
    alpha, beta = -3., 1.

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    struct_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(struct_true)
    dis[dis == 0] = np.inf
    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = counts[:n, :n] + counts[n:, n:] + counts[:n, n:] + counts[n:, :n]
    counts = np.triu(counts, 1)
    counts = sparse.coo_matrix(counts)

    struct_, infer_var = pastis_algorithms.pastis_poisson(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, bcc_lambda=bcc_lambda,
        hsc_lambda=hsc_lambda, hsc_r=hsc_r, print_freq=None, history_freq=None,
        save_freq=None)

    assert infer_var['converged']

    # FIXME TODO couldn't we at least compare the distance error or something? these tests are too simplistic!


def test_pastis_poisson_diploid_partially_ambig():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    bcc_lambda = 0
    hsc_lambda = 0
    hsc_r = None
    alpha, beta = -3., 1.

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    struct_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(struct_true)
    dis[dis == 0] = np.inf
    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = counts[:, :n] + counts[:, n:]
    np.fill_diagonal(counts[:n, :], 0)
    np.fill_diagonal(counts[n:, :], 0)
    counts = sparse.coo_matrix(counts)

    struct_, infer_var = pastis_algorithms.pastis_poisson(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, bcc_lambda=bcc_lambda,
        hsc_lambda=hsc_lambda, hsc_r=hsc_r, print_freq=None, history_freq=None,
        save_freq=None)

    assert infer_var['converged']


def test_pastis_poisson_diploid_combo():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    bcc_lambda = 0
    hsc_lambda = 0
    hsc_r = None
    alpha, beta = -3., 1.
    ratio_ambig, ratio_pa, ratio_ua = 0.2, 0.7, 0.1

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    struct_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(struct_true)
    dis[dis == 0] = np.inf
    poisson_intensity = dis ** alpha

    ambig_counts = ratio_ambig * beta * poisson_intensity
    ambig_counts[np.isnan(ambig_counts) | np.isinf(ambig_counts)] = 0
    ambig_counts = ambig_counts[:n, :n] + ambig_counts[n:, n:] + ambig_counts[:n, n:] + ambig_counts[n:, :n]
    ambig_counts = np.triu(ambig_counts, 1)
    ambig_counts = sparse.coo_matrix(ambig_counts)

    pa_counts = ratio_pa * beta * poisson_intensity
    pa_counts[np.isnan(pa_counts) | np.isinf(pa_counts)] = 0
    pa_counts = pa_counts[:, :n] + pa_counts[:, n:]
    np.fill_diagonal(pa_counts[:n, :], 0)
    np.fill_diagonal(pa_counts[n:, :], 0)
    pa_counts = sparse.coo_matrix(pa_counts)

    ua_counts = ratio_ua * beta * poisson_intensity
    ua_counts[np.isnan(ua_counts) | np.isinf(ua_counts)] = 0
    ua_counts = np.triu(ua_counts, 1)
    ua_counts = sparse.coo_matrix(ua_counts)

    counts = [ambig_counts, pa_counts, ua_counts]

    struct_, infer_var = pastis_algorithms.pastis_poisson(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, bcc_lambda=bcc_lambda,
        hsc_lambda=hsc_lambda, hsc_r=hsc_r, print_freq=None, history_freq=None,
        save_freq=None)

    assert infer_var['converged']

    # Make sure estimated betas are appropriate given nreads per counts matrix
    infer_beta = np.array(infer_var['beta'])
    sim_ratio = np.array([ratio_ambig, ratio_pa, ratio_ua])
    assert_array_almost_equal(
        infer_beta / infer_beta.sum(), sim_ratio / sim_ratio.sum(), decimal=2)


def test_pastis_poisson_diploid_unambig_bcc_constraint():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    bcc_lambda = 1  # FIXME update
    hsc_lambda = 0
    hsc_r = None
    alpha, beta = -3., 1.

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    struct_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(struct_true)
    dis[dis == 0] = np.inf
    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = np.triu(counts, 1)
    counts = sparse.coo_matrix(counts)

    struct_, infer_var = pastis_algorithms.pastis_poisson(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, bcc_lambda=bcc_lambda,
        hsc_lambda=hsc_lambda, hsc_r=hsc_r, print_freq=None, history_freq=None,
        save_freq=None)

    assert infer_var['converged']


def test_pastis_poisson_diploid_unambig_hsc_constraint():
    lengths = np.array([40])
    ploidy = 2
    seed = 42
    bcc_lambda = 0
    hsc_lambda = 1e4  # FIXME update
    true_interhomo_dis = np.array([15.])
    hsc_r = None
    alpha, beta = -3., 1.
    multiscale_reform = True
    use_multiscale_variance = False
    max_attempt = 1e3

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()

    # Create structure, without any beads overlapping
    struct_true = None
    dis = np.zeros((n * ploidy, n * ploidy))
    attempt = 0
    while attempt < max_attempt and dis[
            np.triu_indices(dis.shape[0], 1)].min() <= 1e-6:
        attempt += 1

        # Make structure, very basic random walk
        struct_true = np.zeros((n * ploidy, 3), dtype=float)
        for i in range(struct_true.shape[0]):
            coord1 = random_state.choice([0, 1, 2])
            coord2 = random_state.choice([x for x in (0, 1, 2) if x != coord1])
            struct_true[i:, coord1] += random_state.choice([1, -1])
            struct_true[i:, coord2] += random_state.choice([1, -1]) * 0.5

        # Center homologs, so that distance between barycenters is 0
        begin = end = 0
        for i in range(len(lengths)):
            end += lengths[i]
            struct_true[n:][begin:end] -= struct_true[n:][begin:end].mean(axis=0)
            struct_true[:n][begin:end] -= struct_true[:n][begin:end].mean(axis=0)
            begin = end

        # Separate homologs
        begin = end = 0
        for i in range(len(lengths)):
            end += lengths[i]
            struct_true[begin:end, 0] += true_interhomo_dis[i]
            begin = end

        dis = euclidean_distances(struct_true)

    if attempt >= max_attempt:
        raise ValueError("Couldn't create struct without overlapping beads.")

    # Debugging stuff
    print(f'tries={attempt}, dis.min()='
          f'{dis[np.triu_indices(dis.shape[0], 1)].min():.3g}')
    begin = end = 0
    for i in range(len(lengths)):
        end += lengths[i]
        hmlg1 = struct_true[n:][begin:end]
        hmlg2 = struct_true[:n][begin:end]
        h1_rad = np.sqrt((np.square(hmlg1 - hmlg1.mean(axis=0))).sum(axis=1))
        h2_rad = np.sqrt((np.square(hmlg2 - hmlg2.mean(axis=0))).sum(axis=1))
        print(f'hmlg radiuses max: {h1_rad.max():.3g} & {h2_rad.max():.3g}')
        hmlg_sep = np.sqrt((np.square(hmlg1 - hmlg2)).sum(axis=1))
        print(f'hmlg sep: min={hmlg_sep.min():.3g}, max={hmlg_sep.max():.3g}')
        begin = end

    # Make counts
    dis[dis == 0] = np.inf
    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = np.triu(counts, 1)
    counts = sparse.coo_matrix(counts)

    struct_, infer_var = pastis_algorithms.pastis_poisson(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, beta=beta,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda, hsc_r=hsc_r,
        print_freq=None, history_freq=None, save_freq=None,
        struct_true=struct_true, multiscale_reform=multiscale_reform,
        use_multiscale_variance=use_multiscale_variance)
    infer_hsc_r = infer_var['hsc_r']
    interhomo_dis = _inter_homolog_dis(struct_, lengths=lengths)

    print(f"hmlg sep: true={true_interhomo_dis}, infer={infer_hsc_r}, "
          f"final={interhomo_dis}")

    assert infer_var['converged']

    # Make sure inference of hsc_r yields an acceptable result
    assert_array_almost_equal(true_interhomo_dis, infer_hsc_r, decimal=0)

    # Make sure inferred homologs are separated >= inferred hsc_r
    assert interhomo_dis >= infer_hsc_r - 1e-6
