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
    est_hmlg_sep = None
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
        hsc_lambda=hsc_lambda, est_hmlg_sep=est_hmlg_sep, print_freq=None,
        history_freq=None, save_freq=None)

    assert infer_var['converged']


def test_pastis_poisson_diploid_unambig():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    bcc_lambda = 0
    hsc_lambda = 0
    est_hmlg_sep = None
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
        hsc_lambda=hsc_lambda, est_hmlg_sep=est_hmlg_sep, print_freq=None,
        history_freq=None, save_freq=None)

    assert infer_var['converged']


def test_pastis_poisson_diploid_ambig():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    bcc_lambda = 0
    hsc_lambda = 0
    est_hmlg_sep = None
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
        hsc_lambda=hsc_lambda, est_hmlg_sep=est_hmlg_sep, print_freq=None,
        history_freq=None, save_freq=None)

    assert infer_var['converged']

    # TODO couldn't we at least compare the distance error or something? these tests are too simplistic!


def test_pastis_poisson_diploid_partially_ambig():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    bcc_lambda = 0
    hsc_lambda = 0
    est_hmlg_sep = None
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
        hsc_lambda=hsc_lambda, est_hmlg_sep=est_hmlg_sep,
        print_freq=None, history_freq=None, save_freq=None)

    assert infer_var['converged']


def test_pastis_poisson_diploid_combo():
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    bcc_lambda = 0
    hsc_lambda = 0
    est_hmlg_sep = None
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
    ambig_counts = ambig_counts[:n, :n] + ambig_counts[n:, n:] + \
        ambig_counts[:n, n:] + ambig_counts[n:, :n]
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
        hsc_lambda=hsc_lambda, est_hmlg_sep=est_hmlg_sep,
        print_freq=None, history_freq=None, save_freq=None)

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
    bcc_lambda = 1  # TODO update
    hsc_lambda = 0
    est_hmlg_sep = None
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
        hsc_lambda=hsc_lambda, est_hmlg_sep=est_hmlg_sep,
        print_freq=None, history_freq=None, save_freq=None)

    assert infer_var['converged']


def test_pastis_poisson_diploid_unambig_hsc_constraint():
    lengths = np.array([40])
    ploidy = 2
    seed = 42
    bcc_lambda = 0
    hsc_lambda = 1e4  # TODO update
    true_interhomo_dis = np.array([15.])
    est_hmlg_sep = None
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
            # FIXME update random walk based on below

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
        print(f'{i}: radiuses max: {h1_rad.max():.3g} & {h2_rad.max():.3g}')
        hmlg_sep = np.sqrt((np.square(hmlg1 - hmlg2)).sum(axis=1))
        print(f'{i}: sep: min={hmlg_sep.min():.3g}, max={hmlg_sep.max():.3g}')
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
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda, est_hmlg_sep=est_hmlg_sep,
        print_freq=None, history_freq=None, save_freq=None,
        struct_true=struct_true, multiscale_reform=multiscale_reform,
        use_multiscale_variance=use_multiscale_variance)

    assert infer_var['converged']

    infer_est_hmlg_sep = infer_var['est_hmlg_sep']
    interhomo_dis = _inter_homolog_dis(struct_, lengths=lengths)

    print(f"hmlg sep: true={true_interhomo_dis}, infer={infer_est_hmlg_sep}, "
          f"final={interhomo_dis}")

    # Make sure inference of est_hmlg_sep yields an acceptable result
    assert_array_almost_equal(true_interhomo_dis, infer_est_hmlg_sep, decimal=0)

    # Make sure inferred homologs are separated >= inferred est_hmlg_sep
    assert interhomo_dis >= infer_est_hmlg_sep - 1e-6


@pytest.mark.parametrize("multiscale_factor", [1, 2, 4, 8])
def test_pastis_poisson_diploid_unambig_hsc_constraint_multiscale(multiscale_factor):
    lengths = np.array([40])  # FIXME !!! 40
    ploidy = 2
    seed = 42

    bcc_lambda = 1  # FIXME !!! set to 0 !!!
    hsc_lambda = 1e4  # TODO update
    true_interhomo_dis = np.array([1])  # FIXME !!! 15 or 5
    est_hmlg_sep = true_interhomo_dis * 0.9  # FIXME
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
            struct_true[i:, coord1] += random_state.choice(
                [1, -1]) * random_state.normal(scale=0.1)
            struct_true[i:, coord2] += random_state.choice(
                [1, -1]) * 0.5 * random_state.normal(scale=0.1)

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
        print(f'{i}: radiuses max: {h1_rad.max():.3g} & {h2_rad.max():.3g}')
        hmlg_sep = np.sqrt((np.square(hmlg1 - hmlg2)).sum(axis=1))
        print(f'{i}: sep: min={hmlg_sep.min():.3g}, max={hmlg_sep.max():.3g}')
        begin = end

    # Make counts (unambiguous)
    dis[dis == 0] = np.inf
    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = np.triu(counts, 1)
    counts = sparse.coo_matrix(counts)

    struct_init = 'mds'
    # # Make struct_init
    # struct_init = struct_true.copy()
    # # init_interhomo_dis = true_interhomo_dis - (-9.344022901380811e-05)
    # init_interhomo_dis = true_interhomo_dis + 1e-10
    # # Center homologs, so that distance between barycenters is 0
    # begin = end = 0
    # for i in range(len(lengths)):
    #     end += lengths[i]
    #     struct_init[n:][begin:end] -= struct_init[n:][begin:end].mean(axis=0)
    #     struct_init[:n][begin:end] -= struct_init[:n][begin:end].mean(axis=0)
    #     begin = end
    # # Separate homologs
    # begin = end = 0
    # for i in range(len(lengths)):
    #     end += lengths[i]
    #     struct_init[begin:end, 0] += init_interhomo_dis[i]
    #     begin = end
    # print(_inter_homolog_dis(struct_init, lengths=lengths))

    callback_freq = {'print': 0, 'history': 0, 'save': 0}
    struct_, infer_var = pastis_algorithms.infer(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, beta=beta,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda, est_hmlg_sep=est_hmlg_sep,
        callback_freq=callback_freq, struct_true=struct_true,
        multiscale_factor=multiscale_factor,
        multiscale_reform=multiscale_reform,
        use_multiscale_variance=use_multiscale_variance,
        init=struct_init)  # FIXME

    assert infer_var['converged']

    interhomo_dis = _inter_homolog_dis(struct_, lengths=lengths)

    print(f"hmlg sep: true={true_interhomo_dis}, est_hmlg_sep={est_hmlg_sep}, final={interhomo_dis}")

    # Make sure inferred homologs are separated >= inferred est_hmlg_sep
    assert interhomo_dis >= est_hmlg_sep - 1e-6

    # Make sure separation of inferred homologs is roughly accurate
    assert_array_almost_equal(true_interhomo_dis, interhomo_dis, decimal=0)


@pytest.mark.parametrize("multiscale_factor", [1, 2, 4, 8])
def test_pastis_poisson_diploid_ambig_hsc_constraint_multiscale(multiscale_factor):
    lengths = np.array([40])  # FIXME !!! 40
    ploidy = 2
    seed = 42

    bcc_lambda = 1  # FIXME !!! set to 0 !!!
    hsc_lambda = 0  # TODO update
    true_interhomo_dis = np.array([0.7])  # FIXME !!! 15 or 5
    est_hmlg_sep = true_interhomo_dis * 1  # FIXME
    alpha, beta = -3., 1.
    multiscale_reform = True
    use_multiscale_variance = False
    max_attempt = 1e3
    scale = 0.75

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
            struct_true[i:, coord1] += random_state.choice(
                [1, -1]) * random_state.normal(scale=0.1 * scale) * scale
            struct_true[i:, coord2] += random_state.choice(
                [0.5, -0.5]) * random_state.normal(scale=0.1 * scale) * scale

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
        print(f'{i}: radiuses max: {h1_rad.max():.3g} & {h2_rad.max():.3g}')
        hmlg_sep = np.sqrt((np.square(hmlg1 - hmlg2)).sum(axis=1))
        print(f'{i}: sep: min={hmlg_sep.min():.3g}, max={hmlg_sep.max():.3g}')
        begin = end

    # Make counts (ambiguous)
    dis[dis == 0] = np.inf
    counts = beta * dis ** alpha
    # use_poisson = True  # FIXME
    # if use_poisson:
    #     counts = random_state.poisson(counts)
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = counts[:n, :n] + counts[n:, n:] + counts[:n, n:] + counts[n:, :n]
    counts = np.triu(counts, 1)
    counts = sparse.coo_matrix(counts)

    callback_freq = {'print': 0, 'history': 0, 'save': 0}
    struct_, infer_var = pastis_algorithms.infer(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, beta=beta,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda, est_hmlg_sep=est_hmlg_sep,
        callback_freq=callback_freq, struct_true=struct_true,
        multiscale_factor=multiscale_factor,
        multiscale_reform=multiscale_reform,
        use_multiscale_variance=use_multiscale_variance)

    assert infer_var['converged']

    interhomo_dis = _inter_homolog_dis(struct_, lengths=lengths)

    print(f"hmlg sep: true={true_interhomo_dis}, est_hmlg_sep={est_hmlg_sep}, final={interhomo_dis}")

    # Make sure inferred homologs are separated >= inferred est_hmlg_sep
    assert interhomo_dis >= est_hmlg_sep - 1e-6

    # Make sure separation of inferred homologs is roughly accurate
    assert_array_almost_equal(true_interhomo_dis, interhomo_dis, decimal=0)


@pytest.mark.parametrize("multiscale_factor", [1, 2, 4, 8])
def test_pastis_poisson_diploid_ambig_hsc_constraint_multiscale2(multiscale_factor):
    from topsy.datasets.samples_generator import make_3d_genome

    lengths = np.array([5])  # FIXME !!! 40
    ploidy = 2
    seed = 42

    bcc_lambda = 1  # FIXME !!! set to 0 !!!
    hsc_lambda = 10  # TODO update
    true_interhomo_dis = np.array([0.7])  # FIXME !!! 15 or 5
    est_hmlg_sep = true_interhomo_dis * 1  # FIXME
    alpha, beta = -3., 1.
    multiscale_reform = True
    use_multiscale_variance = False

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()

    # Create structure, without any beads overlapping
    struct_true = make_3d_genome(
        lengths, ploidy=ploidy, random_state=random_state,
        distance_btwn_chrom=None, better_sim=True, verbose=False)

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

    # Debugging stuff
    print(f'dis.min()='
          f'{dis[np.triu_indices(dis.shape[0], 1)].min():.3g}')
    begin = end = 0
    for i in range(len(lengths)):
        end += lengths[i]
        hmlg1 = struct_true[n:][begin:end]
        hmlg2 = struct_true[:n][begin:end]
        h1_rad = np.sqrt((np.square(hmlg1 - hmlg1.mean(axis=0))).sum(axis=1))
        h2_rad = np.sqrt((np.square(hmlg2 - hmlg2.mean(axis=0))).sum(axis=1))
        print(f'{i}: radiuses max: {h1_rad.max():.3g} & {h2_rad.max():.3g}')
        hmlg_sep = np.sqrt((np.square(hmlg1 - hmlg2)).sum(axis=1))
        print(f'{i}: sep: min={hmlg_sep.min():.3g}, max={hmlg_sep.max():.3g}')
        begin = end

    # Make counts (ambiguous)
    dis[dis == 0] = np.inf
    counts = beta * dis ** alpha
    # use_poisson = True  # FIXME
    # if use_poisson:
    #     counts = random_state.poisson(counts)
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = counts[:n, :n] + counts[n:, n:] + counts[:n, n:] + counts[n:, :n]
    counts = np.triu(counts, 1)
    counts = sparse.coo_matrix(counts)

    callback_freq = {'print': 0, 'history': 0, 'save': 0}
    struct_, infer_var = pastis_algorithms.infer(
        counts, lengths=lengths, ploidy=ploidy, outdir=None, alpha=alpha,
        seed=seed, normalize=False, filter_threshold=0, beta=beta,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda, est_hmlg_sep=est_hmlg_sep,
        callback_freq=callback_freq, struct_true=struct_true,
        multiscale_factor=multiscale_factor,
        multiscale_reform=multiscale_reform,
        use_multiscale_variance=use_multiscale_variance)

    assert infer_var['converged']

    interhomo_dis = _inter_homolog_dis(struct_, lengths=lengths)

    print(f"hmlg sep: true={true_interhomo_dis}, est_hmlg_sep={est_hmlg_sep}, final={interhomo_dis}")

    # Make sure inferred homologs are separated >= inferred est_hmlg_sep
    assert interhomo_dis >= est_hmlg_sep - 1e-6

    # Make sure separation of inferred homologs is roughly accurate
    assert_array_almost_equal(true_interhomo_dis, interhomo_dis, decimal=0)