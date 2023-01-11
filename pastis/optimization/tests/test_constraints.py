import sys
import pytest
import numpy as np
from sklearn.metrics import euclidean_distances
from scipy import sparse
from numpy.testing import assert_allclose

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from utils import get_counts

    from pastis.optimization import constraints
    from pastis.optimization.counts import preprocess_counts
    from pastis.optimization.multiscale_optimization import decrease_lengths_res
    from pastis.optimization.multiscale_optimization import decrease_struct_res


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

    struct_true_lowres = decrease_struct_res(
        struct_true, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy, fullres_struct_nan=fullres_struct_nan)

    constraint = constraints.prep_constraints(
        lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor,
        bcc_lambda=1, hsc_lambda=0, bcc_version='2019', data_interchrom=None,
        est_hmlg_sep=None, hsc_perc_diff=None,
        fullres_struct_nan=fullres_struct_nan, verbose=False)[0]
    constraint.check()
    obj = constraint.apply(
        struct_true_lowres, alpha=alpha, epsilon=None, counts=counts, bias=None)
    assert obj < 1e-6


@pytest.mark.parametrize("multiscale_factor", [1, 2, 4, 8])
def test_constraint_hsc2019(multiscale_factor):
    lengths = np.array([30])
    ploidy = 2
    seed = 42
    true_interhmlg_dis = np.array([10.])  # Should be same shape as lengths
    alpha, beta = -3, 1
    struct_nan = np.array([0, 1, 2, 3, 12, 15, 25])

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()

    struct_true = np.zeros((n * ploidy, 3), dtype=float)
    for i in range(struct_true.shape[0]):
        coord = random_state.choice([0, 1, 2])
        struct_true[i:, coord] += random_state.choice([1, -1])

    # Fill struct_nan of structure with nan
    if struct_nan is not None:
        struct_true[struct_nan] = np.nan

    # Separate homologs
    begin = end = 0
    for i in range(len(lengths)):
        end += lengths[i]
        struct_true[begin:end] -= np.nanmean(struct_true[begin:end], axis=0)
        struct_true[(n + begin):(n + end)] -= np.nanmean(
            struct_true[(n + begin):(n + end)], axis=0)
        struct_true[begin:end, 0] += true_interhmlg_dis[i]
        begin = end

    # Fill struct_nan of structure with junk
    if struct_nan is not None:
        struct_true[struct_nan] = np.array([[100, 1000, 10000]]) * np.flip(
            struct_nan + 1).reshape(-1, 1)

    # Make counts
    counts_raw = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity="ua", struct_nan=struct_nan, random_state=random_state,
        use_poisson=False, bias=None)

    counts, _, fullres_struct_nan = preprocess_counts(
        counts_raw, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, beta=beta, verbose=False)

    struct_true_lowres = decrease_struct_res(
        struct_true, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy, fullres_struct_nan=fullres_struct_nan)
    struct_true_lowres = np.where(
        np.isnan(struct_true_lowres), 0, struct_true_lowres)

    constraint = constraints.prep_constraints(
        lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor,
        bcc_lambda=0, hsc_lambda=1, hsc_version='2019', data_interchrom=None,
        est_hmlg_sep=true_interhmlg_dis, hsc_perc_diff=None,
        fullres_struct_nan=fullres_struct_nan, verbose=False)[0]
    constraint.check()
    obj = constraint.apply(
        struct_true_lowres, alpha=alpha, epsilon=None, counts=counts, bias=None)

    # actual_interhmlg_dis = constraint._homolog_separation(struct_true_lowres)
    # assert_allclose(true_interhmlg_dis, actual_interhmlg_dis)

    assert obj < 1e-6
