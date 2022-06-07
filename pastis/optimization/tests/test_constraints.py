import sys
import pytest
import numpy as np
from sklearn.metrics import euclidean_distances
from scipy import sparse
from numpy.testing import assert_allclose

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from pastis.optimization import constraints
    from pastis.optimization.counts import preprocess_counts
    from pastis.optimization.multiscale_optimization import decrease_lengths_res
    from pastis.optimization.multiscale_optimization import decrease_struct_res


@pytest.mark.parametrize("multiscale_factor", [1, 2, 4])
def test_bcc_constraint(multiscale_factor):
    lengths = np.array([20])
    ploidy = 2
    alpha, beta = -3., 1.

    n = lengths.sum()
    struct_true = np.concatenate(
        [np.arange(n * ploidy).reshape(-1, 1), np.zeros((n * ploidy, 1)),
            np.zeros((n * ploidy, 1))], axis=1)
    dis = euclidean_distances(struct_true)
    dis[dis == 0] = np.inf
    counts_raw = beta * dis ** alpha
    counts_raw[np.isnan(counts_raw) | np.isinf(counts_raw)] = 0
    counts_raw = np.triu(counts_raw, 1)
    counts_raw = sparse.coo_matrix(counts_raw)

    counts, _, _, fullres_struct_nan = preprocess_counts(
        counts_raw, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, normalize=False,
        filter_threshold=0, beta=beta, verbose=False)

    struct_true_lowres = decrease_struct_res(
        struct_true, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy, fullres_struct_nan=fullres_struct_nan)

    constraint = constraints.constraints = prep_constraints(
        counts=counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, bcc_lambda=1,
        hsc_lambda=0, bcc_version='2019',
        fullres_struct_nan=None, verbose=False)[0]
    constraint.check()
    obj = constraint.apply(struct_true_lowres)['obj_bcc']
    assert obj < 1e-6


@pytest.mark.parametrize("multiscale_factor", [1, 2, 4, 8])
def test_hsc_constraint(multiscale_factor):
    lengths = np.array([30])
    ploidy = 2
    seed = 42
    true_interhmlg_dis = np.array([10.])  # Should be same shape as lengths
    alpha, beta = -3., 1.
    nan_indices = np.array([0, 1, 2, 3, 12, 15, 25])

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()

    struct_true = np.zeros((n * ploidy, 3), dtype=float)
    for i in range(struct_true.shape[0]):
        coord = random_state.choice([0, 1, 2])
        struct_true[i:, coord] += random_state.choice([1, -1])

    # Fill nan_indices of structure with nan
    if nan_indices is not None:
        struct_true[nan_indices] = np.nan

    # Separate homologs
    begin = end = 0
    for i in range(len(lengths)):
        end += lengths[i]
        struct_true[begin:end] -= np.nanmean(struct_true[begin:end], axis=0)
        struct_true[(n + begin):(n + end)] -= np.nanmean(
            struct_true[(n + begin):(n + end)], axis=0)
        struct_true[begin:end, 0] += true_interhmlg_dis[i]
        begin = end

    # Fill nan_indices of structure with junk
    if nan_indices is not None:
        struct_true[nan_indices] = np.array([[100, 1000, 10000]]) * np.flip(
            nan_indices + 1).reshape(-1, 1)

    # Make counts
    dis = euclidean_distances(struct_true)
    dis[dis == 0] = np.inf
    counts_raw = beta * dis ** alpha
    counts_raw[np.isnan(counts_raw) | np.isinf(counts_raw)] = 0
    counts_raw = np.triu(counts_raw, 1)

    # Zero out nan_indices of counts
    if nan_indices is not None:
        counts_raw[nan_indices, :] = 0
        counts_raw[:, nan_indices] = 0
    counts_raw = sparse.coo_matrix(counts_raw)

    counts, _, _, fullres_struct_nan = preprocess_counts(
        counts_raw, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, normalize=False,
        filter_threshold=0, beta=beta, verbose=False)

    struct_true_lowres = decrease_struct_res(
        struct_true, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy, fullres_struct_nan=fullres_struct_nan)
    struct_true_lowres = np.where(
        np.isnan(struct_true_lowres), 0, struct_true_lowres)

    constraint = constraints.constraints = prep_constraints(
        counts=counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, bcc_lambda=0,
        hsc_lambda=1, hsc_version='2019',
        est_hmlg_sep=true_interhmlg_dis, hsc_perc_diff=None,
        fullres_struct_nan=fullres_struct_nan, verbose=False)[0]
    constraint.check()

    print(np.isnan(struct_true_lowres).sum())

    actual_interhmlg_dis = constraint._homolog_separation(struct_true_lowres)
    assert_allclose(true_interhmlg_dis, actual_interhmlg_dis)

    obj = constraint.apply(struct_true_lowres)['obj_hsc']
    assert obj < 1e-6
