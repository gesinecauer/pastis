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
    from pastis.optimization.counts import _format_counts, preprocess_counts
    from pastis.optimization.multiscale_optimization import decrease_lengths_res
    from pastis.optimization.multiscale_optimization import decrease_struct_res


def test_bcc_constraint():
    lengths = np.array([20])
    ploidy = 2
    alpha, beta = -3., 1.

    n = lengths.sum()
    X_true = np.concatenate(
        [np.arange(n * ploidy).reshape(-1, 1), np.zeros((n * ploidy, 1)),
            np.zeros((n * ploidy, 1))], axis=1)
    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    counts = np.triu(counts, 1)
    counts = sparse.coo_matrix(counts)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta)

    constraint = constraints.Constraints(
        counts, lengths=lengths, ploidy=ploidy, multiscale_factor=1,
        constraint_lambdas={'bcc': 1},
        constraint_params=None)
    constraint.check()
    obj = constraint.apply(X_true)['obj_bcc']
    assert obj < 1e-6


@pytest.mark.parametrize("multiscale_factor", [1, 2, 3, 4])
def test_hsc_constraint(multiscale_factor):
    lengths = np.array([30])
    ploidy = 2
    seed = 42
    true_interhmlg_dis = np.array([10.])  # Should be same shape as lengths
    alpha, beta = -3., 1.
    nan_indices = None
    # nan_indices = np.array([0, 1, 2, 3, 12, 15, 25])
    multiscale_reform = False

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()

    X_true = np.zeros((n * ploidy, 3), dtype=float)
    for i in range(X_true.shape[0]):
        X_true[i:, random_state.choice([0, 1, 2])] += 1

    X_true[n:] -= X_true[n:].mean(axis=0)
    X_true[:n] -= X_true[:n].mean(axis=0)
    begin = end = 0
    for i in range(len(lengths)):
        end += lengths[i]
        X_true[begin:end, 0] += true_interhmlg_dis[i]
        begin = end

    dis = euclidean_distances(X_true)
    dis[dis == 0] = np.inf
    counts_raw = beta * dis ** alpha
    counts_raw[np.isnan(counts_raw) | np.isinf(counts_raw)] = 0
    counts_raw = np.triu(counts_raw, 1)
    counts_raw = sparse.coo_matrix(counts_raw)

    # Fill nan_indices with junk
    if nan_indices is not None:
        X_true[nan_indices] = np.array([[100, 1000, 10000]]) * np.flip(
            nan_indices + 1).reshape(-1, 1)
        counts_raw[nan_indices, :] = 0
        counts_raw[:, nan_indices] = 0

    if multiscale_factor == 1:
        fullres_torm = None
    else:
        _, _, _, fullres_torm = preprocess_counts(
            counts_raw, lengths=lengths, ploidy=ploidy, multiscale_factor=1,
            normalize=False, filter_threshold=0, beta=beta,
            multiscale_reform=multiscale_reform, verbose=False)

    counts, _, _, _ = preprocess_counts(
        counts_raw, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, normalize=False,
        filter_threshold=0, beta=beta, fullres_torm=fullres_torm,
        multiscale_reform=multiscale_reform, verbose=False)

    constraint = constraints.Constraints(
        counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, constraint_lambdas={'hsc': 1},
        constraint_params={'hsc': true_interhmlg_dis})
    constraint.check()
    obj = constraint.apply(X_true)['obj_hsc']
    assert obj < 1e-6
