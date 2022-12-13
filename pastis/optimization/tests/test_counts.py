import sys
import pytest
import numpy as np
from sklearn.metrics import euclidean_distances
from scipy import sparse
from numpy.testing import assert_array_equal

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    import pastis.optimization.counts as counts_py


@pytest.mark.parametrize(
    "ambiguity,multiscale_factor",
    [('ambig', 1), ('pa', 1), ('ua', 1), ('ambig', 2), ('pa', 2), ('ua', 2),
     ('ambig', 4), ('pa', 4), ('ua', 4), ('ambig', 8), ('pa', 8), ('ua', 8)])
def test_ambiguate_counts(ambiguity, multiscale_factor):
    if ambiguity not in ('ambig', 'pa', 'ua'):
        raise ValueError(f"Ambiguity not understood: {ambiguity}")
    lengths = np.array([10, 21])
    ploidy = 2
    seed = 42
    alpha, beta = -3., 1.
    nan_indices = np.array([0, 1, 2, 3, 12, 15, 25, 40])

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    struct_true = random_state.rand(n * ploidy, 3)
    dis = euclidean_distances(struct_true)
    dis[dis == 0] = np.inf

    counts = beta * dis ** alpha
    counts[np.isnan(counts) | np.isinf(counts)] = 0
    if ambiguity == 'ambig':
        counts = counts[:n, :n] + counts[
            n:, n:] + counts[:n, n:] + counts[n:, :n]
        counts = np.triu(counts, 1)
    elif ambiguity == 'pa':
        counts = counts[:, :n] + counts[:, n:]
        np.fill_diagonal(counts[:n, :], 0)
        np.fill_diagonal(counts[n:, :], 0)
    elif ambiguity == 'ua':
        counts = np.triu(counts, 1)
    counts[nan_indices[nan_indices < counts.shape[0]], :] = np.nan
    counts[:, nan_indices[nan_indices < counts.shape[1]]] = np.nan
    counts = sparse.coo_matrix(counts)

    counts_ambig_true = counts_py.ambiguate_counts(
        counts, lengths=lengths, ploidy=ploidy, exclude_zeros=False)

    counts_object = counts_py._format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta,
        exclude_zeros=False, multiscale_factor=multiscale_factor)
    counts_ambig = counts_object.ambiguate().toarray()

    assert_array_equal(
        np.invert(np.isnan(counts_ambig_true)) & (counts_ambig_true != 0),
        np.invert(np.isnan(counts_ambig)) & (counts_ambig != 0))
    assert_array_equal(counts_ambig_true, counts_ambig)
