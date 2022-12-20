import numpy as np
from sklearn.metrics import euclidean_distances
from scipy import sparse


def get_new_bead(random_state, distance=1., noise=0.1, prev_bead=None):
    if prev_bead is None:
        prev_bead = np.zeros(3)
    dis_sq = distance ** 2
    x_sq = random_state.uniform(0, dis_sq)
    x = np.sqrt(x_sq) * random_state.choice([-1, 1])
    y_sq = random_state.uniform(0, dis_sq - x_sq)
    y = np.sqrt(y_sq) * random_state.choice([-1, 1])
    z_sq = dis_sq - x_sq - y_sq
    z = np.sqrt(z_sq) * random_state.choice([-1, 1])
    coord_diffs = np.array([x, y, z])
    random_state.shuffle(coord_diffs)
    coord_noise = noise * random_state.randn(*(1, 3))
    return coord_diffs + coord_noise + prev_bead


def get_counts(struct, ploidy, lengths, alpha=-3, beta=1, ambiguity='ua',
               struct_nan=None, random_state=None, use_poisson=False):
    if ambiguity is None:
        ambiguity = 'ua'
    if ambiguity.lower() not in ('ua', 'ambig', 'pa'):
        raise ValueError(f"Ambiguity not understood: {ambiguity}")
    if use_poisson and random_state is None:
        random_state = np.random.RandomState(seed=0)

    lengths = np.array(lengths, copy=False, ndmin=1, dtype=int)
    n = lengths.sum()

    dis = euclidean_distances(struct)
    dis[dis == 0] = np.inf

    counts = beta * dis ** alpha
    if use_poisson:
        counts = random_state.poisson(counts)
    if ploidy == 1 or ambiguity.lower() == 'ua':
        counts = np.triu(counts, 1)
    elif ambiguity.lower() == 'ambig':
        counts = counts[:n, :n] + counts[
            n:, n:] + counts[:n, n:] + counts[n:, :n]
        counts = np.triu(counts, 1)
    elif ambiguity.lower() == 'pa':
        counts = counts[:, :n] + counts[:, n:]
        np.fill_diagonal(counts[:n, :], 0)
        np.fill_diagonal(counts[n:, :], 0)
    if struct_nan is not None:
        counts[struct_nan[struct_nan < counts.shape[0]], :] = 0
        counts[:, struct_nan[struct_nan < counts.shape[1]]] = 0
    counts = sparse.coo_matrix(counts)
    return counts
