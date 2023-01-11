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
               struct_nan=None, random_state=None, use_poisson=False,
               bias=None):
    """Simulate Hi-C counts from 3D structure"""
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

    if bias is not None:
        counts *= np.tile(bias, int(counts.shape[0] / n)).reshape(-1, 1)
        counts *= np.tile(bias, int(counts.shape[1] / n)).reshape(-1, 1).T

    return sparse.coo_matrix(counts)


def get_struct_randwalk(lengths, ploidy, random_state=None,
                        true_interhmlg_dis=None, max_attempt=1e3, scale=0.75):
    """Make structure via a very basic random walk, avoid overlapping beads"""

    if random_state is None:
        random_state = np.random.RandomState(seed=0)
    lengths = np.array(lengths, copy=False, ndmin=1, dtype=int)
    n = lengths.sum()

    # Create structure, without any beads overlapping
    struct = None
    dis = np.zeros((n * ploidy, n * ploidy))
    attempt = 0
    while attempt < max_attempt and dis[
            np.triu_indices(dis.shape[0], 1)].min() <= 1e-6:
        attempt += 1

        # Make structure
        struct = np.zeros((n * ploidy, 3), dtype=float)
        for i in range(struct.shape[0]):
            coord1 = random_state.choice([0, 1, 2])
            coord2 = random_state.choice([x for x in (0, 1, 2) if x != coord1])
            struct[i:, coord1] += random_state.choice(
                [1, -1]) * random_state.normal(scale=0.1 * scale) * scale
            struct[i:, coord2] += random_state.choice(
                [0.5, -0.5]) * random_state.normal(scale=0.1 * scale) * scale

        if ploidy == 2 and true_interhmlg_dis is not None:
            # Center homologs, so that distance between barycenters is 0
            begin = end = 0
            for i in range(len(lengths)):
                end += lengths[i]
                # struct[n:][begin:end] -= struct[n:][begin:end].mean(axis=0)
                struct[(begin + n):(end + n)] -= struct[
                    (begin + n):(end + n)].mean(axis=0)
                struct[begin:end] -= struct[begin:end].mean(axis=0)
                begin = end

            # Separate homologs
            begin = end = 0
            for i in range(len(lengths)):
                end += lengths[i]
                struct[begin:end, 0] += true_interhmlg_dis[i]
                begin = end

        dis = euclidean_distances(struct)

    if attempt >= max_attempt:
        raise ValueError("Couldn't create struct without overlapping beads.")

    # Debugging stuff  # TODO remove
    print(f'debugging get_struct_randwalk():  tries={attempt}, dis.min()='
          f'{dis[np.triu_indices(dis.shape[0], 1)].min():.3g}')
    begin = end = 0
    for i in range(len(lengths)):
        end += lengths[i]
        hmlg1 = struct[n:][begin:end]
        hmlg2 = struct[:n][begin:end]
        h1_rad = np.sqrt((np.square(hmlg1 - hmlg1.mean(axis=0))).sum(axis=1))
        h2_rad = np.sqrt((np.square(hmlg2 - hmlg2.mean(axis=0))).sum(axis=1))
        print(f'{i}: radiuses max: {h1_rad.max():.3g} & {h2_rad.max():.3g}')
        hmlg_sep = np.sqrt((np.square(hmlg1 - hmlg2)).sum(axis=1))
        print(f'{i}: sep: min={hmlg_sep.min():.3g}, max={hmlg_sep.max():.3g}')
        begin = end

    return struct


def decrease_struct_res_correct(struct, multiscale_factor, lengths, ploidy):
    if multiscale_factor == 1:
        return struct

    struct = struct.copy().reshape(-1, 3)
    lengths = np.array(lengths, copy=False, ndmin=1, dtype=int)

    struct_lowres = []
    begin = end = 0
    for length in np.tile(lengths, ploidy):
        end += length
        struct_chrom = struct[begin:end]
        remainder = struct_chrom.shape[0] % multiscale_factor
        struct_chrom_reduced = np.nanmean(
            struct_chrom[:struct_chrom.shape[0] - remainder, :].reshape(
                -1, multiscale_factor, 3), axis=1)
        if remainder == 0:
            struct_lowres.append(struct_chrom_reduced)
        else:
            struct_chrom_overhang = np.nanmean(
                struct_chrom[struct_chrom.shape[0] - remainder:, :],
                axis=0).reshape(-1, 3)
            struct_lowres.extend([struct_chrom_reduced, struct_chrom_overhang])
        begin = end

    return np.concatenate(struct_lowres)


def decrease_counts_res_correct(counts, multiscale_factor, lengths):
    if multiscale_factor == 1:
        return counts

    is_sparse = sparse.issparse(counts)
    if is_sparse:
        counts = counts.toarray()
    lengths = np.array(lengths, copy=False, ndmin=1, dtype=int)
    triu = counts.shape[0] == counts.shape[1]
    if triu:
        counts = np.triu(counts, 1)

    lengths_lowres = np.ceil(
        lengths.astype(float) / multiscale_factor).astype(int)
    map_factor_row = int(counts.shape[0] / lengths.sum())
    map_factor_col = int(counts.shape[1] / lengths.sum())

    counts_lowres = np.zeros((
        lengths_lowres.sum() * map_factor_row,
        lengths_lowres.sum() * map_factor_col), dtype=counts.dtype)
    tiled_lengths = np.tile(lengths, max(map_factor_row, map_factor_col))
    tiled_lengths_lowres = np.tile(
        lengths_lowres, max(map_factor_row, map_factor_col))

    for c1 in range(lengths.shape[0] * map_factor_row):
        for i in range(tiled_lengths[c1]):
            row_fullres = tiled_lengths[:c1].sum() + i
            i_lowres = int(np.ceil(float(i + 1) / multiscale_factor) - 1)
            row_lowres = tiled_lengths_lowres[:c1].sum() + i_lowres
            if triu:
                c2_start = c1
            else:
                c2_start = 0
            for c2 in range(c2_start, lengths.shape[0] * map_factor_col):
                if triu and c1 == c2:
                    j_start = i + 1
                else:
                    j_start = 0
                for j in range(j_start, tiled_lengths[c2]):
                    col_fullres = tiled_lengths[:c2].sum() + j
                    j_lowres = int(np.ceil(float(j + 1) / multiscale_factor) - 1)
                    col_lowres = tiled_lengths_lowres[:c2].sum() + j_lowres
                    bin_fullres = counts[row_fullres, col_fullres]
                    if triu:
                        on_diag = c1 == c2 and i_lowres == j_lowres
                    elif c1 >= lengths.shape[0]:
                        on_diag = (c1 - lengths.shape[0]) == c2 and i_lowres == j_lowres
                    elif c2 >= lengths.shape[0]:
                        on_diag = c1 == (c2 - lengths.shape[0]) and i_lowres == j_lowres
                    else:
                        on_diag = c1 == c2 and i_lowres == j_lowres
                    if not np.isnan(bin_fullres) and not on_diag:
                        counts_lowres[row_lowres, col_lowres] += bin_fullres

    if is_sparse:
        counts_lowres = sparse.coo_matrix(counts_lowres)

    return counts_lowres

