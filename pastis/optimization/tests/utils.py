import numpy as np
from sklearn.metrics import euclidean_distances
from scipy import sparse
import sys

if sys.version_info[0] >= 3:
    from pastis.optimization.constraints import get_counts_interchrom


def get_coord_diff_from_euc_dis(rng, nghbr_bead_dis=1, ndim=3):
    """Euclidian distance ||x_{i} - x_{i-1}|| to 3D difference x_{i} - x_{i-1}
    """
    dis_sq = nghbr_bead_dis ** 2
    x_sq = rng.uniform(0, dis_sq)
    x = np.sqrt(x_sq) * rng.choice([-1, 1])
    if ndim == 3:
        y_sq = rng.uniform(0, dis_sq - x_sq)
        y = np.sqrt(y_sq) * rng.choice([-1, 1])
    else:
        y_sq = y = 0
    z_sq = dis_sq - x_sq - y_sq
    z = np.sqrt(z_sq) * rng.choice([-1, 1])
    if ndim == 3:
        coord_diffs = np.array([x, y, z])
    else:
        coord_diffs = np.array([x, z])
    rng.shuffle(coord_diffs)
    return coord_diffs


def make_3d_bead(rng, nghbr_bead_dis=1, noise=0.1, prev_bead=None,
                 ndim=3):
    """Create new bead in 3D structure"""
    if prev_bead is None:
        prev_bead = np.zeros(ndim)
    coord_diffs = get_coord_diff_from_euc_dis(
        rng, nghbr_bead_dis=nghbr_bead_dis, ndim=ndim)
    coord_noise = noise * rng.standard_normal((1, ndim))
    return coord_diffs + coord_noise + prev_bead


def make_3d_struct(nbeads, rng, nghbr_bead_dis=1, noise=0.1,
                   dis_min_cutoff=0.3, dis_max_cutoff_factor=0.4, max_iter=10,
                   ndim=3, verbose=False, count_restart=0):
    """Make 3D structure via basic random walk

    Make 3D structure, avoiding overlapping beads & (optionally) avoiding overly
    extended conformations"""

    if dis_max_cutoff_factor is None:
        dis_max_cutoff_factor = np.inf

    beads = np.zeros((nbeads, ndim))

    dis_min = np.inf
    for i in range(1, nbeads):
        # Come up with some cutoff for max allowed distance between any 2 beads
        dis_max_cutoff = (i * nghbr_bead_dis) ** dis_max_cutoff_factor

        beads[i] = make_3d_bead(
            rng=rng, nghbr_bead_dis=nghbr_bead_dis, noise=noise,
            prev_bead=beads[i - 1], ndim=ndim)
        if i > 1:
            dis_min = euclidean_distances(beads[:(i - 1)], [beads[i]]).min()
        dis_max = euclidean_distances(beads[:i], [beads[i]]).max()

        attempt = 1
        while attempt < max_iter and (
                dis_min < dis_min_cutoff or dis_max > dis_max_cutoff):
            if verbose:
                if dis_min < dis_min_cutoff:
                    print(f"    Position {i} invalid, bead is too close to"
                          f" other beads (attempt {attempt})")
                if dis_max > dis_max_cutoff:
                    print(f"    Position {i} invalid, structure is in an overly"
                          f" extended confrormation (attempt {attempt})")
            beads[i] = make_3d_bead(
                rng=rng, nghbr_bead_dis=nghbr_bead_dis, noise=noise,
                prev_bead=beads[i - 1], ndim=ndim)
            if i > 1:
                dis_min = euclidean_distances(beads[:(i - 1)], [beads[i]]).min()
            dis_max = euclidean_distances(beads[:i], [beads[i]]).max()
            attempt += 1

        if attempt == max_iter:
            if count_restart >= 100:
                raise ValueError("Restarted over 100x, something is wrong...")
            if verbose:
                print('    STARTING OVER', flush=True)
            return make_3d_struct(
                nbeads=nbeads, rng=rng, nghbr_bead_dis=nghbr_bead_dis,
                noise=noise, dis_min_cutoff=dis_min_cutoff,
                dis_max_cutoff_factor=dis_max_cutoff_factor, max_iter=max_iter,
                ndim=ndim, verbose=verbose, count_restart=count_restart + 1)

    return beads


def separate_homologs(struct, lengths, true_interhmlg_dis):
    """Separate homolog centers of mass by specified distance"""
    lengths = np.array(lengths, copy=False, ndmin=1, dtype=int)
    n = lengths.sum()

    if isinstance(true_interhmlg_dis, (int, float)):
        true_interhmlg_dis = np.repeat(true_interhmlg_dis, lengths.size)

    begin = end = 0
    for i in range(lengths.size):
        end += lengths[i]
        struct[begin:end] -= np.nanmean(struct[begin:end], axis=0)
        struct[(n + begin):(n + end)] -= np.nanmean(
            struct[(n + begin):(n + end)], axis=0)
        struct[begin:end, 0] += true_interhmlg_dis[i]
        begin = end

    return struct


def get_struct_randwalk(lengths, ploidy, random_state=None,
                        true_interhmlg_dis=None, nghbr_bead_dis=1, noise=0.1,
                        dis_min_cutoff=0.3, dis_max_cutoff_factor=0.4,
                        max_iter=10, verbose=False, scale=0.75):
    """Make 3D structure via very basic random walk, avoid overlapping beads"""

    if random_state is None:
        random_state = np.random.RandomState(seed=0)
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(seed=random_state)
    lengths = np.array(lengths, copy=False, ndmin=1, dtype=int)
    n = lengths.sum()

    # Create structure, without any beads overlapping
    struct = None
    dis = np.zeros((n * ploidy, n * ploidy))
    attempt = 0
    while attempt < max_iter and dis[
            np.triu_indices(dis.shape[0], 1)].min() <= dis_min_cutoff:
        attempt += 1

        struct = make_3d_struct(
            nbeads=n * ploidy, rng=random_state, nghbr_bead_dis=nghbr_bead_dis,
            noise=noise, dis_min_cutoff=dis_min_cutoff,
            dis_max_cutoff_factor=dis_max_cutoff_factor, max_iter=max_iter,
            verbose=verbose)

        if ploidy == 2 and true_interhmlg_dis is not None:
            struct = separate_homologs(
                struct, lengths=lengths, true_interhmlg_dis=true_interhmlg_dis)

        dis = euclidean_distances(struct)

    if attempt >= max_iter:
        raise ValueError("Couldn't create struct without overlapping beads.")

    # Debugging stuff  # TODO remove
    if verbose:
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


def get_counts(struct, ploidy, lengths, alpha=-3, beta=1, ambiguity='ua',
               struct_nan=None, random_state=None, use_poisson=False,
               bias=None):
    """Simulate Hi-C counts from 3D structure"""
    if ambiguity is None:
        ambiguity = 'ua'
    if ambiguity.lower() not in ('ua', 'ambig', 'pa'):
        raise ValueError(f"Ambiguity not understood: {ambiguity}")
    if use_poisson:
        if random_state is None:
            random_state = np.random.RandomState(seed=0)
        elif isinstance(random_state, int):
            random_state = np.random.RandomState(seed=random_state)

    lengths = np.array(lengths, copy=False, ndmin=1, dtype=int)
    n = lengths.sum()

    dis = euclidean_distances(struct)
    dis[dis == 0] = np.inf

    counts = beta * dis ** alpha
    if bias is not None:
        counts *= np.tile(bias, ploidy).reshape(-1, 1)
        counts *= np.tile(bias, ploidy).reshape(-1, 1).T
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

    return sparse.coo_matrix(counts)


def get_true_data_interchrom(struct_true, ploidy, lengths, alpha=-3, beta=1,
                             random_state=None, use_poisson=False):

    """For convenience, create data_interchrom from unambig inter-hmlg counts

    Enables generation of data_interchrom for one simulated chromosome"""

    counts_unambig = sparse.coo_matrix(sum([get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity="ua", struct_nan=None, random_state=random_state,
        use_poisson=use_poisson, bias=None).toarray() for i in range(4)]))
    data_interchrom = get_counts_interchrom(
        counts_unambig, lengths=np.tile(lengths, 2), ploidy=1,
        filter_threshold=0, normalize=False, bias=None, verbose=False)
    return data_interchrom


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

    if sparse.issparse(counts):
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

    return sparse.coo_matrix(counts_lowres)
