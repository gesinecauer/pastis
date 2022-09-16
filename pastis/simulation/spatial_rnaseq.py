import numpy as np
from sklearn.metrics import euclidean_distances
import os
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='', category=UserWarning)
    warnings.filterwarnings('ignore', message='', category=FutureWarning)
    from iced.io import write_counts, write_lengths


def get_new_beads(rng, nbeads, circle_radius=5, center=(0, 0)):
    """Uniform distribution of beads within a circle"""
    r = circle_radius * np.sqrt(rng.uniform(size=nbeads))
    theta = rng.uniform(size=nbeads) * 2 * np.pi
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    beads = np.stack([x, y], axis=1)
    return beads


def remove_overlaps(beads, overlap_radius=0.2):
    """Ensure distance between beads is greater than a given radius"""
    dis = euclidean_distances(beads)
    np.fill_diagonal(dis, np.nan)

    for i in range(beads.shape[0]):
        if np.nanmin(dis[i]) < overlap_radius:
            dis[i, :] = np.nan
            dis[:, i] = np.nan
            beads[i] = np.nan

    beads = beads[~np.isnan(beads[:, 0])]
    return beads


def get_struct_2d(rng, nbeads, circle_radius=5, overlap_radius=0.2,
                  center=(0, 0), extra_bead_factor=2, beads=None, attempt=1,
                  verbose=False):
    """Get non-overlapping beads within a circle"""

    if verbose:
        if beads is None:
            print(f"Attempting to create {nbeads} beads...", flush=True)
        else:
            print(f"...try {attempt}, need an additional"
                  f" {nbeads - beads.shape[0]} beads", flush=True)

    # Add new beads
    if beads is None:
        nbeads_new = nbeads * extra_bead_factor
    else:
        nbeads_new = (nbeads - beads.shape[0]) * extra_bead_factor
    beads_new = get_new_beads(
        rng=rng, nbeads=nbeads_new, circle_radius=circle_radius, center=center)
    if beads is not None:
        beads_new = np.concatenate([beads, beads_new])

    # Remove overlaps between beads
    beads_new = remove_overlaps(beads_new, overlap_radius=overlap_radius)
    if beads is None or beads_new.shape[0] > beads.shape[0]:
        beads = beads_new

    # Add extra beads if necessary
    if beads.shape[0] < nbeads:
        return get_struct_2d(
            rng=rng, nbeads=nbeads, circle_radius=circle_radius,
            overlap_radius=overlap_radius, center=center,
            extra_bead_factor=extra_bead_factor, beads=beads,
            attempt=attempt + 1, verbose=verbose)
    else:
        return beads[:nbeads]


def get_dis_alpha(struct, nreads=None, alpha=-1):
    struct = struct.copy()
    mask = ~np.isnan(struct[:, 0])
    struct[~mask] = 0
    dis = euclidean_distances(struct)
    dis[~mask, :] = np.inf
    dis[:, ~mask] = np.inf
    dis[dis == 0] = np.inf
    dis_alpha = dis ** alpha
    if nreads is None:
        beta = 1.
    else:
        beta = nreads / np.triu(dis_alpha, 1).sum()
    return dis_alpha, beta


def get_counts(rng, struct, nreads, alpha=-1, distrib='poisson'):

    if distrib is not None and distrib.lower() not in (
            'poisson', 'negbinom', 'none'):
        raise ValueError(f"Distribution not recognized: {distrib}")

    dis_alpha, beta = get_dis_alpha(struct, nreads=nreads, alpha=alpha)

    if distrib is None or distrib.lower() == 'none':
        counts = dis_alpha
        beta = 1
    elif distrib.lower() == 'poisson':
        counts = rng.poisson(beta * dis_alpha)
    elif distrib.lower() == 'negbinom':
        raise NotImplementedError

    counts = np.triu(counts, 1)
    return counts, beta


def sim_dataset(nreads, nbeads, seed=0, directory='', alpha=-1, circle_radius=5,
                overlap_radius=0.2, distrib='poisson', redo=False,
                verbose=False):
    """"""

    # Define output directory & files
    sim_directory = (f"sim_spatial_rnaseq.nbeads{nbeads}_alpha{alpha}"
                     f"_circle{circle_radius}_overlap{overlap_radius}")
    if distrib is None or distrib.lower() == 'none':
        sim_directory += '_raw-dis-alpha'
    else:
        sim_directory += f'_{distrib.lower()}_nreads{nreads}'
    outdir = os.path.join(directory, sim_directory, f'{seed:03d}')
    lengths_file = os.path.join(outdir, "counts.bed")
    struct_true_file = os.path.join(outdir, "struct_true.coords")
    counts_file = os.path.join(outdir, "counts.matrix")
    metadata_file = os.path.join(outdir, "dataset_info.txt")

    # Check if already exists
    if (not redo) and all([os.path.isfile(x) for x in (
            lengths_file, struct_true_file, counts_file, metadata_file)]):
        return outdir

    rng = np.random.default_rng(seed)
    if verbose:
        print(f"Simulating spatial transcriptomics:\n\t{outdir}", flush=True)

    # Make non-overlapping beads within a circle
    struct_true = get_struct_2d(
        rng=rng, nbeads=nbeads, circle_radius=circle_radius,
        overlap_radius=overlap_radius, verbose=max(0, verbose - 1))

    # Get counts and beta (structure size scaling factor)
    counts, beta = get_counts(
        rng=rng, struct=struct_true, nreads=nreads, alpha=alpha,
        distrib=distrib)

    # Save data
    os.makedirs(outdir, exist_ok=True)
    lengths = np.array([nbeads])
    write_lengths(lengths_file, lengths)
    np.savetxt(struct_true_file, struct_true)
    write_counts(counts_file, counts)

    # Save metadata
    counts_eq_0 = (counts[np.triu_indices(nbeads, 1)] == 0).sum()
    total_counts_bins = (nbeads * nbeads - nbeads) / 2
    perc_zero_bins = counts_eq_0 / total_counts_bins * 100
    info = ['ploidy\t1', f'alpha\t{alpha:g}', f'seed\t{seed}',
            f'nreads\t{nreads:g}', 'ua\t1', 'pa\t0', f'lengths\t{nbeads}',
            f'beta\t{beta:g}', f'beta.ua\t{beta:g}'
            f"perc_zero.ua\t{perc_zero_bins:g}",
            f'circle_radius\t{circle_radius:g}',
            f'overlap_radius\t{overlap_radius:g}', f'distrib\t{distrib}']
    with open(metadata_file, 'w') as f:
        f.write('\n'.join(info))

    return outdir


def sim_spatial_rnaseq(nreads, nbeads, num_struct=1, seed=0, directory='',
                       alpha=-1, circle_radius=5, overlap_radius=0.2,
                       distrib='poisson', redo=False, verbose=False):
    """"""

    outdirs = []
    for x in range(seed, seed + num_struct):
        outdirs.append(sim_spatial_rnaseq(
            nreads=nreads, nbeads=nbeads, seed=x, directory=directory,
            alpha=alpha, circle_radius=circle_radius,
            overlap_radius=overlap_radius, distrib=distrib, redo=redo,
            verbose=verbose))
    return outdirs


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--nreads", default=1e3, type=float)
    parser.add_argument("--nbeads", default=500, type=int)
    parser.add_argument("--num_struct", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--directory", default='')
    parser.add_argument("--alpha", default=-1, type=float)
    parser.add_argument("--circle_radius", default=5, type=float)
    parser.add_argument("--overlap_radius", default=0.2, type=float)
    parser.add_argument("--distrib", default='poisson', type=str,
                        choices=['poisson', 'negbinom', 'none'])
    parser.add_argument('--redo', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    sim_spatial_rnaseq(
        nreads=args.nreads, nbeads=args.nbeads, num_struct=args.num_struct,
        seed=args.seed, directory=args.directory, alpha=args.alpha,
        circle_radius=args.circle_radius, overlap_radius=args.overlap_radius,
        distrib=args.distrib, redo=args.redo, verbose=args.verbose)


if __name__ == "__main__":
    main()
