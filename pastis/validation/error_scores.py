import os
import re
import numpy as np
import pandas as pd
from rmsd import kabsch
from sklearn.metrics import euclidean_distances
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='', category=UserWarning)
    warnings.filterwarnings('ignore', message='', category=FutureWarning)
    from _realignment import realign_structures


def get_distance_error(structX, structY, rescale_X_to_Y=False,
                       return_diff=False):
    """Get MSE of distance matrices"""

    # Check input
    if structX.shape != structY.shape:
        raise ValueError("Shapes of the two structures need to be the same.")

    mask = np.isfinite(structX[:, 0]) & np.isfinite(structY[:, 0])
    if mask.sum() == 0 or (
            all(np.unique(structX) == [0]) and all(np.unique(structY) == [0])):
        warnings.warn('Skipping blank struct, all values are nan, inf, or 0.')
        return None

    # Optionally rescale structures (no need to realign or check mirror image)
    if rescale_X_to_Y:
        structX = realign_structures(
            structX, structY, rescale_X_to_Y=rescale_X_to_Y,
            error_type='disterror')[0]

    dis_diff_all = euclidean_distances(structX[mask]) - euclidean_distances(
        structY[mask])
    disterror = np.sqrt(np.mean(np.square(dis_diff_all)))
    if return_diff:
        return disterror, dis_diff_all.mean()
    else:
        return disterror


def get_rmsd(structX, structY, rescale_X_to_Y=False):
    """Get root mean square deviation of structure coords"""

    # Check input
    if structX.shape != structY.shape:
        raise ValueError("Shapes of the two structures need to be the same.")

    mask = np.isfinite(structX[:, 0]) & np.isfinite(structY[:, 0])
    if mask.sum() == 0 or (
            all(np.unique(structX) == [0]) and all(np.unique(structY) == [0])):
        warnings.warn('Skipping blank struct, all values are nan, inf, or 0.')
        return None

    # Realign, check reflection (mirror image) of structures, optionally rescale
    structX = structX - np.nanmean(structX, axis=0)
    structY = structY - np.nanmean(structY, axis=0)
    structX = realign_structures(
        structX, structY, rescale_X_to_Y=rescale_X_to_Y, error_type='rmsd')[0]
    structX[mask] = np.dot(structX[mask], kabsch(structX[mask], structY[mask]))

    rmsd_sq = np.mean(np.square(structX[mask] - structY[mask]))
    return np.sqrt(rmsd_sq)


def compare_btwn_struct(structX, structY, rescale_X_to_Y=True, outfile=None,
                        verbose=True):
    """Gets error between 3D structures.

    Gets error between two 3D structures.

    Parameters
    ----------
    structX : str or array of float
        The 3D structure to evaluate, or the filename of this structure (with
        three space-separated columns indicating the x, y, and z coordinates).
    structY : str or array of float
        The (baseline) 3D structure to compare to, or the filename of this
        structure (with three space-separated columns indicating the x, y, and z
        coordinates).
    rescale_X_to_Y : bool, optional
        Whether to rescale `structX` to optimally match `structY`.
    outfile : str, optional
        Path to the output file of computed error scores. If outfile is None,
        error scores will not be saved to file.
    verbose : bool, optional
        Verbosity

    Returns
    -------
    error_scores : pd.Series
        Computed error scores.
    """

    # Load data
    if isinstance(structX, str):
        structX = np.loadtxt(structX)
    else:
        structX = structX.copy()
    if len(structX.shape) > 1 and structX.shape[1] > 2 and np.all(
            structX[:, 2:] == 0):
        structX = structX[:, :2]
    structX = structX.reshape(-1, 2)

    if isinstance(structY, str):
        structY = np.loadtxt(structY)
    else:
        structY = structY.copy()
    if len(structY.shape) > 1 and structY.shape[1] > 2 and np.all(
            structY[:, 2:] == 0):
        structY = structY[:, :2]
    structY = structY.reshape(-1, 2)

    # Check validity of inputs
    if len(structX) != len(structY):
        dims_structX = ', '.join([str(d) for d in structX.shape])
        dims_structY = ', '.join([str(d) for d in structY.shape])
        raise ValueError(f"Structures are of not of equal size. Dimensions are "
                         f"({dims_structX}) and ({dims_structY}).")

    # Compute error scores
    error = pd.Series()
    error['rmsd'] = get_rmsd(
        structX, structY, rescale_X_to_Y=rescale_X_to_Y)
    error['rmse_distance'], error['mean_err_distance'] = get_distance_error(
        structX, structY, rescale_X_to_Y=rescale_X_to_Y, return_diff=True)

    if verbose:
        print(error.map('{:.3g}'.format), flush=True)
    if outfile is not None:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        error.to_csv(outfile, sep='\t', header=False)

    return error.to_dict()


def get_outfile(structures, primary_file=None, rescale=False):
    # structures is tuple of tuples, sort of like some_dict.items() would be

    # Example btwn_struct outfiles:
    # error.true_vs_infer000, error.true_vs_init000, error.infer000_vs_infer001

    struct_keys = list([k for k, v in structures])
    struct_files = list([v for k, v in structures])

    if not all([isinstance(x, str) for x in struct_files]):
        return None

    acceptable_keys = ['infer', 'true', 'init', 'null']
    unacceptable_keys = [x for x in struct_keys if x not in acceptable_keys]
    if len(unacceptable_keys) > 0:
        raise ValueError(f"Keys must be in [{', '.join(acceptable_keys)}]. "
                         f"Unrecognized key(s): {', '.join(unacceptable_keys)}")
    if primary_file is not None and primary_file not in struct_files:
        raise ValueError(f"Primary structure key ({primary_file}) must be in"
                         " structure files:"
                         f" [{', '.join(struct_files)}]")

    # Get output directory
    if 'infer' in struct_keys or 'init' in struct_keys:
        outdir = os.path.commonpath([
            os.path.dirname(v) for k, v in structures if k not in (
                'true', 'null')])
    else:
        outdir = os.path.commonpath([
            os.path.dirname(v) for k, v in structures])

    # Get output filename
    primary_name = ""
    secondary_name = []
    for key, struct_file in structures:
        struct_file = re.sub(r'\.coords$', '', os.path.basename(struct_file))
        if key != 'true' and re.match(
                r'(^|.+\.)([0-9]{3})($|\..+)', struct_file) is not None:
            seed = int(re.sub(
                r'(^|.+\.)([0-9]{3})($|\..+)', r'\2', struct_file))
            struct_name = f"{key}{seed:03d}"
        else:
            struct_name = key
        if primary_file is not None and struct_file == primary_file:
            primary_name = struct_name
        else:
            secondary_name.append(struct_name)
    secondary_name = '_'.join(secondary_name)
    if primary_name != "":
        primary_name = f"{primary_name}_vs_"

    if rescale:
        basename = f"error.{primary_name}{secondary_name}"
    else:
        basename = f"error_rescaled.{primary_name}{secondary_name}"
    outfile = os.path.join(outdir, basename)

    # infer_vs_true  true.btwn_hmlgs  infer.btwn_hmlgs  infer_vs_infer
    return outfile


def run_infer_vs_infer(idx, struct_infer, rescale=False, verbose=False):
    i, j = idx
    if i >= j:
        return

    error = []
    for primary_file in (struct_infer[i], struct_infer[j]):
        outfile = get_outfile(
            [('infer', struct_infer[i]), ('infer', struct_infer[j])],
            primary_file=struct_infer[i], rescale=rescale)
        if verbose:
            print(os.path.basename(outfile), flush=True)
        error.append(compare_btwn_struct(
            structX=struct_infer[i], structY=struct_infer[j], outfile=outfile,
            rescale_X_to_Y=rescale, verbose=False))
    return error


def run_infer_vs_null(idx, struct_infer, struct_null, rescale=False,
                      verbose=False):
    i, j = idx

    error = []
    for primary_file in (struct_infer[i], struct_null[j]):
        outfile = get_outfile(
            [('infer', struct_infer[i]), ('null', struct_null[j])],
            primary_file=primary_file, rescale=rescale)
        if verbose:
            print(os.path.basename(outfile), flush=True)
        error.append(compare_btwn_struct(
            structX=struct_infer[i], structY=struct_infer[j], outfile=outfile,
            rescale_X_to_Y=rescale, verbose=False))
    return error


def get_error_scores(struct_true=None, struct_infer=None, struct_null=None,
                     rescale=False, infer_vs_infer=False, verbose=True):

    # Parse structures
    if struct_true is None and struct_infer is None:
        raise ValueError("Must supply struct_true and/or struct_infer to"
                         " generate error scores")
    if isinstance(struct_true, str) and not os.path.exists(struct_true):
        raise ValueError(f"File does not exist: {struct_true}")
    if struct_infer is not None:
        if not isinstance(struct_infer, list):
            struct_infer = [struct_infer]
        for each_struct in struct_infer:
            if not os.path.exists(each_struct):
                raise ValueError(f"File does not exist: {each_struct}")
    if struct_null is not None:
        if not isinstance(struct_null, list):
            struct_null = [struct_null]
        for each_struct in struct_null:
            if not os.path.exists(each_struct):
                raise ValueError(f"File does not exist: {each_struct}")

    results = {}
    outfile = None

    # Get error scores between inferred and true structures
    if struct_true is not None and struct_infer is not None:
        error_all = []
        for i in range(len(struct_infer)):
            outfile = get_outfile(
                [('infer', struct_infer[i]), ('true', struct_true)],
                primary_file=struct_true, rescale=rescale)
            if verbose:
                print(os.path.basename(outfile), flush=True)
            error_all.append(compare_btwn_struct(
                structX=struct_infer[i], structY=struct_true, outfile=outfile,
                rescale_X_to_Y=rescale, verbose=False))
        error_combo = pd.DataFrame(error_all).mean()
        results['infer_vs_true'] = error_combo
        if verbose:
            print("\nError scores between the true & inferred structures",
                  flush=True)
            print(error_combo.map('{:.3g}'.format), flush=True)

    # If multiple inferred structures inputted, assess how similar they are
    if struct_infer is not None and len(struct_infer) > 1 and infer_vs_infer:
        all_idx = np.indices(
            (len(struct_infer), len(struct_infer))).reshape(2, -1).T
        all_idx = all_idx[all_idx[:, 0] < all_idx[:, 1]]
        error_all = []
        for idx in all_idx:
            error_all.extend(run_infer_vs_infer(
                idx=idx, struct_infer=struct_infer, rescale=rescale,
                verbose=False))
        error_combo = pd.DataFrame(error_all).mean()
        results['infer_vs_infer'] = error_combo
        if verbose:
            print(f"\nMean error scores between the {len(struct_infer)}"
                  " inferred structures.", flush=True)
            print(error_combo.map('{:.3g}'.format), flush=True)

    # Get error scores between inferred and null structures
    if struct_infer is not None and struct_null is not None:
        all_idx = np.indices(
            (len(struct_infer), len(struct_null))).reshape(2, -1).T
        error_all = []
        for idx in all_idx:
            error_all.extend(run_infer_vs_null(
                idx=idx, struct_infer=struct_infer, struct_null=struct_null,
                rescale=rescale, verbose=False))
        error_combo = pd.DataFrame(error_all).mean()
        results['infer_vs_null'] = error_combo
        if verbose:
            print(f"\nMean error scores between the {len(struct_infer)}"
                  " inferred structures and "
                  f"{len(struct_null)} null structures.", flush=True)
            print(error_combo.map('{:.3g}'.format), flush=True)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--true", type=str,
                        help="Simulated true structure file")
    parser.add_argument("--infer", type=str, nargs='+',
                        help="Inferred structure file(s)")
    parser.add_argument("--null", type=str, nargs='+',
                        help="Inferred NULL structure file(s)")
    parser.add_argument('--infer_vs_infer', default=False, action='store_true',
                        help="If inputting multiple inferred structures, also"
                             "compare among inferred structures")
    parser.add_argument('--rescale', default=False, action='store_true',
                        help="Whether to optimally rescale structures")
    parser.add_argument('--verbose', default=False, action='store_true',
                        help="Verbosity")

    args = parser.parse_args()

    get_error_scores(
        struct_true=args.true, struct_infer=args.infer, struct_null=args.null,
        infer_vs_infer=args.infer_vs_infer, rescale=args.rescale,
        verbose=args.verbose)


if __name__ == "__main__":
    main()
