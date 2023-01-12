import sys
import numpy as np
import warnings
from scipy import sparse
from scipy.interpolate import interp1d
from iced.io import load_lengths
from .utils_poisson import _struct_replace_nan

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")


def decrease_lengths_res(lengths, multiscale_factor):
    """Reduce resolution of chromosome lengths.

    Determine the number of beads per homolog of each chromosome at the
    specified resolution.

    Parameters
    ----------
    lengths : array_like of int
        Number of beads per homolog of each chromosome at current resolution.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 does not change the resolution.

    Returns
    -------
    array of int
        Number of beads per homolog of each chromosome at the given
        `multiscale_factor`.
    """

    if multiscale_factor == 1:
        return np.array(lengths, copy=False, ndmin=1, dtype=int)
    else:
        return np.ceil(
            np.asarray(lengths, dtype=float) / multiscale_factor).astype(int)


def increase_struct_res_gaussian(struct, current_multiscale_factor,
                                 final_multiscale_factor, lengths, ploidy,
                                 std_dev, random_state=None):
    # TODO remove, and also remove unit test
    """Estimate a high-resolution structure from a low-resolution structure.

    Increase resolution of a structure by assuming that the difference between
    each high-resolution bead and it's corresponding low-resolution bead
    is normally distributed along each of the 3 axes  (with mean = 0 and
    standard deviation = epsilon / sqrt(2).)

    Parameters
    ----------
    struct : array of float
        3D chromatin structure at the current low resolution.
    current_multiscale_factor : int
        Multiscale factor of the current low-resolution structure.
    multiscale_factor : int
        Desired multiscale factor of the output (higher-resolution) structure.
    lengths : array_like of int
        Number of beads per homolog of each chromosome at FULL resolution
        (at multiscale_factor of 1).
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    std_dev : float
        Standard deviation of the normal distributions.
    seed : int, optional
        Random seed used when sampling from the normal distributions.

    Returns
    -------
    struct_highres : array of float
        3D chromatin structure that has been updated to match the specified high
        resolution.
    """

    if current_multiscale_factor == final_multiscale_factor:
        return struct

    rescale_by = current_multiscale_factor / final_multiscale_factor
    if rescale_by != int(rescale_by):
        raise ValueError(
            'The factor by which to increase resolution must be an integer. The'
            f' current multiscale factor is {current_multiscale_factor} and the'
            f' desired output multiscale factor is {final_multiscale_factor}'
            f' (ratio of {rescale_by:.3g})')

    if isinstance(struct, str):
        struct = np.loadtxt(struct)
    struct = struct.reshape(-1, 3)
    if isinstance(lengths, str):
        lengths = load_lengths(lengths)
    lengths = np.array(lengths, copy=False, ndmin=1, dtype=int)
    lengths_current = decrease_lengths_res(
        lengths=lengths, multiscale_factor=current_multiscale_factor)
    if random_state is None:
        random_state = np.random.RandomState(0)

    # Replace NaN in low-res struct via linear interpolation
    struct = _struct_replace_nan(
        struct, lengths=lengths, ploidy=ploidy, kind='linear',
        random_state=random_state)

    # Estimate full-resolution structure
    grouped_idx, bad_idx = _get_struct_index(
        multiscale_factor=current_multiscale_factor, lengths=lengths,
        ploidy=ploidy)
    grouped_idx = grouped_idx.astype(float)
    grouped_idx[bad_idx] = np.nan

    struct_fullres = []
    for i in range(struct.shape[0]):
        lowres_bead = struct[i]
        num_highres_beads = np.invert(np.isnan(grouped_idx[:, i])).sum()
        highres_beads = random_state.normal(
            lowres_bead, std_dev, (num_highres_beads, 3))
        struct_fullres.append(highres_beads)
    struct_fullres = np.concatenate(struct_fullres)

    if final_multiscale_factor == 1:
        struct_highres = struct_fullres
    else:
        struct_highres = decrease_struct_res(
            struct_fullres, multiscale_factor=final_multiscale_factor,
            lengths=lengths, ploidy=ploidy)

    return struct_highres


def increase_struct_res(struct, multiscale_factor, lengths, ploidy,
                        random_state=None):
    """Linearly interpolate structure to increase resolution.

    Increase resolution of structure via linear interpolation between beads.

    Parameters
    ----------
    struct : array of float
        3D chromatin structure at the current low resolution.
    multiscale_factor : int
        Factor by which to increase the resolution. A value of 2 doubles the
        resolution. A value of 1 does not change the resolution.
    lengths : array_like of int
        Number of beads per homolog of each chromosome at increased resolution
        (the desired resolution of the output structure).
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.

    Returns
    -------
    struct_highres : array of float
        3D chromatin structure that has been linearly interpolated to the
        specified high resolution.
    """

    # Setup
    if random_state is None:
        random_state = np.random.RandomState(seed=0)
    if int(multiscale_factor) != multiscale_factor:
        raise ValueError('The multiscale_factor must be an integer')
    multiscale_factor = int(multiscale_factor)
    if multiscale_factor == 1:
        return struct
    if isinstance(struct, str):
        struct = np.loadtxt(struct)
    struct = struct.reshape(-1, 3)
    if isinstance(lengths, str):
        lengths = load_lengths(lengths)
    lengths = np.array(lengths, copy=False, ndmin=1, dtype=int)
    lengths_lowres = decrease_lengths_res(
        lengths=lengths, multiscale_factor=multiscale_factor)

    # Get highres idx
    idx, bad_idx = _get_struct_index(
        multiscale_factor=multiscale_factor, lengths=lengths, ploidy=ploidy)
    idx = idx.astype(float)
    idx[bad_idx] = np.nan

    struct_highres = np.full((lengths.sum() * ploidy, 3), np.nan)
    begin_lowres = end_lowres = begin_highres = end_highres = 0
    for i in range(lengths.size * ploidy):
        end_lowres += np.tile(lengths_lowres, ploidy)[i]
        end_highres += np.tile(lengths, ploidy)[i]

        # Get index for this molecule at low & high res
        nan_mask_lowres = np.isnan(struct[begin_lowres:end_lowres, 0])
        chrom_idx = idx[:, begin_lowres:end_lowres]
        chrom_xloc_lowres = np.nanmean(chrom_idx[:, ~nan_mask_lowres], axis=0)
        chrom_idx_lowres = np.arange(begin_lowres, end_lowres)[~nan_mask_lowres]
        chrom_idx_highres = np.arange(begin_highres, end_highres)

        # Create highres beads
        if (~nan_mask_lowres).sum() == 0:  # 0 non-NaN beads in lowres
            chrom_idx_highres = chrom_idx_highres[
                ~np.isnan(chrom_idx_highres)].astype(int)
            random_chrom = random_state.uniform(
                low=-1, high=1, size=(chrom_idx_highres.size, 3))
            struct_highres[chrom_idx_highres] = random_chrom
        elif (~nan_mask_lowres).sum() == 1:  # Only 1 non-NaN bead in lowres
            chrom_idx_highres = chrom_idx_highres[
                ~np.isnan(chrom_idx_highres)].astype(int)
            lowres_bead = struct[chrom_idx_lowres]
            struct_highres[chrom_idx_highres] = random_state.normal(
                lowres_bead, 1, (chrom_idx_highres.size, 3))
        else:  # Enough non-NaN beads in lowres to interpolate
            struct_highres[chrom_idx_highres, 0] = interp1d(
                x=chrom_xloc_lowres, y=struct[chrom_idx_lowres, 0],
                kind="linear", fill_value="extrapolate")(chrom_idx_highres)
            struct_highres[chrom_idx_highres, 1] = interp1d(
                x=chrom_xloc_lowres, y=struct[chrom_idx_lowres, 1],
                kind="linear", fill_value="extrapolate")(chrom_idx_highres)
            struct_highres[chrom_idx_highres, 2] = interp1d(
                x=chrom_xloc_lowres, y=struct[chrom_idx_lowres, 2],
                kind="linear", fill_value="extrapolate")(chrom_idx_highres)

        begin_lowres = end_lowres
        begin_highres = end_highres

    return struct_highres


def _group_counts_multiscale(counts, lengths, ploidy, multiscale_factor=1,
                             exclude_zeros=False):
    """Group together full-res counts corresponding to a given low-res distance.

    Prepare counts for multi-resolution optimization by aggregating sets of
    full-res counts bins, such that each set corresponds to a single low-res
    distance bin.

    Parameters
    ----------
    counts : coo_matrix
        Counts data at full resolution.
    lengths : array_like of int
        Number of beads per homolog of each chromosome at full resolution.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 does not change the resolution.

    Returns
    -------
    data_grouped : array
        TODO
    idx_lowres : tuple of arrays
        TODO
    shape_lowres : tuple of int
        TODO
    mask : array
        TODO
    """
    from .counts import _check_counts_matrix, _get_included_counts_bins

    counts = _check_counts_matrix(counts, lengths=lengths, ploidy=ploidy)

    if multiscale_factor > 1:
        lengths_lowres = decrease_lengths_res(
            lengths, multiscale_factor=multiscale_factor)
        shape_lowres = tuple(np.array(
            counts.shape / lengths.sum() * lengths_lowres.sum(), dtype=int))

        idx_fullres, idx_lowres = _get_fullres_counts_index(
            multiscale_factor=multiscale_factor, lengths=lengths, ploidy=ploidy,
            counts_fullres_shape=counts.shape)
        row_fullres, col_fullres, _ = idx_fullres
        row_lowres, col_lowres = idx_lowres

        data_grouped = counts.toarray()[row_fullres, col_fullres].reshape(
            multiscale_factor ** 2, -1)

        # Get mask of included counts bins
        # If bins aren't included, set them to zero
        mask = _get_included_counts_bins(
            counts, lengths=lengths, ploidy=ploidy, check_counts=False,
            exclude_zeros=exclude_zeros)[row_fullres, col_fullres].reshape(
            multiscale_factor ** 2, -1)
        data_grouped[~mask] = 0

        # If every single full-res bin in a group is either zero or excluded,
        # remove that group from the output.
        not_all_zero = np.sum(data_grouped, axis=0) != 0
        data_grouped = data_grouped[:, not_all_zero]
        mask = mask[:, not_all_zero]
        if np.all(mask):
            mask = None
        idx_lowres = row_lowres[not_all_zero], col_lowres[not_all_zero]
    else:
        idx_lowres = counts.row, counts.col
        data_grouped = counts.data
        shape_lowres = counts.shape
        mask = None

    return data_grouped, idx_lowres, shape_lowres, mask


def decrease_counts_res(counts, multiscale_factor, lengths, ploidy,
                        return_indices=False, remove_diag=True):
    """Decrease resolution of counts matrices by summing adjacent bins.

    Decrease the resolution of the contact counts matrices. Each bin in a
    low-resolution counts matrix is the sum of corresponding high-resolution
    counts matrix bins.

    Parameters
    ----------
    counts : array or coo_matrix
        Counts data at full resolution, ideally without normalization or
        filtering.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 does not change the resolution.
    lengths : array_like of int
        Number of beads per homolog of each chromosome at full resolution.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.

    Returns
    -------
    counts_lowres : coo_matrix
        Counts data at reduced resolution, as specified by the given
        `multiscale_factor`.
    """

    from .counts import _check_counts_matrix

    counts = _check_counts_matrix(
        counts, lengths=lengths, ploidy=ploidy)

    if multiscale_factor == 1:
        if return_indices:
            return counts, counts.row, counts.col
        else:
            return counts

    lengths_lowres = decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor)

    counts_lowres_shape = tuple(np.array(
        counts.shape / lengths.sum() * lengths_lowres.sum(), dtype=int))

    idx_fullres, idx_lowres = _get_fullres_counts_index(
        multiscale_factor=multiscale_factor, lengths=lengths, ploidy=ploidy,
        counts_fullres_shape=counts.shape, remove_diag=remove_diag)
    row_fullres, col_fullres, _ = idx_fullres
    row_lowres, col_lowres = idx_lowres

    data_lowres = counts.toarray()[row_fullres, col_fullres].reshape(
        multiscale_factor ** 2, -1).sum(axis=0)
    counts_lowres = sparse.coo_matrix(
        (data_lowres[data_lowres != 0],
            (row_lowres[data_lowres != 0], col_lowres[data_lowres != 0])),
        shape=counts_lowres_shape)

    if return_indices:
        return counts_lowres, row_fullres, col_fullres
    else:
        return counts_lowres


def _get_lowres_counts_index(multiscale_factor, lengths, ploidy,
                             counts_fullres_shape, remove_diag=True):
    """Get indices of generic low-res counts matrix for the given ambiguity"""
    from .counts import _get_included_counts_bins

    if counts_fullres_shape is None:
        raise ValueError("Must input counts_fullres_shape")

    # Get rows & cols of dummy low-res counts matrix
    lengths_lowres = decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor)
    counts_lowres_shape = np.array(
        counts_fullres_shape / lengths.sum() * lengths_lowres.sum(), dtype=int)

    dummy_counts_lowres = sparse.coo_matrix(_get_included_counts_bins(
        np.ones(counts_lowres_shape, dtype=np.uint8), lengths=lengths_lowres,
        ploidy=ploidy, check_counts=False,
        remove_diag=remove_diag).astype(np.uint8))
    row_lowres = dummy_counts_lowres.row
    col_lowres = dummy_counts_lowres.col

    return row_lowres, col_lowres


def _get_fullres_counts_index(multiscale_factor, lengths, ploidy,
                              lowres_idx=None, counts_fullres_shape=None,
                              remove_diag=True):
    """Convert low-res indices to full-res indices.

    Return full-res counts indices grouped by the corresponding low-res bin.
    If low-res indices are not provided, indices of a generic low-res counts
    matrix will be created.
    """
    if lowres_idx is None:
        row_lowres, col_lowres = _get_lowres_counts_index(
            multiscale_factor=multiscale_factor, lengths=lengths, ploidy=ploidy,
            counts_fullres_shape=counts_fullres_shape, remove_diag=remove_diag)
    else:
        row_lowres, col_lowres = lowres_idx

    # Get full-res bead indices, grouped by low-res bead
    idx, bad_idx = _get_struct_index(
        multiscale_factor=multiscale_factor, lengths=lengths, ploidy=ploidy)
    idx = idx.T
    bad_idx = bad_idx.T

    # Create full-res counts indices, grouped by low-res counts bin
    row = idx[row_lowres]
    row = np.repeat(row, multiscale_factor, axis=1)
    col = idx[col_lowres]
    col = np.tile(col, (1, multiscale_factor))
    bad_row = bad_idx[row_lowres]
    bad_row = np.repeat(bad_row, multiscale_factor, axis=1)
    bad_col = bad_idx[col_lowres]
    bad_col = np.tile(bad_col, (1, multiscale_factor))
    bad_idx = bad_row | bad_col
    row[bad_idx] = 0
    col[bad_idx] = 0

    row = row.T.flatten()
    col = col.T.flatten()
    bad_idx = bad_idx.T.flatten()
    return (row, col, bad_idx), (row_lowres, col_lowres)


def _get_struct_index(multiscale_factor, lengths, ploidy):
    """Return full-res struct index grouped by the corresponding low-res bead.
    """

    lengths_lowres = decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor)

    remainders = np.mod(lengths, multiscale_factor)
    num_false = multiscale_factor - remainders[remainders != 0]
    where_false = lengths_lowres.cumsum()[remainders != 0] - 1

    mask = np.full((lengths_lowres.sum(), multiscale_factor), True)
    for i in range(num_false.shape[0]):
        mask[where_false[i], -num_false[i]:] = False
    mask = np.tile(mask, (ploidy, 1))

    idx = np.zeros(
        (lengths_lowres.sum() * ploidy, multiscale_factor), dtype=int)
    idx[mask] = np.arange(lengths.sum() * ploidy)
    idx = idx.T
    bad_idx = ~mask.T

    return idx, bad_idx


def _group_highres_struct(struct, multiscale_factor, lengths, ploidy,
                          fullres_struct_nan=None):
    """Group beads of full-res struct by the low-res bead they correspond to.

    Axes of final array:
        0: all highres beads corresponding to each lowres bead, size = multiscale_factor
        1: beads, size = struct[0]
        2: coordinates, size = struct[1] = 3
    """

    if multiscale_factor == 1:
        return struct.reshape(1, -1, 3)

    idx, bad_idx = _get_struct_index(
        multiscale_factor=multiscale_factor, lengths=lengths, ploidy=ploidy)

    if fullres_struct_nan is not None and fullres_struct_nan.shape != 0:
        nan_mask_lowres = np.isin(idx, fullres_struct_nan)
        bad_idx[nan_mask_lowres] = True
        idx[nan_mask_lowres] = 0

    # Apply to struct, and set incorrect idx to np.nan
    grouped_struct = np.where(
        np.repeat(bad_idx.reshape(-1, 1), 3, axis=1), np.nan,
        struct.reshape(-1, 3)[idx.flatten(), :]).reshape(
        multiscale_factor, -1, 3)

    return grouped_struct


def decrease_struct_res(struct, multiscale_factor, lengths, ploidy,
                        fullres_struct_nan=None):
    """Decrease resolution of structure by averaging adjacent beads.

    Decrease the resolution of the 3D chromatin structure. Each bead in the
    low-resolution structure is the mean of corresponding beads in the
    high-resolution structure.

    Parameters
    ----------
    struct : array of float
        3D chromatin structure at full resolution.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 does not change the resolution.
    lengths : array_like of int
        Number of beads per homolog of each chromosome at full resolution.

    Returns
    -------
    array of float
        3D chromatin structure at reduced resolution, as specified by the given
        `multiscale_factor`.
    """

    if multiscale_factor == 1:
        return struct

    grouped_struct = _group_highres_struct(
        struct, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy, fullres_struct_nan=fullres_struct_nan)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message="Mean of empty slice")
        struct_lowres = np.nanmean(grouped_struct, axis=0)

    return struct_lowres


def _count_fullres_per_lowres_bead(multiscale_factor, lengths, ploidy,
                                   fullres_struct_nan=None):
    """Count the number of full-res beads corresponding to each low-res bead.
    """

    if multiscale_factor == 1:
        return None

    fullres_idx, bad_idx = _get_struct_index(
        multiscale_factor=multiscale_factor,
        lengths=lengths, ploidy=ploidy)

    if fullres_struct_nan is not None and fullres_struct_nan.size != 0:
        n = lengths.sum()
        if ploidy == 1 and fullres_struct_nan.max() >= n:
            fullres_struct_nan[fullres_struct_nan >= n] -= n
            fullres_struct_nan = np.unique(fullres_struct_nan)
        if fullres_struct_nan.max() >= ploidy * n:
            raise ValueError("Inconsistent number of beads.")
        bad_idx[np.isin(fullres_idx, fullres_struct_nan)] = True

    return (~bad_idx).sum(axis=0)


# def _get_stretch_of_fullres_beads(multiscale_factor, lengths, ploidy,
#                                   fullres_struct_nan=None):
#     """TODO
#     """

#     if multiscale_factor == 1:
#         return None

#     fullres_idx, bad_idx = _get_struct_index(
#         multiscale_factor=multiscale_factor,
#         lengths=lengths, ploidy=ploidy)

#     if fullres_struct_nan is not None and fullres_struct_nan.size != 0:
#         n = lengths.sum()
#         if ploidy == 1 and fullres_struct_nan.max() >= n:
#             fullres_struct_nan[fullres_struct_nan >= n] -= n
#             fullres_struct_nan = np.unique(fullres_struct_nan)
#         if fullres_struct_nan.max() >= ploidy * n:
#             raise ValueError("Inconsistent number of beads.")
#         bad_idx[np.isin(fullres_idx, fullres_struct_nan)] = True

#     arr = np.tile(
#         np.arange(bad_idx.shape[0], dtype=float).reshape(-1, 1),
#         (1, bad_idx.shape[1]))
#     arr[bad_idx] = np.nan
#     with warnings.catch_warnings():
#         warnings.filterwarnings('ignore', message='All-NaN slice encountered',
#                                 category=RuntimeWarning)
#         stretch_fullres_beads = np.nanmax(arr, axis=0) - np.nanmin(
#             arr, axis=0) + 1
#     stretch_fullres_beads[np.isnan(stretch_fullres_beads)] = 0

#     print('\n')
#     print('total =', (stretch_fullres_beads != 0).sum())
#     print('one =', (stretch_fullres_beads == 1).sum())
#     print('two =', (stretch_fullres_beads == 2).sum())
#     print('multiscale_factor =', (stretch_fullres_beads == multiscale_factor).sum())
#     print(stretch_fullres_beads[~np.isin(stretch_fullres_beads, [0, 1, 2, multiscale_factor])])
#     print('\n')

#     return stretch_fullres_beads


# def get_multiscale_variances_from_struct(structures, lengths, multiscale_factor,
#                                          mixture_coefs=None, replace_nan=True,
#                                          verbose=True):
#     """Compute multiscale variances from full-res structure.

#     Generates multiscale variances at the specified resolution from the
#     inputted full-resolution structure(s). Multiscale variances are defined as
#     follows: for each low-resolution bead, the variances of the distances
#     between all high-resolution beads that correspond to that low-resolution
#     bead.

#     Parameters
#     ----------
#     structures : array of float or list of array of float
#         3D chromatin structure(s) at full resolution.
#     lengths : array_like of int
#         Number of beads per homolog of each chromosome at full resolution.
#     multiscale_factor : int, optional
#         Factor by which to reduce the resolution. A value of 2 halves the
#         resolution. A value of 1 does not change the resolution.

#     Returns
#     -------
#     array of float
#         Multiscale variances: for each low-resolution bead, the variances of the
#         distances between all high-resolution beads that correspond to that
#         low-resolution bead.
#     """

#     from .utils_poisson import _format_structures

#     if multiscale_factor == 1:
#         return None

#     structures = _format_structures(structures, mixture_coefs=mixture_coefs)
#     struct_length = set([s.shape[0] for s in structures])
#     if len(struct_length) > 1:
#         raise ValueError("Structures are of different shapes.")
#     else:
#         struct_length = struct_length.pop()
#     if struct_length / lengths.sum() not in (1, 2):
#         raise ValueError("Structures do not appear to be haploid or diploid.")
#     ploidy = int(struct_length / lengths.sum())
#     structures = _format_structures(structures, lengths=lengths,
#                                     ploidy=ploidy, mixture_coefs=mixture_coefs)

#     multiscale_variances = []
#     for struct in structures:
#         struct_grouped = _group_highres_struct(
#             struct, multiscale_factor=multiscale_factor, lengths=lengths,
#             ploidy=ploidy)
#         multiscale_variances.append(
#             _var3d(struct_grouped, replace_nan=replace_nan))
#     multiscale_variances = np.mean(multiscale_variances, axis=0)

#     if verbose:
#         print(f"MULTISCALE VARIANCE ({multiscale_factor}x):"
#               f" {np.median(multiscale_variances):.3g}", flush=True)

#     return multiscale_variances


def _var3d(struct_grouped, replace_nan=True):
    """Compute variance of beads in 3D.
    """

    # struct_grouped.shape = (multiscale_factor, nbeads, 3)
    multiscale_variances = np.full(struct_grouped.shape[1], np.nan)
    for i in range(struct_grouped.shape[1]):
        struct_group = struct_grouped[:, i, :]
        beads_in_group = np.invert(np.isnan(struct_group[:, 0])).sum()
        if beads_in_group < 1:  # FIXME bugfix 8/28/20... previously was <= 1
            var = np.nan
        else:
            mean_coords = np.nanmean(struct_group, axis=0)
            # Euclidian distance formula = ((A - B) ** 2).sum(axis=1) ** 0.5
            var = (1 / beads_in_group) * \
                np.nansum((struct_group - mean_coords) ** 2)
        multiscale_variances[i] = var

    if np.isnan(multiscale_variances).sum() == multiscale_variances.shape[0]:
        raise ValueError("Multiscale variances are nan for every bead.")

    if replace_nan:
        multiscale_variances[np.isnan(multiscale_variances)] = np.nanmedian(
            multiscale_variances)

    return multiscale_variances


# def get_multiscale_epsilon_from_dis(structures, lengths, multiscale_factor,
#                                     mixture_coefs=None, verbose=True):
#     """Compute multiscale epsilon from full-res structure.
#     """

#     from .utils_poisson import _format_structures

#     if multiscale_factor == 1:
#         return None

#     structures = _format_structures(structures, mixture_coefs=mixture_coefs)
#     struct_length = set([s.shape[0] for s in structures])
#     if len(struct_length) > 1:
#         raise ValueError("Structures are of different shapes.")
#     else:
#         struct_length = struct_length.pop()
#     if struct_length / lengths.sum() not in (1, 2):
#         raise ValueError("Structures do not appear to be haploid or diploid.")
#     ploidy = int(struct_length / lengths.sum())
#     structures = _format_structures(structures, lengths=lengths,
#                                     ploidy=ploidy, mixture_coefs=mixture_coefs)

#     std_all = []
#     for struct in structures:
#         mask = np.invert(np.isnan(structures[0][:, 0]))
#         dummy = sparse.coo_matrix(np.triu(
#             np.ones((mask.sum(), mask.sum())), 1))
#         dis = np.sqrt((np.square(
#             struct[mask][dummy.row] - struct[mask][dummy.col])).sum(axis=1))
#         dis_coo = sparse.coo_matrix(
#             (dis, (dummy.row, dummy.col)), shape=dummy.shape)
#         with warnings.catch_warnings():
#             warnings.filterwarnings(
#                 "ignore", category=UserWarning,
#                 message="Counts matrix must only contain integers or NaN")
#             dis_grouped = _group_counts_multiscale(
#                 dis_coo, lengths=lengths, ploidy=ploidy,
#                 multiscale_factor=multiscale_factor)[0]
#         std_all.append(_get_epsilon_from_dis(dis_grouped))
#     std_all = np.mean(std_all)

#     if verbose:
#         print(f"MULTISCALE EPSILON (estimate) ({multiscale_factor}x):"
#               f" {std_all:.3g}", flush=True)

#     return std_all


# def _get_epsilon_from_dis(dis_grouped, alpha=-3.):
#     """TODO
#     """

#     from scipy.stats import norm, mode
#     #import plotille  # FIXME
#     #from topsy.datasets.samples_generator import get_diff_coords_from_euc_dis

#     # dis_grouped.shape = (multiscale_factor ** 2, nbins)
#     tmp = []
#     for i in range(dis_grouped.shape[1]):
#         dis_fullres = dis_grouped[:, i]
#         if (dis_fullres == 0).sum() > 0:  # Lowres dist bin is on diagonal
#             print((dis_fullres == 0).sum(), (dis_fullres != 0).sum())
#             continue
#         fake_counts_lowres = (dis_fullres ** alpha).sum()
#         dis_lowres = fake_counts_lowres ** (1 / alpha)
#         tmp.append(dis_fullres.flatten() - dis_lowres)

#     tmp = np.concatenate(tmp)
#     est_diff = tmp #* np.sqrt(1 / 3)

#     est_diff_rounded = np.array([float(f"{x:.2g}") for x in est_diff])
#     est_diff_mode = mode(est_diff_rounded, axis=None)[0][0]

#     #est_diff -= np.median(est_diff)
#     #est_diff -= est_diff_mode
#     #est_diff = np.append(est_diff, -est_diff)

#     # fig = plotille.Figure()
#     # fig.height = 25
#     # fig.set_y_limits(min_=0)
#     # fig.histogram(est_diff, bins=160, lc=None)
#     # fig.plot([0, 0], [0, 2400])
#     # fig.plot([np.median(est_diff), np.median(est_diff)], [0, 2400])
#     # fig.plot([np.mean(est_diff), np.mean(est_diff)], [0, 2400])
#     # #fig.plot([est_diff_mode, est_diff_mode], [0, 2400])
#     # print(fig.show(legend=False))

#     mu, stddev = norm.fit(est_diff)
#     #print(f'MIN={est_diff.min():.3g}    MU={mu:.3g}   MEAN={np.mean(est_diff):.3g}   MED={np.median(est_diff):.3g}')

#     if np.isnan(stddev):
#         raise ValueError("Multiscale dist stddev is nan.")

#     return stddev


def get_epsilon_from_struct(structures, lengths, multiscale_factor,
                            mixture_coefs=None, replace_nan=True, verbose=True):
    """Compute multiscale epsilon from full-res structure.

    Generates multiscale epsilons at the specified resolution from the
    inputted full-resolution structure(s). Multiscale epsilons are defined as
    follows: for each low-resolution bead, the variances of the distances
    between all high-resolution beads that correspond to that low-resolution
    bead. TODO update

    Parameters
    ----------
    structures : array of float or list of array of float
        3D chromatin structure(s) at full resolution.
    lengths : array_like of int
        Number of beads per homolog of each chromosome at full resolution.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 does not change the resolution.

    Returns
    -------
    array of float
        Multiscale variances: for each low-resolution bead, the variances of the
        distances between all high-resolution beads that correspond to that
        low-resolution bead.
    """

    from .utils_poisson import _format_structures

    if multiscale_factor == 1:
        return None

    structures = _format_structures(structures, mixture_coefs=mixture_coefs)
    struct_length = set([s.shape[0] for s in structures])
    if len(struct_length) > 1:
        raise ValueError("Structures are of different shapes.")
    else:
        struct_length = struct_length.pop()
    if struct_length / lengths.sum() not in (1, 2):
        raise ValueError("Structures do not appear to be haploid or diploid.")
    ploidy = int(struct_length / lengths.sum())
    structures = _format_structures(structures, lengths=lengths,
                                    ploidy=ploidy, mixture_coefs=mixture_coefs)

    multiscale_variances = []
    for struct in structures:
        struct_grouped = _group_highres_struct(
            struct, multiscale_factor=multiscale_factor, lengths=lengths,
            ploidy=ploidy)
        multiscale_variances.append(
            _var3d(struct_grouped, replace_nan=replace_nan))
    multiscale_variances = np.mean(multiscale_variances, axis=0)  # ie >1 struct

    multiscale_epsilon = np.sqrt(multiscale_variances * 2 / 3)

    if verbose:
        print(f"MULTISCALE EPSILON ({multiscale_factor}x):"
              f" {np.mean(multiscale_epsilon):.3g}", flush=True)

    return multiscale_epsilon



# def get_epsilon_from_struct_old(structures, lengths, multiscale_factor,
#                                            mixture_coefs=None, replace_nan=True,
#                                            verbose=True):
#     """Compute multiscale epsilon from full-res structure.
#     """

#     from .utils_poisson import _format_structures

#     if multiscale_factor == 1:
#         return None

#     structures = _format_structures(structures, mixture_coefs=mixture_coefs)
#     struct_length = set([s.shape[0] for s in structures])
#     if len(struct_length) > 1:
#         raise ValueError("Structures are of different shapes.")
#     else:
#         struct_length = struct_length.pop()
#     if struct_length / lengths.sum() not in (1, 2):
#         raise ValueError("Structures do not appear to be haploid or diploid.")
#     ploidy = int(struct_length / lengths.sum())
#     structures = _format_structures(structures, lengths=lengths,
#                                     ploidy=ploidy, mixture_coefs=mixture_coefs)

#     std_per_bead = []
#     std_all = []
#     for struct in structures:
#         struct_grouped = _group_highres_struct(
#             struct, multiscale_factor=multiscale_factor, lengths=lengths,
#             ploidy=ploidy)
#         std_per_bead_tmp, std_all_tmp = _get_epsilon(
#             struct_grouped, replace_nan=replace_nan)
#         std_per_bead.append(std_per_bead_tmp)
#         std_all.append(std_all_tmp)
#     std_per_bead = np.mean(std_per_bead, axis=0)
#     std_all = np.mean(std_all)

#     if verbose:
#         print(f"MULTISCALE EPSILON per-bead  ({multiscale_factor}x):"
#               f" {np.median(std_per_bead):.3g}", flush=True)
#         print(f"MULTISCALE EPSILON all-beads ({multiscale_factor}x):"
#               f" {std_all:.3g}", flush=True)

#     return std_all, std_per_bead


# def _get_epsilon(struct_grouped, replace_nan=True):
#     """TODO
#     """

#     from scipy.stats import norm

#     # struct_grouped.shape = (multiscale_factor, nbeads, 3)
#     mu_per_bead = np.full(struct_grouped.shape[1], np.nan)
#     std_per_bead = np.full(struct_grouped.shape[1], np.nan)
#     all_diff = []
#     for i in range(struct_grouped.shape[1]):
#         struct_group = struct_grouped[:, i, :]
#         beads_in_group = np.invert(np.isnan(struct_group[:, 0])).sum()
#         if beads_in_group == 0:
#             stddev = mu = np.nan
#         else:
#             mean_coords = np.nanmean(struct_group, axis=0)
#             diff = struct_group - mean_coords
#             diff *= 2  # FIXME IS THIS CORRECT?
#             diff = diff[~np.isnan(diff[:, 0])]
#             all_diff.append(diff)
#             mu, stddev = norm.fit(diff)
#         std_per_bead[i] = stddev
#         mu_per_bead[i] = mu

#     if np.isnan(std_per_bead).sum() == std_per_bead.shape[0]:
#         raise ValueError("Multiscale stddev are nan for every bead.")

#     all_diff = np.concatenate(all_diff)
#     mu_all, std_all = norm.fit(all_diff)

#     if not np.isclose(np.median(mu_per_bead), 0):
#         warnings.warn(f"Multiscale mu (per-bead) is {np.median(mu_per_bead)}, expected 0.")
#     if not np.isclose(mu_all, 0):
#         warnings.warn(f"Multiscale mu (all-beads) is {np.median(mu_all)}, expected 0.")

#     if replace_nan:
#         std_per_bead[np.isnan(std_per_bead)] = np.nanmedian(
#             std_per_bead)

#     return std_per_bead, std_all


def _choose_max_multiscale_factor(lengths, min_beads):
    """Choose the maximum multiscale factor, given `min_beads`.
    """

    multiscale_factor = 1
    while decrease_lengths_res(
            lengths, multiscale_factor * 2).min() >= min_beads:
        multiscale_factor *= 2
    return multiscale_factor


def _choose_max_multiscale_rounds(lengths, min_beads):
    """Choose the maximum number of multiscale rounds, given `min_beads`.
    """

    multiscale_factor = _choose_max_multiscale_factor(
        lengths, min_beads=min_beads)
    multiscale_rounds = np.log2(multiscale_factor) + 1
    return multiscale_rounds
