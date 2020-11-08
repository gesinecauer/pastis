import sys
import numpy as np
from scipy import sparse
from scipy.interpolate import interp1d
import warnings
from iced.io import load_lengths

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

    return np.ceil(
        np.array(lengths).astype(float) / multiscale_factor).astype(int)


def increase_struct_res(struct, multiscale_factor, lengths, mask=None):
    """Linearly interpolate structure to increase resolution.

    Increase resolution of structure via linear interpolation between beads.

    Parameters
    ----------
    struct : array of float
        3D chromatin structure at low resolution.
    lengths : array_like of int
        Number of beads per homolog of each chromosome at high resolution (the
        desired resolution of the output structure).
    multiscale_factor : int, optional
        Factor by which to increase the resolution. A value of 2 doubles the
        resolution. A value of 1 does not change the resolution.

    Returns
    -------
    struct_highres : array of float
        3D chromatin structure that has been linearly interpolated to the
        specified high resolution.
    """

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
    lengths = np.array(lengths).astype(int)
    lengths_lowres = decrease_lengths_res(
        lengths=lengths, multiscale_factor=multiscale_factor)
    ploidy = struct.shape[0] / lengths_lowres.sum()
    if ploidy != 1 and ploidy != 2:
        raise ValueError("Not consistent with haploid or diploid... struct is"
                         " %d beads (and 3 cols), sum of lengths is %d" %
                         (struct.reshape(-1, 3).shape[0], lengths_lowres.sum()))
    ploidy = int(ploidy)

    indices = _get_struct_indices(
        ploidy=ploidy, multiscale_factor=multiscale_factor,
        lengths=lengths).reshape(multiscale_factor, -1)
    if mask is not None:
        indices[~mask.reshape(multiscale_factor, -1)] = np.nan

    struct_highres = np.full((lengths.sum() * ploidy, 3), np.nan)
    begin_lowres, end_lowres = 0, 0
    for i in range(lengths.shape[0] * ploidy):
        end_lowres += np.tile(lengths_lowres, ploidy)[i]

        # Beads of struct that are NaN
        struct_nan = np.isnan(struct[begin_lowres:end_lowres, 0])

        # Get indices for this chrom at low & high res
        chrom_indices = indices[:, begin_lowres:end_lowres]
        chrom_indices[:, struct_nan] = np.nan
        chrom_indices_lowres = np.nanmean(chrom_indices, axis=0)
        chrom_indices_highres = chrom_indices.T.flatten()

        # Note which beads are unknown
        highres_mask = ~np.isnan(chrom_indices_highres)
        highres_mask[highres_mask] = (chrom_indices_highres[highres_mask] >=
                                      np.nanmin(chrom_indices_lowres)) & (chrom_indices_highres[highres_mask] <= np.nanmax(chrom_indices_lowres))
        unknown_beads = np.where(~highres_mask)[
            0] + np.tile(lengths, ploidy)[:i].sum()
        unknown_beads = unknown_beads[unknown_beads < np.tile(lengths, ploidy)[
            :i + 1].sum()]
        unknown_beads_at_begin = [unknown_beads[k] for k in range(len(unknown_beads)) if unknown_beads[
            k] == unknown_beads.min() or all([unknown_beads[k] - j == unknown_beads[k - j] for j in range(k + 1)])]
        if len(unknown_beads) - len(unknown_beads_at_begin) > 0:
            unknown_beads_at_end = [unknown_beads[k] for k in range(len(unknown_beads)) if unknown_beads[k] == unknown_beads.max(
            ) or all([unknown_beads[k] + j == unknown_beads[k + j] for j in range(len(unknown_beads) - k)])]
            chrom_indices_highres = np.arange(
                max(unknown_beads_at_begin) + 1, min(unknown_beads_at_end))
        else:
            unknown_beads_at_end = []
            chrom_indices_highres = np.arange(
                max(unknown_beads_at_begin) + 1, int(np.nanmax(chrom_indices_highres)) + 1)

        struct_highres[chrom_indices_highres, 0] = interp1d(
            chrom_indices_lowres[~struct_nan],
            struct[begin_lowres:end_lowres, 0][~struct_nan],
            kind="linear")(chrom_indices_highres)
        struct_highres[chrom_indices_highres, 1] = interp1d(
            chrom_indices_lowres[~struct_nan],
            struct[begin_lowres:end_lowres, 1][~struct_nan],
            kind="linear")(chrom_indices_highres)
        struct_highres[chrom_indices_highres, 2] = interp1d(
            chrom_indices_lowres[~struct_nan],
            struct[begin_lowres:end_lowres, 2][~struct_nan],
            kind="linear")(chrom_indices_highres)

        # Fill in beads at start
        diff_beads_at_chr_start = struct_highres[chrom_indices_highres[
            1], :] - struct_highres[chrom_indices_highres[0], :]
        how_far = 1
        for j in reversed(unknown_beads_at_begin):
            struct_highres[j, :] = struct_highres[chrom_indices_highres[
                0], :] - diff_beads_at_chr_start * how_far
            how_far += 1
        # Fill in beads at end
        diff_beads_at_chr_end = struct_highres[
            chrom_indices_highres[-2], :] - struct_highres[chrom_indices_highres[-1], :]
        how_far = 1
        for j in unknown_beads_at_end:
            struct_highres[j, :] = struct_highres[
                chrom_indices_highres[-1], :] - diff_beads_at_chr_end * how_far
            how_far += 1

        begin_lowres = end_lowres

    return struct_highres


def _convert_indices_to_full_res(rows, cols, rows_max, cols_max,
                                 multiscale_factor, lengths, n, counts_shape,
                                 ploidy):
    """Return full-res counts indices grouped by the corresponding low-res bin.
    """

    if multiscale_factor == 1:
        return rows, cols

    # Convert low-res indices to full-res
    nnz = len(rows)
    x, y = np.indices((multiscale_factor, multiscale_factor))
    rows = np.repeat(x.flatten(), nnz)[:nnz * multiscale_factor ** 2] + \
        np.tile(rows * multiscale_factor, multiscale_factor ** 2)
    cols = np.repeat(y.flatten(), nnz)[:nnz * multiscale_factor ** 2] + \
        np.tile(cols * multiscale_factor, multiscale_factor ** 2)
    rows = rows.reshape(multiscale_factor ** 2, -1)
    cols = cols.reshape(multiscale_factor ** 2, -1)
    # Figure out which rows / cols are out of bounds
    bins_for_rows = np.tile(lengths, int(counts_shape[0] / n)).cumsum()
    bins_for_cols = np.tile(lengths, int(counts_shape[1] / n)).cumsum()
    for i in range(lengths.shape[0] * ploidy):
        rows_binned = np.digitize(rows, bins_for_rows)
        cols_binned = np.digitize(cols, bins_for_cols)
        incorrect_rows = np.invert(
            np.equal(rows_binned, np.floor(rows_binned.mean(axis=0))))
        incorrect_cols = np.invert(
            np.equal(cols_binned, np.floor(cols_binned.mean(axis=0))))
        row_mask = np.floor(rows_binned.mean(axis=0)) == i
        col_mask = np.floor(cols_binned.mean(axis=0)) == i
        row_vals = np.unique(rows[:, row_mask][incorrect_rows[:, row_mask]])
        col_vals = np.unique(cols[:, col_mask][incorrect_cols[:, col_mask]])
        for val in np.flip(row_vals, axis=0):
            rows[rows > val] -= 1
        for val in np.flip(col_vals, axis=0):
            cols[cols > val] -= 1
        # Because if the last low-res bin in this homolog of this chromosome is
        # all zero, that could mess up indices for subsequent
        # homologs/chromosomes
        rows_binned = np.digitize(rows, bins_for_rows)
        row_mask = np.floor(rows_binned.mean(axis=0)) == i
        current_rows = rows[:, row_mask][
            np.invert(incorrect_rows)[:, row_mask]]
        if current_rows.shape[0] > 0 and i < bins_for_rows.shape[0]:
            max_row = current_rows.max()
            if max_row < bins_for_rows[i] - 1:
                rows[rows > max_row] -= multiscale_factor - \
                    (bins_for_rows[i] - max_row - 1)
        cols_binned = np.digitize(cols, bins_for_cols)
        col_mask = np.floor(cols_binned.mean(axis=0)) == i
        current_cols = cols[:, col_mask][
            np.invert(incorrect_cols)[:, col_mask]]
        if current_cols.shape[0] > 0 and i < bins_for_cols.shape[0]:
            max_col = current_cols.max()
            if max_col < bins_for_cols[i] - 1:
                cols[cols > max_col] -= multiscale_factor - \
                    (bins_for_cols[i] - max_col - 1)

    incorrect_indices = incorrect_rows + incorrect_cols + \
        (rows >= rows_max) + (cols >= cols_max)
    rows[incorrect_indices] = 0
    cols[incorrect_indices] = 0
    rows = rows.flatten()
    cols = cols.flatten()
    return rows, cols


def _process_multiscale_counts(counts, multiscale_factor, lengths, ploidy):
    """TODO
    """

    from .counts import _row_and_col, _check_counts_matrix

    if multiscale_factor == 1:
        rows, cols = _row_and_col(counts)
        return counts, rows, cols

    input_is_sparse = sparse.issparse(counts)

    counts = _check_counts_matrix(
        counts, lengths=lengths, ploidy=ploidy, exclude_zeros=True).toarray()

    lengths_lowres = decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor)

    dummy_counts_lowres = np.ones(
        np.array(counts.shape / lengths.sum() * lengths_lowres.sum()).astype(int))
    dummy_counts_lowres = _check_counts_matrix(
        dummy_counts_lowres, lengths=lengths_lowres, ploidy=ploidy,
        exclude_zeros=True).toarray().astype(int)
    dummy_counts_lowres = sparse.coo_matrix(dummy_counts_lowres)

    rows_lowres, cols_lowres = _row_and_col(dummy_counts_lowres)

    rows_fullres, cols_fullres = _convert_indices_to_full_res(
        rows_lowres, cols_lowres, rows_max=counts.shape[0],
        cols_max=counts.shape[1], multiscale_factor=multiscale_factor,
        lengths=lengths, n=lengths_lowres.sum(),
        counts_shape=dummy_counts_lowres.shape, ploidy=ploidy)

    data_lowres = counts[rows_fullres, cols_fullres].reshape(
        multiscale_factor ** 2, -1).sum(axis=0)
    counts_lowres = sparse.coo_matrix(
        (data_lowres[data_lowres != 0],
            (rows_lowres[data_lowres != 0], cols_lowres[data_lowres != 0])),
        shape=dummy_counts_lowres.shape)

    if not input_is_sparse:
        counts_lowres = _check_counts_matrix(
            counts_lowres, lengths=lengths_lowres, ploidy=ploidy,
            exclude_zeros=False)

    return counts_lowres, rows_fullres, cols_fullres


def _group_counts_multiscale(counts, lengths, ploidy, multiscale_factor=1,
                             multiscale_reform=False, dummy=False):
    """TODO
    """

    from .counts import _counts_indices_to_3d_indices

    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)

    if isinstance(counts, np.ndarray) or sparse.issparse(counts):
        counts_coo = sparse.coo_matrix(counts)
    else:
        counts_coo = counts.tocoo()

    if multiscale_reform and multiscale_factor != 1:
        counts_lowres, rows_grp, cols_grp = _process_multiscale_counts(
            counts_coo, multiscale_factor=multiscale_factor,
            lengths=lengths, ploidy=ploidy)
        indices = counts_lowres.row, counts_lowres.col
        indices3d = _counts_indices_to_3d_indices(
            counts_lowres, n=lengths_lowres.sum(), ploidy=ploidy)
        data_grouped = counts_coo.toarray()[rows_grp, cols_grp].reshape(
            multiscale_factor ** 2, -1)
        data_grouped = data_grouped[:, data_grouped.sum(axis=0) != 0]
        if dummy:
            data_grouped = np.zeros_like(data_grouped)
        nnz_lowres = counts_lowres.nnz
    else:
        indices = counts_coo.row, counts_coo.col
        indices3d = _counts_indices_to_3d_indices(
            counts_coo, n=lengths_lowres.sum(), ploidy=ploidy)
        data_grouped = None
        nnz_lowres = counts_coo.nnz

    return data_grouped, indices, indices3d, nnz_lowres


def decrease_counts_res(counts, multiscale_factor, lengths, ploidy):
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
    counts_lowres : list of array or coo_matrix
        Counts data at reduced resolution, as specified by the given
        `multiscale_factor`.
    lengths_lowres : array of int
        Number of beads per homolog of each chromosome at the given
        `multiscale_factor`.
    """

    from .counts import _row_and_col, _check_counts_matrix

    if multiscale_factor == 1:
        return counts

    input_is_sparse = sparse.issparse(counts)

    counts = _check_counts_matrix(
        counts, lengths=lengths, ploidy=ploidy, exclude_zeros=True).toarray()

    lengths_lowres = decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor)

    dummy_counts_lowres = np.ones(
        np.array(counts.shape / lengths.sum() * lengths_lowres.sum()).astype(int))
    dummy_counts_lowres = _check_counts_matrix(
        dummy_counts_lowres, lengths=lengths_lowres, ploidy=ploidy,
        exclude_zeros=True).toarray().astype(int)
    dummy_counts_lowres = sparse.coo_matrix(dummy_counts_lowres)

    rows_lowres, cols_lowres = _row_and_col(dummy_counts_lowres)

    rows_fullres, cols_fullres = _convert_indices_to_full_res(
        rows_lowres, cols_lowres, rows_max=counts.shape[0],
        cols_max=counts.shape[1], multiscale_factor=multiscale_factor,
        lengths=lengths, n=lengths_lowres.sum(),
        counts_shape=dummy_counts_lowres.shape, ploidy=ploidy)

    data_lowres = counts[rows_fullres, cols_fullres].reshape(
        multiscale_factor ** 2, -1).sum(axis=0)
    counts_lowres = sparse.coo_matrix(
        (data_lowres[data_lowres != 0],
            (rows_lowres[data_lowres != 0], cols_lowres[data_lowres != 0])),
        shape=dummy_counts_lowres.shape)

    if not input_is_sparse:
        counts_lowres = _check_counts_matrix(
            counts_lowres, lengths=lengths_lowres, ploidy=ploidy,
            exclude_zeros=False)

    return counts_lowres


def _get_struct_indices(ploidy, multiscale_factor, lengths):
    """Return full-res struct indices grouped by the corresponding low-res bead.
    """

    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)

    indices = np.arange(lengths_lowres.sum() * ploidy).astype(float)
    indices = np.repeat(np.indices([multiscale_factor]), indices.shape[
                        0]) + np.tile(indices * multiscale_factor, multiscale_factor)
    indices = indices.reshape(multiscale_factor, -1)

    # Figure out which rows / cols are out of bounds
    bins = np.tile(lengths, ploidy).cumsum()
    for i in range(lengths.shape[0] * ploidy):
        indices_binned = np.digitize(indices, bins)
        incorrect_indices = np.invert(
            np.equal(indices_binned, indices_binned.min(axis=0)))
        index_mask = indices_binned.min(axis=0) == i
        vals = np.unique(
            indices[:, index_mask][incorrect_indices[:, index_mask]])
        for val in np.flip(vals, axis=0):
            indices[indices > val] -= 1
    incorrect_indices += indices >= lengths.sum() * ploidy

    incorrect_indices = incorrect_indices.flatten()
    indices = indices.flatten()

    # If a bin spills over chromosome / homolog boundaries, set it to NaN - it
    # will get ignored later
    indices[incorrect_indices] = np.nan

    return indices


def _group_highres_struct(struct, multiscale_factor, lengths, indices=None, mask=None):
    """Group beads of full-res struct by the low-res bead they correspond to.

    Axes of final array:
        0: all highres beads corresponding to each lowres bead, size = multiscale factor
        1: beads, size = struct[0]
        2: coordinates, size = struct[1] = 3
    """

    lengths = np.array(lengths).astype(int)

    ploidy = struct.reshape(-1, 3).shape[0] / lengths.sum()
    if ploidy != 1 and ploidy != 2:
        raise ValueError("Not consistent with haploid or diploid... struct is"
                         " %d beads (and 3 cols), sum of lengths is %d" % (
                             struct.reshape(-1, 3).shape[0], lengths.sum()))
    ploidy = int(ploidy)

    if indices is None:
        indices = _get_struct_indices(
            ploidy=ploidy, multiscale_factor=multiscale_factor, lengths=lengths)
    else:
        indices = indices.copy()
    incorrect_indices = np.isnan(indices)

    # If a bin spills over chromosome / homolog boundaries, set it to whatever
    # - it will get ignored later
    indices[incorrect_indices] = 0
    indices = indices.astype(int)

    # Apply mask
    if mask is not None and mask != [None]:
        indices[~mask] = 0
        incorrect_indices = (incorrect_indices + np.invert(
            mask)).astype(bool).astype(int)

    # Apply to struct, and set incorrect indices to np.nan
    return np.where(np.repeat(incorrect_indices.reshape(-1, 1), 3, axis=1), np.nan,
                    struct.reshape(-1, 3)[indices, :]).reshape(multiscale_factor, -1, 3)


def decrease_struct_res(struct, multiscale_factor, lengths, indices=None, mask=None):
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

    if int(multiscale_factor) != multiscale_factor:
        raise ValueError(
            "The multiscale_factor to reduce size by must be an integer.")
    if multiscale_factor == 1:
        return struct

    grouped_struct = _group_highres_struct(
        struct, multiscale_factor, lengths, indices=indices, mask=mask)

    return np.nanmean(grouped_struct, axis=0)


def _count_fullres_per_lowres_bead(multiscale_factor, lengths, ploidy,
                                   fullres_torm=None):
    """Count the number of full-res beads corresponding to each low-res bead.
    """

    if multiscale_factor == 1:
        return None

    fullres_indices = _get_struct_indices(
        ploidy=ploidy, multiscale_factor=multiscale_factor,
        lengths=lengths).reshape(multiscale_factor, -1)

    if fullres_torm is not None and fullres_torm.sum() != 0:
        fullres_indices[fullres_indices == np.where(fullres_torm)[0]] = np.nan

    return (~ np.isnan(fullres_indices)).sum(axis=0)


def get_multiscale_variances_from_struct(structures, lengths, multiscale_factor,
                                         mixture_coefs=None, verbose=True,
                                         replace_nan=True):
    """Compute multiscale variances from full-res structure.

    Generates multiscale variances at the specified resolution from the
    inputted full-resolution structure(s). Multiscale variances are defined as
    follows: for each low-resolution bead, the variances of the distances
    between all high-resolution beads that correspond to that low-resolution
    bead.

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
            struct, multiscale_factor, lengths)
        multiscale_variances.append(
            _var3d(struct_grouped, replace_nan=replace_nan))
    multiscale_variances = np.mean(multiscale_variances, axis=0)

    if verbose:
        print("MULTISCALE VARIANCE: %.3g" % np.median(multiscale_variances),
              flush=True)

    return multiscale_variances


def _var3d(struct_grouped, replace_nan=True):
    """Compute variance of beads in 3D.
    """

    # struct_grouped.shape = (multiscale_factor, nbeads, 3)
    multiscale_variances = np.full(struct_grouped.shape[1], np.nan)
    for i in range(struct_grouped.shape[1]):
        struct_group = struct_grouped[:, i, :]
        beads_in_group = np.invert(np.isnan(struct_group[:, 0])).sum()
        if beads_in_group == 0:
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


def est_multiscale_epsilon_from_counts(counts, lengths, ploidy,
                                       multiscale_factor, alpha, bias,
                                       mixture_coefs=None, verbose=True):
    """Compute multiscale epsilon from full-res structure.
    """

    if multiscale_factor == 1:
        return None

    lengths_lowres = decrease_lengths_res(
        lengths=lengths, multiscale_factor=multiscale_factor)

    multiscale_stddev = []
    for counts_maps in counts:
        if counts_maps.type == 'zero':
            continue
        bias_per_bin = counts_maps.bias_per_bin(bias, ploidy)
        counts_norm_data = counts_maps.data / bias_per_bin / counts_maps.beta
        counts_norm = sparse.coo_matrix(
            (counts_norm_data, (counts_maps.row, counts_maps.col)),
            shape=counts_maps.shape)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning,
                message="Counts matrix must only contain integers or NaN")
            counts_grouped = _group_counts_multiscale(
                counts_norm, lengths=lengths, ploidy=ploidy,
                multiscale_factor=multiscale_factor, multiscale_reform=True)[0]
        # counts_grouped.shape = (multiscale_factor ** 2, nbins)
        counts_grouped = counts_grouped[:, counts_grouped.min(axis=0) != 0]
        sort_index = counts_grouped.sum(axis=0).argsort()
        counts_grouped = counts_grouped[:, sort_index]
        if counts_maps.ambiguity == 'ambig':
            top = lengths_lowres.sum()
        else:
            top = lengths_lowres.sum() * ploidy
        counts_grouped_top = counts_grouped[:, -top:]
        faux_dis_grouped = counts_grouped_top ** (1 / alpha)
        multiscale_stddev.append(_stddev_distances(faux_dis_grouped))
    multiscale_stddev = np.mean(multiscale_stddev)

    if verbose:
        print("MULTISCALE STD-DEV: %.3g" % multiscale_stddev,
              flush=True)

    return multiscale_stddev


def get_multiscale_epsilon_from_struct(structures, lengths, multiscale_factor,
                                       mixture_coefs=None, verbose=True):
    """Compute multiscale epsilon from full-res structure.
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

    multiscale_stddev = []
    for struct in structures:
        mask = np.invert(np.isnan(structures[0][:, 0]))
        dummy = sparse.coo_matrix(np.triu(
            np.ones((mask.sum(), mask.sum())), 1))
        dis = np.sqrt((np.square(
            struct[mask][dummy.row] - struct[mask][dummy.col])).sum(axis=1))
        dis_coo = sparse.coo_matrix(
            (dis, (dummy.row, dummy.col)), shape=dummy.shape)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning,
                message="Counts matrix must only contain integers or NaN")
            dis_grouped = _group_counts_multiscale(
                dis_coo, lengths=lengths, ploidy=ploidy,
                multiscale_factor=multiscale_factor, multiscale_reform=True)[0]
        multiscale_stddev.append(_stddev_distances(dis_grouped))
    multiscale_stddev = np.mean(multiscale_stddev)

    if verbose:
        print("MULTISCALE STD-DEV: %.3g" % multiscale_stddev,
              flush=True)

    return multiscale_stddev


def _stddev_distances(dis_grouped):
    """TODO
    """

    from scipy.stats import norm

    # dis_grouped.shape = (multiscale_factor ** 2, nbins)
    dis_corr = []
    for i in range(dis_grouped.shape[1]):
        dis_group = dis_grouped[:, i]
        dis_corr.append(dis_group - dis_group.mean())

    dis_corr = np.concatenate(dis_corr)
    mu, stddev = norm.fit(dis_corr)

    if np.isnan(stddev):
        raise ValueError("Multiscale dist stddev is nan.")

    return stddev


def _choose_max_multiscale_rounds(lengths, min_beads):
    """Choose the maximum number of multiscale rounds, given `min_beads`.
    """

    multiscale_rounds = 0
    while decrease_lengths_res(
            lengths, 2 ** (multiscale_rounds + 1)).min() >= min_beads:
        multiscale_rounds += 1
    return multiscale_rounds


def _choose_max_multiscale_factor(lengths, min_beads):
    """Choose the maximum multiscale factor, given `min_beads`.
    """

    return 2 ** _choose_max_multiscale_rounds(
        lengths=lengths, min_beads=min_beads)
