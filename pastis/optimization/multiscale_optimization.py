import sys
import numpy as np
import warnings
from scipy import sparse
from scipy.interpolate import interp1d
from iced.io import load_lengths

from absl import logging as absl_logging
absl_logging.set_verbosity('error')
from jax.config import config as jax_config
jax_config.update("jax_platform_name", "cpu")
jax_config.update("jax_enable_x64", True)
import jax.numpy as ag_np

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
        return np.asarray(lengths)
    else:
        return np.ceil(
            np.asarray(lengths, dtype=float) / multiscale_factor).astype(int)


def increase_struct_res_gaussian(struct, current_multiscale_factor,
                                 final_multiscale_factor, lengths, std_dev,
                                 random_state=None):
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

    lengths = np.array(lengths).astype(int)
    lengths_current = decrease_lengths_res(
        lengths=lengths, multiscale_factor=current_multiscale_factor)

    ploidy = struct.shape[0] / lengths_current.sum()
    if ploidy != 1 and ploidy != 2:
        raise ValueError("Not consistent with haploid or diploid... struct has"
                         f" {struct.reshape(-1, 3).shape[0]} beads (and 3 cols)"
                         f", sum of lengths is {lengths_current.sum()}")
    ploidy = int(ploidy)

    if random_state is None:
        random_state = np.random.RandomState(0)

    # Estimate full-resolution structure
    grouped_idx, bad_idx = _get_struct_index(
        multiscale_factor=current_multiscale_factor, lengths=lengths,
        ploidy=ploidy)
    grouped_idx = grouped_idx.astype(float)
    grouped_idx[bad_idx] = np.nan

    struct_fullres = []
    for i in range(struct.shape[0]):
        lowres_bead = struct[i]
        if np.isnan(lowres_bead[0]):  # Linearly interpolate lowres bead if NaN
            lowres_bead = np.nanmean(struct[(i - 1):(i + 2)], axis=0)
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


def increase_struct_res(struct, multiscale_factor, lengths, mask=None):
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

    idx, bad_idx = _get_struct_index(
        multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)
    # idx = idx.reshape(multiscale_factor, -1).astype(float)  # TODO remove
    # idx[bad_idx.reshape(multiscale_factor, -1)] = np.nan
    idx = idx.astype(float)
    idx[bad_idx] = np.nan

    if mask is not None:
        idx[~mask.reshape(multiscale_factor, -1)] = np.nan

    struct_highres = np.full((lengths.sum() * ploidy, 3), np.nan)
    begin_lowres = end_lowres = 0
    for i in range(lengths.shape[0] * ploidy):
        end_lowres += np.tile(lengths_lowres, ploidy)[i]

        # Beads of struct that are NaN
        struct_nan_mask = np.isnan(struct[begin_lowres:end_lowres, 0])

        # Get index for this chrom at low & high res
        chrom_indices = idx[:, begin_lowres:end_lowres]
        chrom_indices[:, struct_nan_mask] = np.nan
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
            chrom_indices_lowres[~struct_nan_mask],
            struct[begin_lowres:end_lowres, 0][~struct_nan_mask],
            kind="linear")(chrom_indices_highres)
        struct_highres[chrom_indices_highres, 1] = interp1d(
            chrom_indices_lowres[~struct_nan_mask],
            struct[begin_lowres:end_lowres, 1][~struct_nan_mask],
            kind="linear")(chrom_indices_highres)
        struct_highres[chrom_indices_highres, 2] = interp1d(
            chrom_indices_lowres[~struct_nan_mask],
            struct[begin_lowres:end_lowres, 2][~struct_nan_mask],
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


def _convert_indices_to_full_res(rows_lowres, cols_lowres, rows_max, cols_max,
                                 multiscale_factor, lengths, n, counts_shape,
                                 ploidy):
    """Return full-res counts indices grouped by the corresponding low-res bin.
    """

    if multiscale_factor == 1:
        return rows_lowres, cols_lowres

    lengths_lowres = decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor)
    n = lengths_lowres.sum()  # TODO remove n from args

    # Convert low-res indices to full-res
    nnz_lowres = len(rows_lowres)
    x, y = np.indices((multiscale_factor, multiscale_factor))
    rows = np.repeat(x.flatten(), nnz_lowres)[
        :nnz_lowres * multiscale_factor ** 2] + \
        np.tile(rows_lowres * multiscale_factor, multiscale_factor ** 2)
    cols = np.repeat(y.flatten(), nnz_lowres)[
        :nnz_lowres * multiscale_factor ** 2] + \
        np.tile(cols_lowres * multiscale_factor, multiscale_factor ** 2)
    rows = rows.reshape(multiscale_factor ** 2, -1)
    cols = cols.reshape(multiscale_factor ** 2, -1)
    # Figure out which rows / cols are out of bounds
    bins_for_rows = np.tile(lengths, int(counts_shape[0] / n)).cumsum()
    bins_for_cols = np.tile(lengths, int(counts_shape[1] / n)).cumsum()
    for i in range(lengths.shape[0] * ploidy):
        rows_binned = np.digitize(rows, bins_for_rows)
        cols_binned = np.digitize(cols, bins_for_cols)
        rows_good_bin = np.floor(rows_binned.mean(axis=0))
        cols_good_bin = np.floor(cols_binned.mean(axis=0))
        incorrect_rows = np.invert(np.equal(rows_binned, rows_good_bin))
        incorrect_cols = np.invert(np.equal(cols_binned, cols_good_bin))
        row_mask = rows_good_bin == i
        col_mask = cols_good_bin == i
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
        rows_good_bin = np.floor(rows_binned.mean(axis=0))
        row_mask = rows_good_bin == i
        current_rows = rows[:, row_mask][np.invert(incorrect_rows)[:, row_mask]]
        if current_rows.shape[0] > 0 and i < bins_for_rows.shape[0]:
            max_row = current_rows.max()
            if max_row < bins_for_rows[i] - 1:
                rows[rows > max_row] -= multiscale_factor - \
                    (bins_for_rows[i] - max_row - 1)
        cols_binned = np.digitize(cols, bins_for_cols)
        cols_good_bin = np.floor(cols_binned.mean(axis=0))
        col_mask = cols_good_bin == i
        current_cols = cols[:, col_mask][np.invert(incorrect_cols)[:, col_mask]]
        if current_cols.shape[0] > 0 and i < bins_for_cols.shape[0]:
            max_col = current_cols.max()
            if max_col < bins_for_cols[i] - 1:
                cols[cols > max_col] -= multiscale_factor - \
                    (bins_for_cols[i] - max_col - 1)

    bad_idx = incorrect_rows + incorrect_cols + \
        (rows >= rows_max) + (cols >= cols_max)
    rows[bad_idx] = 0
    cols[bad_idx] = 0
    rows = rows.flatten()
    cols = cols.flatten()
    bad_idx = bad_idx.flatten()
    return rows, cols


def _group_counts_multiscale(counts, lengths, ploidy, multiscale_factor=1,
                             dummy=False, exclude_each_fullres_zero=False,
                             include_all_nan_groups=False):
    """Group together full-res counts corresponding to a given low-res distance.

    Prepare counts for multi-resolution optimization by aggregating sets of
    full-res counts bins, such that each set corresponds to a single low-res
    distance bin.

    Parameters
    ----------
    counts : array or coo_matrix
        Counts data at full resolution, ideally without normalization or
        filtering.
    lengths : array_like of int
        Number of beads per homolog of each chromosome at full resolution.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 does not change the resolution.
    dummy : bool, optional
        Return zeros in the shape of the grouped counts array.
    exclude_each_fullres_zero : bool, optional
        If true: if any of the individual full-res bins in a group are zero,
        exclude the entire group of full-res bins from the output.
        Individual full-res bins that are NaN do not result in the group's
        exclusion. Groups where every single full-res bin is NaN are still
        excluded from the ouptut by default, unless include_all_nan_groups=True
    include_all_nan_groups : bool, optional
        Whether groups where every single full-res bin is NaN are excluded from
        the output.

    Returns
    -------
    data_grouped : array
        TODO
    indices : tuple of arrays
        TODO
    indices3d : tuple of arrays
        TODO
    shape_lowres : tuple of int
        TODO
    mask : array
        TODO
    """

    # data_grouped, indices, indices3d, shape_lowres, mask

    revert_bug_fix = False  # TODO remove junk
    unmask_zeros_in_sparse = False  # TODO remove junk
    ignore_exclude_each_fullres_zero = False  # TODO remove junk

    if ignore_exclude_each_fullres_zero:
        exclude_each_fullres_zero = False
    if unmask_zeros_in_sparse and exclude_each_fullres_zero:
        data_grouped = indices = indices3d = mask = np.array([])
        shape_lowres = (0,)
        raise NotImplementedError("what should shape_lowres be here?")  # nnz_lowres = 0
        return data_grouped, indices, indices3d, shape_lowres, mask

    from .counts import _counts_indices_to_3d_indices, _check_counts_matrix
    from .counts import _row_and_col

    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)

    if isinstance(counts, np.ndarray):
        counts_coo = sparse.coo_matrix(counts)
    else:
        counts_coo = counts.tocoo()

    if multiscale_factor > 1:
        counts_arr = _check_counts_matrix(
            counts_coo, lengths=lengths, ploidy=ploidy, exclude_zeros=False)

        counts_lowres, rows_grp, cols_grp = decrease_counts_res(
            counts_coo, multiscale_factor=multiscale_factor,
            lengths=lengths, ploidy=ploidy, return_indices=True)
        row_lowres = counts_lowres.row
        col_lowres = counts_lowres.col
        shape_lowres = counts_lowres.shape

        if unmask_zeros_in_sparse:
            counts_lowres, rows_grp, cols_grp = decrease_counts_res(
                counts_arr, multiscale_factor=multiscale_factor,
                lengths=lengths, ploidy=ploidy, return_indices=True)
            row_lowres, col_lowres = _row_and_col(counts_lowres)
            shape_lowres = counts_lowres.shape
            raise NotImplementedError("what should shape_lowres be here?")  # FIXME nnz_lowres = len(row_lowres)

        data_grouped = counts_arr[rows_grp, cols_grp].reshape(
            multiscale_factor ** 2, -1)
        if not unmask_zeros_in_sparse:
            data_grouped = data_grouped[:, np.nansum(data_grouped, axis=0) != 0]

        if revert_bug_fix or unmask_zeros_in_sparse:
            exclude_each_fullres_zero = False
            data_grouped[np.isnan(data_grouped)] = 0

        if exclude_each_fullres_zero:
            #data_grouped[np.isnan(data_grouped)] = 0  # TODO remove junk
            #min_gt0 = np.min(data_grouped, axis=0) != 0
            #min_gt0 = data_grouped.min(axis=0) != 0

            # latest bug fix 3/26:
            ####### min_gt0 = data_grouped.min(axis=0) != 0
            ####### min_gt0 = (data_grouped.min(axis=0) != 0) & (~np.isnan(data_grouped.min(axis=0)))

            # If any of the full-res bins are zero, remove the entire group.
            # Full-res NaN bins are ok and don't warrant the group's removal.
            # However, we previously excluded groups where every single full-res
            # bin was NaN.
            min_gt0 = np.nanmin(data_grouped, axis=0) != 0
            data_grouped = data_grouped[:, min_gt0]
            counts_lowres = sparse.coo_matrix((
                counts_lowres.data[min_gt0],
                (row_lowres[min_gt0], col_lowres[min_gt0])),
                shape=counts_lowres.shape)
            row_lowres = counts_lowres.row
            col_lowres = counts_lowres.col
            shape_lowres = counts_lowres.shape

        mask = ~np.isnan(data_grouped)
        data_grouped[~mask] = 0

        indices = row_lowres, col_lowres
        indices3d = _counts_indices_to_3d_indices(
            counts_lowres, nbeads_lowres=lengths_lowres.sum() * ploidy)

        if dummy:
            data_grouped = np.zeros_like(data_grouped)
        # shape_lowres = counts_lowres.shape
    else:
        indices = counts_coo.row, counts_coo.col
        indices3d = _counts_indices_to_3d_indices(
            counts_coo, nbeads_lowres=lengths_lowres.sum() * ploidy)
        data_grouped = counts_coo.data
        mask = None
        shape_lowres = counts_coo.shape

    return data_grouped, indices, indices3d, shape_lowres, mask


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
    counts_lowres : list of array or coo_matrix
        Counts data at reduced resolution, as specified by the given
        `multiscale_factor`.
    """

    # TODO refactor this fxn & _convert_indices_to_full_res to be similar to new _get_struct_index

    from .counts import _row_and_col, _check_counts_matrix

    if multiscale_factor == 1:
        if return_indices:
            rows, cols = _row_and_col(counts)
            return counts, rows, cols
        else:
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
        exclude_zeros=True, remove_diag=remove_diag).toarray().astype(int)
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
            exclude_zeros=False, remove_diag=remove_diag)

    if return_indices:
        return counts_lowres, rows_fullres, cols_fullres
    else:
        return counts_lowres


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
                          idx=None, fullres_struct_nan=None):
    """Group beads of full-res struct by the low-res bead they correspond to.

    Axes of final array:
        0: all highres beads corresponding to each lowres bead, size = multiscale factor
        1: beads, size = struct[0]
        2: coordinates, size = struct[1] = 3
    """

    if multiscale_factor == 1:
        return struct.reshape(1, -1, 3)

    if idx is None:
        idx, bad_idx = _get_struct_index(
            multiscale_factor=multiscale_factor, lengths=lengths, ploidy=ploidy)
    else:
        raise NotImplementedError  # TODO

    if fullres_struct_nan is not None and fullres_struct_nan.shape != 0:
        struct_nan_mask = np.isin(idx, fullres_struct_nan)
        bad_idx[struct_nan_mask] = True
        idx[struct_nan_mask] = 0

    # Apply to struct, and set incorrect idx to np.nan
    grouped_struct = ag_np.where(
        np.repeat(bad_idx.reshape(-1, 1), 3, axis=1), np.nan,
        struct.reshape(-1, 3)[idx.flatten(), :]).reshape(
        multiscale_factor, -1, 3)

    return grouped_struct


def decrease_struct_res(struct, multiscale_factor, lengths, ploidy,
                        idx=None, fullres_struct_nan=None):
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
        ploidy=ploidy, idx=idx, fullres_struct_nan=fullres_struct_nan)

    return ag_np.nanmean(grouped_struct, axis=0)


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


def get_multiscale_variances_from_struct(structures, lengths, multiscale_factor,
                                         mixture_coefs=None, replace_nan=True,
                                         verbose=True):
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
            struct, multiscale_factor=multiscale_factor, lengths=lengths,
            ploidy=ploidy)
        multiscale_variances.append(
            _var3d(struct_grouped, replace_nan=replace_nan))
    multiscale_variances = np.mean(multiscale_variances, axis=0)

    if verbose:
        print(f"MULTISCALE VARIANCE ({multiscale_factor}x):"
              f" {np.median(multiscale_variances):.3g}", flush=True)

    return multiscale_variances


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


def get_multiscale_epsilon_from_dis(structures, lengths, multiscale_factor,
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

    std_all = []
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
                multiscale_factor=multiscale_factor)[0]
        std_all.append(_get_epsilon_from_dis(dis_grouped))
    std_all = np.mean(std_all)

    if verbose:
        print(f"MULTISCALE EPSILON (estimate) ({multiscale_factor}x):"
              f" {std_all:.3g}", flush=True)

    return std_all


def _get_epsilon_from_dis(dis_grouped, alpha=-3.):
    """TODO
    """

    from scipy.stats import norm, mode
    #import plotille  # FIXME
    #from topsy.datasets.samples_generator import get_diff_coords_from_euc_dis

    # dis_grouped.shape = (multiscale_factor ** 2, nbins)
    tmp = []
    for i in range(dis_grouped.shape[1]):
        dis_fullres = dis_grouped[:, i]
        if (dis_fullres == 0).sum() > 0:  # Lowres dist bin is on diagonal
            print((dis_fullres == 0).sum(), (dis_fullres != 0).sum())
            continue
        fake_counts_lowres = (dis_fullres ** alpha).sum()
        dis_lowres = fake_counts_lowres ** (1 / alpha)
        tmp.append(dis_fullres.flatten() - dis_lowres)

    tmp = np.concatenate(tmp)
    est_diff = tmp #* np.sqrt(1 / 3)

    est_diff_rounded = np.array([float(f"{x:.2g}") for x in est_diff])
    est_diff_mode = mode(est_diff_rounded, axis=None)[0][0]

    #est_diff -= np.median(est_diff)
    #est_diff -= est_diff_mode
    #est_diff = np.append(est_diff, -est_diff)

    # fig = plotille.Figure()
    # fig.height = 25
    # fig.set_y_limits(min_=0)
    # fig.histogram(est_diff, bins=160, lc=None)
    # fig.plot([0, 0], [0, 2400])
    # fig.plot([np.median(est_diff), np.median(est_diff)], [0, 2400])
    # fig.plot([np.mean(est_diff), np.mean(est_diff)], [0, 2400])
    # #fig.plot([est_diff_mode, est_diff_mode], [0, 2400])
    # print(fig.show(legend=False))

    mu, stddev = norm.fit(est_diff)
    #print(f'MIN={est_diff.min():.3g}    MU={mu:.3g}   MEAN={np.mean(est_diff):.3g}   MED={np.median(est_diff):.3g}')

    if np.isnan(stddev):
        raise ValueError("Multiscale dist stddev is nan.")

    return stddev


def get_multiscale_epsilon_from_struct(structures, lengths, multiscale_factor,
                                       mixture_coefs=None, replace_nan=True,
                                       verbose=True):
    """Compute multiscale epsilon from full-res structure.

    Generates multiscale epsilons at the specified resolution from the
    inputted full-resolution structure(s). Multiscale epsilons are defined as
    follows: for each low-resolution bead, the variances of the distances
    between all high-resolution beads that correspond to that low-resolution
    bead. FIXME

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
    multiscale_variances = np.mean(multiscale_variances, axis=0)

    multiscale_epsilon = np.sqrt(multiscale_variances * 2 / 3)

    if verbose:
        print(f"MULTISCALE EPSILON ({multiscale_factor}x):"
              f" {np.mean(multiscale_epsilon):.3g}", flush=True)

    return multiscale_epsilon



def get_multiscale_epsilon_from_struct_old(structures, lengths, multiscale_factor,
                                           mixture_coefs=None, replace_nan=True,
                                           verbose=True):
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

    std_per_bead = []
    std_all = []
    for struct in structures:
        struct_grouped = _group_highres_struct(
            struct, multiscale_factor=multiscale_factor, lengths=lengths,
            ploidy=ploidy)
        std_per_bead_tmp, std_all_tmp = _get_epsilon(
            struct_grouped, replace_nan=replace_nan)
        std_per_bead.append(std_per_bead_tmp)
        std_all.append(std_all_tmp)
    std_per_bead = np.mean(std_per_bead, axis=0)
    std_all = np.mean(std_all)

    if verbose:
        print(f"MULTISCALE EPSILON per-bead  ({multiscale_factor}x):"
              f" {np.median(std_per_bead):.3g}", flush=True)
        print(f"MULTISCALE EPSILON all-beads ({multiscale_factor}x):"
              f" {std_all:.3g}", flush=True)

    return std_all, std_per_bead


def _get_epsilon(struct_grouped, replace_nan=True):
    """TODO
    """

    from scipy.stats import norm

    # struct_grouped.shape = (multiscale_factor, nbeads, 3)
    mu_per_bead = np.full(struct_grouped.shape[1], np.nan)
    std_per_bead = np.full(struct_grouped.shape[1], np.nan)
    all_diff = []
    for i in range(struct_grouped.shape[1]):
        struct_group = struct_grouped[:, i, :]
        beads_in_group = np.invert(np.isnan(struct_group[:, 0])).sum()
        if beads_in_group == 0:
            stddev = mu = np.nan
        else:
            mean_coords = np.nanmean(struct_group, axis=0)
            diff = struct_group - mean_coords
            diff *= 2  # FIXME IS THIS CORRECT?
            diff = diff[~np.isnan(diff[:, 0])]
            all_diff.append(diff)
            mu, stddev = norm.fit(diff)
        std_per_bead[i] = stddev
        mu_per_bead[i] = mu

    if np.isnan(std_per_bead).sum() == std_per_bead.shape[0]:
        raise ValueError("Multiscale stddev are nan for every bead.")

    all_diff = np.concatenate(all_diff)
    mu_all, std_all = norm.fit(all_diff)

    if not np.isclose(np.median(mu_per_bead), 0):
        warnings.warn(f"Multiscale mu (per-bead) is {np.median(mu_per_bead)}, expected 0.")
    if not np.isclose(mu_all, 0):
        warnings.warn(f"Multiscale mu (all-beads) is {np.median(mu_all)}, expected 0.")

    if replace_nan:
        std_per_bead[np.isnan(std_per_bead)] = np.nanmedian(
            std_per_bead)

    return std_per_bead, std_all


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
