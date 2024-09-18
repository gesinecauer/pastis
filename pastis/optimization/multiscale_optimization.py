import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import numpy as np
import warnings
from scipy import sparse
from scipy.interpolate import interp1d
from scipy import optimize

from .utils_poisson import _setup_jax
_setup_jax()
import jax.numpy as jnp
from jax import grad


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

    # Output type should be the same as type(lengths)
    if isinstance(lengths, jnp.ndarray):
        tmp_np = jnp
    else:
        tmp_np = np

    lengths = tmp_np.array(lengths, copy=False, ndmin=1, dtype=int).ravel()

    if multiscale_factor == 1:
        return lengths
    else:
        return tmp_np.ceil(lengths / multiscale_factor).astype(int)


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

    from ..io.read import _get_lengths, _get_struct

    # Setup
    if random_state is None:
        random_state = np.random.RandomState(seed=0)
    if int(multiscale_factor) != multiscale_factor:
        raise ValueError('The multiscale_factor must be an integer')
    multiscale_factor = int(multiscale_factor)
    lengths = _get_lengths(lengths)
    lengths_lowres = decrease_lengths_res(
        lengths=lengths, multiscale_factor=multiscale_factor)
    struct = _get_struct(struct, lengths=lengths_lowres, ploidy=ploidy)
    if multiscale_factor == 1:
        return struct

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

        # Create highres beads for this molecule
        if (~nan_mask_lowres).sum() == 0:  # 0 non-NaN beads in lowres mol
            chrom_idx_highres = chrom_idx_highres[
                ~np.isnan(chrom_idx_highres)].astype(int)
            random_chrom = random_state.uniform(
                low=-1, high=1, size=(chrom_idx_highres.size, 3))
            struct_highres[chrom_idx_highres] = random_chrom
        elif (~nan_mask_lowres).sum() == 1:  # Only 1 non-NaN bead in lowres mol
            chrom_idx_highres = chrom_idx_highres[
                ~np.isnan(chrom_idx_highres)].astype(int)
            lowres_bead = struct[chrom_idx_lowres]
            struct_highres[chrom_idx_highres] = random_state.normal(
                lowres_bead, 1, (chrom_idx_highres.size, 3))
        else:  # There are enough non-NaN beads in lowres mol to interpolate
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
                             exclude_zeros=False, multiscale_reform=True):
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

    if multiscale_factor > 1 and multiscale_reform:
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
        if multiscale_factor > 1:
            counts = decrease_counts_res(
                counts, multiscale_factor=multiscale_factor, lengths=lengths,
                ploidy=ploidy)
        idx_lowres = counts.row, counts.col
        data_grouped = counts.data
        shape_lowres = counts.shape
        mask = None

    return data_grouped, idx_lowres, shape_lowres, mask


def decrease_counts_res(counts, multiscale_factor, lengths, ploidy,
                        return_indices=False, remove_diag=True,
                        warn_on_float=False, mean=False):
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
        counts, lengths=lengths, ploidy=ploidy, warn_on_float=warn_on_float)

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
    row_fullres, col_fullres, bad_idx_fullres = idx_fullres
    row_lowres, col_lowres = idx_lowres

    data_lowres = counts.toarray()[row_fullres, col_fullres].reshape(
        multiscale_factor ** 2, -1).sum(axis=0)

    if mean:
        fullres_per_lowres_bin = np.invert(bad_idx_fullres).reshape(
            multiscale_factor ** 2, -1).sum(axis=0)
        data_lowres /= fullres_per_lowres_bin

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

    if fullres_struct_nan is not None and fullres_struct_nan.size != 0:
        nan_mask_lowres = np.isin(idx, fullres_struct_nan)
        bad_idx[nan_mask_lowres] = True
        idx[nan_mask_lowres] = 0

    # Apply to struct, and set incorrect idx to np.nan
    grouped_struct = np.where(
        np.repeat(bad_idx.reshape(-1, 1), 3, axis=1), np.nan,
        struct.reshape(-1, 3)[idx.ravel(), :]).reshape(
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

    fullres_per_lowres_bead = (~bad_idx).sum(axis=0)
    fullres_per_lowres_bead = np.asarray(fullres_per_lowres_bead, order='C')
    return fullres_per_lowres_bead


def decrease_bias_res(bias, multiscale_factor, lengths, bias_per_hmlg=None):
    """Decrease resoluion of Hi-C biases."""

    from .counts import check_bias_size

    if bias is None or np.all(bias == 1):
        return None

    if multiscale_factor == 1:
        bias = check_bias_size(
            bias, lengths=lengths, bias_per_hmlg=bias_per_hmlg)
        return bias

    lengths_lowres = decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor)
    if bias.size == lengths_lowres.sum():  # Bias is already low-res
        return bias

    bias = check_bias_size(bias, lengths=lengths, bias_per_hmlg=bias_per_hmlg)

    # Bias is the same for both homologs - it is of size lengths.sum()
    idx, bad_idx = _get_struct_index(
        multiscale_factor=multiscale_factor, lengths=lengths, ploidy=1)

    fullres_struct_nan = (bias == 0)
    if fullres_struct_nan.sum() != 0:
        where_nan = np.where(fullres_struct_nan)[0]
        is_nan = np.isin(idx, where_nan)
        bad_idx[is_nan] = True
        idx[is_nan] = 0

    # Apply to bias, and set incorrect idx to np.nan
    grouped_bias = np.where(
        bad_idx.ravel(), np.nan, bias[idx.ravel()]).reshape(
        multiscale_factor, -1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message="Mean of empty slice")
        bias_lowres = np.nanmean(grouped_bias, axis=0)

    return bias_lowres


def _var3d(struct_grouped, replace_nan=False):
    """Compute variance of beads in 3D.
    """

    # Output type should be the same as type(bias)
    if isinstance(struct_grouped, jnp.ndarray):
        tmp_np = jnp
    else:
        tmp_np = np
    invalid = -1

    # FYI, struct_grouped.shape = (multiscale_factor, nbeads_lowres, 3)
    multiscale_variances = tmp_np.full(
        struct_grouped.shape[1], invalid, dtype=float)
    for i in range(struct_grouped.shape[1]):
        struct_group = struct_grouped[:, i, :]
        beads_in_group = tmp_np.invert(tmp_np.isnan(struct_group[:, 0])).sum()
        if beads_in_group < 1:
            var = invalid
        else:
            mean_coords = tmp_np.nanmean(struct_group, axis=0)
            # Euclidian distance formula = ((A - B) ** 2).sum(axis=1) ** 0.5
            var = (1 / beads_in_group) * \
                tmp_np.nansum((struct_group - mean_coords) ** 2)
        if isinstance(struct_grouped, jnp.ndarray):
            multiscale_variances = multiscale_variances.at[i].set(var)
        else:
            multiscale_variances[i] = var

    if tmp_np.isnan(multiscale_variances).sum() == multiscale_variances.shape[0]:
        raise ValueError("Multiscale variances are invalid for every bead.")

    if replace_nan:
        multiscale_variances[
            multiscale_variances == invalid] = np.mean(multiscale_variances[
            multiscale_variances != invalid])
    elif isinstance(struct_grouped, jnp.ndarray):
        multiscale_variances = multiscale_variances[
            multiscale_variances != invalid]
    else:
        multiscale_variances[multiscale_variances == invalid] = np.nan

    return multiscale_variances


def get_epsilon_from_struct(structures, lengths, ploidy, multiscale_factor,
                            mixture_coefs=None, replace_nan=False, verbose=True):
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
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 does not change the resolution.

    Returns  TODO update
    -------
    epsilon_per_bead : array of float
        Multiscale epsilons: for each low-resolution bead, the epsilon of the
        distances between all high-resolution beads that correspond to that
        low-resolution bead.
    """

    from .utils_poisson import _format_structures

    if multiscale_factor == 1:
        return None

    lengths = np.array(lengths, copy=False, ndmin=1, dtype=int).ravel()
    structures = _format_structures(
        structures, lengths=lengths, ploidy=ploidy, mixture_coefs=mixture_coefs)

    multiscale_variances = []
    for struct in structures:
        struct_grouped = _group_highres_struct(
            struct, multiscale_factor=multiscale_factor, lengths=lengths,
            ploidy=ploidy)
        multiscale_variances.append(
            _var3d(struct_grouped, replace_nan=replace_nan))

    epsilon_per_bead = [np.sqrt(x * 2 / 3) for x in multiscale_variances]

    if len(epsilon_per_bead) == 1:
        epsilon_per_bead = epsilon_per_bead[0]
    else:  # If multiple structures are inputted, take mean for each lowres bead
        lowres_nan = np.isnan(epsilon_per_bead[0])
        for i in range(len(epsilon_per_bead)):
            lowres_nan = lowres_nan & np.isnan(epsilon_per_bead[0])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            epsilon_per_bead = np.nanmean(epsilon_per_bead, axis=0)
        epsilon_per_bead[lowres_nan] = np.nan

    # Get epsilon_per_dis from epsilon_per_bead
    # Formula: \epsilon_{xy} = \sqrt{ 0.5 ( \epsilon_x^2 \epsilon_y^2 ) }
    nbeads = epsilon_per_bead.size
    mask = ~np.isnan(epsilon_per_bead)
    per_bead_sq_div2 = np.full(nbeads, np.nan)
    per_bead_sq_div2[mask] = np.square(epsilon_per_bead[mask]) / 2
    epsilon_per_dis = per_bead_sq_div2.reshape(
        1, -1) + per_bead_sq_div2.reshape(-1, 1)
    epsilon_per_dis[~np.isnan(epsilon_per_dis)] **= 0.5

    if np.isnan(epsilon_per_bead).sum() > 0:  # TODO remove
        assert np.array_equal(mask, ~np.isnan(epsilon_per_dis[:, 0]))
        assert np.array_equal(mask, ~np.isnan(epsilon_per_dis[0, :]))
        if nbeads > 1:
            assert np.array_equal(mask, ~np.isnan(epsilon_per_dis[:, 1]))
            assert np.array_equal(mask, ~np.isnan(epsilon_per_dis[1, :]))

    # Get mean of epsilon_per_dis
    epsilon = np.nanmean(epsilon_per_dis[np.triu_indices(nbeads, 1)])

    if verbose:
        print(f"MULTISCALE EPSILON ({multiscale_factor}x): {epsilon:.3g}",
              flush=True)
    return epsilon, epsilon_per_dis


def _make_spiral(n_rotations=2, radius=1.2, z_max=4, n_points=1000):
    """TODO"""
    umax = n_rotations * 2 * jnp.pi

    u = jnp.linspace(0, umax, n_points)
    x = radius * jnp.cos(u)
    y = radius * jnp.sin(u)
    z = u / jnp.pi
    z = z * z_max / (n_rotations * 2)

    return jnp.stack([x, y, z], axis=1)


def _spiral_obj(X, multiscale_factor, nghbr_dis_fullres, nghbr_dis_lowres=None,
                epsilon_prev=None, return_extras=False):
    """TODO"""
    n_rotations, radius, z_max = X
    spiral = _make_spiral(
        n_rotations=n_rotations, radius=radius, z_max=z_max,
        n_points=multiscale_factor * 2)

    nghbr_dis_fullres_ = jnp.linalg.norm(spiral[:-1] - spiral[1:], axis=1)
    mse_fullres = jnp.mean(jnp.square(nghbr_dis_fullres - nghbr_dis_fullres_))
    obj = mse_fullres

    nghbr_dis_lowres_ = mse_lowres = None
    if nghbr_dis_lowres is not None:
        nghbr_dis_lowres_ = jnp.linalg.norm(spiral[:multiscale_factor].mean(
            axis=0) - spiral[multiscale_factor:].mean(axis=0))
        mse_lowres = jnp.square(nghbr_dis_lowres - nghbr_dis_lowres_)
        obj = obj + mse_lowres

    epsilon_prev_ = mse_epsilon_prev = None
    if epsilon_prev is not None:
        multiscale_var_prev_ = _var3d(
            spiral.reshape(spiral.shape[0], 1, 3), replace_nan=False)[0]
        epsilon_prev_ = jnp.sqrt(multiscale_var_prev_ * 2 / 3)
        mse_epsilon_prev = jnp.square(epsilon_prev - epsilon_prev_)
        obj = obj + mse_epsilon_prev

    if not return_extras:
        return obj
    else:
        est_vals = {'fullres': nghbr_dis_fullres_, 'lowres': nghbr_dis_lowres_,
                    'prev': epsilon_prev_}
        mse = {'fullres': mse_fullres, 'lowres': mse_lowres,
               'prev': mse_epsilon_prev}
        return obj, mse, est_vals


_spiral_grad = grad(_spiral_obj)
_spiral_fprime = lambda *args, **kwargs: np.array(_spiral_grad(*args, **kwargs))


def toy_struct_max_epilon(multiscale_factor, nghbr_dis_fullres,
                          nghbr_dis_lowres=None, epsilon_prev=None,
                          random_state=None, init=None, bounds=None,
                          verbose=True):
    """TODO"""

    if verbose:
        if nghbr_dis_lowres is None:
            lowres_desc = ''
        else:
            lowres_desc = f"low-res={nghbr_dis_lowres:.3g}... "
        to_print = [
            "\tCreating toy structure with maximum realistic epsilon",
            f"\tTARGET: mean dist between neighboring beads..."
            f" high-res={nghbr_dis_fullres:.3g}"]
        if epsilon_prev is not None:
            to_print.append(f"\tTARGET: epsilon at previous (lower) resolution:"
                            f" {epsilon_prev:.3g}")
        print("\n".join(to_print), flush=True)

    if bounds is None:
        bounds = np.array([[1e-3, None], [1e-3, None], [1e-3, None]])

    if random_state is None:
        random_state = np.random.RandomState(seed=0)
    if init is None:
        init = random_state.uniform(
            low=np.nanmin(bounds[:, 0]), high=10, size=3)

    results = optimize.fmin_l_bfgs_b(
        _spiral_obj, x0=init, fprime=_spiral_fprime, iprint=-1, maxiter=1e4,
        maxfun=1e4, pgtol=1e-05, factr=1e7, bounds=bounds,
        args=(multiscale_factor, nghbr_dis_fullres, nghbr_dis_lowres,
              epsilon_prev))
    X, obj, d = results
    converged = d['warnflag'] == 0
    conv_desc = d['task']
    if isinstance(conv_desc, bytes):
        conv_desc = conv_desc.decode('utf8')
    n_rotations, radius, z_max = X

    obj, mse, est_vals = _spiral_obj(
        X, multiscale_factor=multiscale_factor,
        nghbr_dis_fullres=nghbr_dis_fullres, nghbr_dis_lowres=nghbr_dis_lowres,
        epsilon_prev=epsilon_prev, return_extras=True)

    if verbose:
        if converged:
            print(f'\tCONVERGED with {obj=:.3g}', flush=True)
        else:
            print(f'\tOPTIMIZATION DID NOT CONVERGE\n\t{conv_desc}', flush=True)
    if verbose and converged:
        print(f'\tINFERED STRUCT: 3D spiral of {radius=:.3g} and height='
              f'{z_max:.3g}, undergoes {n_rotations:.3g} rotations', flush=True)
        if nghbr_dis_lowres is None:
            lowres_desc = ''
        else:
            lowres_desc = f"low-res={est_vals['lowres']._value:.3g} (MSE={mse['lowres']._value:.3g})... "
        to_print = [
            "\tINFERRED STRUCT: mean dist between neighboring beads..."
            f" high-res={est_vals['fullres']._value.mean():.3g} (MSE="
            f"{mse['fullres']._value:.3g})"]
        if epsilon_prev is not None:
            to_print.append(
                f"\tINFERRED STRUCT: epsilon at previous (lower) resolution: "
                f"{est_vals['prev']._value:.3g} (MSE={mse['prev']._value:.3g})")
        print("\n".join(to_print), flush=True)

    if converged:
        spiral = _make_spiral(
            n_rotations=n_rotations, radius=radius, z_max=z_max,
            n_points=multiscale_factor * 2)._value
        est_epsilon_max, _ = get_epsilon_from_struct(
            spiral, lengths=multiscale_factor * 2, ploidy=1,
            multiscale_factor=multiscale_factor, verbose=False)
        return est_epsilon_max, spiral
    else:
        return None, None


def _choose_max_multiscale_factor(lengths, min_beads):
    """Find the lowest resolution where structures have at least `min_beads`."""

    multiscale_factor = 1
    while np.min(decrease_lengths_res(
            lengths, multiscale_factor=multiscale_factor * 2)) >= min_beads:
        multiscale_factor *= 2

    return multiscale_factor


def _choose_max_multiscale_rounds(lengths, min_beads):
    """Choose the maximum number of multiscale rounds, given min_beads."""

    multiscale_factor = _choose_max_multiscale_factor(
        lengths, min_beads=min_beads)
    multiscale_rounds = np.log2(multiscale_factor) + 1
    return multiscale_rounds
