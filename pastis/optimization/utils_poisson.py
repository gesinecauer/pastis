from __future__ import print_function

import numpy as np
import sys
import os
import pandas as pd
from scipy import sparse
from distutils.util import strtobool

from absl import logging as absl_logging
absl_logging.set_verbosity('error')
from jax.config import config as jax_config
jax_config.update("jax_platform_name", "cpu")
jax_config.update("jax_enable_x64", True)

from typing import Any as Array
import jax.numpy as jnp
from jax import custom_jvp
from jax import lax


if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")


def _print_code_header(header, max_length=80, blank_lines=1, verbose=True):
    """Prints a header, for demarcation of output.
    """

    if verbose:
        if not isinstance(header, list):
            header = [header]
        if blank_lines is not None and blank_lines > 0:
            print('\n' * (blank_lines - 1), flush=True)
        print('=' * max_length, flush=True)
        for line in header:
            pad_left = ('=' * int(np.ceil((max_length - len(line) - 2) / 2)))
            if pad_left != '':
                pad_left = pad_left + ' '
            pad_right = ('=' * int(np.floor((max_length - len(line) - 2) / 2)))
            if pad_right != '':
                pad_right = ' ' + pad_right
            print(pad_left + line + pad_right, flush=True)
        print('=' * max_length, flush=True)


def _load_infer_var(infer_var_file):
    """Loads a dictionary of inference variables, including alpha and beta.
    """

    infer_var = dict(pd.read_csv(
        infer_var_file, sep='\t', header=None, squeeze=True,
        index_col=0, dtype=str))
    infer_var['beta'] = [float(b) for b in infer_var['beta'].split()]
    if 'seed' in infer_var:
        infer_var['seed'] = int(float(infer_var['seed']))
    if 'hsc_r' in infer_var:
        infer_var['hsc_r'] = np.array([float(
            r) for r in infer_var['hsc_r'].split()])
    if 'mhs_k' in infer_var:
        infer_var['mhs_k'] = np.array([float(
            r) for r in infer_var['mhs_k'].split()])
    if 'shn_sigma' in infer_var:
        infer_var['shn_sigma'] = float(infer_var['shn_sigma'])
    if 'orient' in infer_var:
        infer_var['orient'] = np.array([float(
            r) for r in infer_var['orient'].split()])
    if 'multiscale_variances' in infer_var:
        infer_var['multiscale_variances'] = float(
            infer_var['multiscale_variances'])
    if 'epsilon' in infer_var:
        if ' ' not in infer_var['epsilon']:
            infer_var['epsilon'] = float(infer_var['epsilon'])
        else:
            infer_var['epsilon'] = np.array(map(float, infer_var['epsilon'].split(' ')))
    infer_var['alpha'] = float(infer_var['alpha'])
    infer_var['converged'] = strtobool(infer_var['converged'])
    return infer_var


def _output_subdir(outdir, chrom_full=None, chrom_subset=None, null=False,
                   piecewise_step=None):
    """Returns subdirectory for inference output files.
    """

    if null:
        outdir = os.path.join(outdir, 'null')

    if chrom_subset is not None and chrom_full is not None:
        if len(chrom_subset) != len(chrom_full):
            outdir = os.path.join(outdir, '.'.join(chrom_subset))

    if piecewise_step is not None:
        if isinstance(piecewise_step, list):
            if len(piecewise_step) == 1:
                piecewise_step = piecewise_step[0]
            else:
                raise ValueError("`piecewise_step` must be None or int.")
        if piecewise_step == 1:
            outdir = os.path.join(outdir, 'step1_lowres_genome')
        elif piecewise_step == 2:
            outdir = os.path.join(outdir, 'step2_fullres_chrom')
        elif piecewise_step == 3:
            outdir = os.path.join(outdir, 'step3_reoriented_chrom')

    return outdir


def _format_structures(structures, lengths=None, ploidy=None,
                       mixture_coefs=None):
    """Reformats and checks shape of structures.
    """

    from .poisson import _format_X

    if isinstance(structures, list):
        if not all([isinstance(struct, jnp.ndarray) for struct in structures]):
            raise ValueError("Individual structures must use numpy.ndarray"
                             "format.")
        try:
            structures = [struct.reshape(-1, 3) for struct in structures]
        except ValueError:
            raise ValueError("Structures should be composed of 3D coordinates")
    else:
        if not isinstance(structures, jnp.ndarray):
            raise ValueError("Structures must be numpy.ndarray or list of"
                             "numpy.ndarrays.")
        try:
            structures = structures.reshape(-1, 3)
        except ValueError:
            raise ValueError("Structure should be composed of 3D coordinates")
        structures, _, _ = _format_X(
            structures, lengths=lengths, ploidy=ploidy,
            mixture_coefs=mixture_coefs)

    if mixture_coefs is not None and len(structures) != len(mixture_coefs):
        raise ValueError("The number of structures (%d) and of mixture "
                         "coefficents (%d) should be identical." %
                         (len(structures), len(mixture_coefs)))

    if len(set([struct.shape[0] for struct in structures])) > 1:
        raise ValueError("Structures are of different shapes.")

    if lengths is not None and ploidy is not None:
        nbeads = lengths.sum() * ploidy
        for struct in structures:
            if struct.shape != (nbeads, 3):
                raise ValueError("Structure is of unexpected shape. Expected"
                                 " shape of (%d, 3), structure is (%s)."
                                 % (nbeads,
                                    ', '.join([str(x) for x in struct.shape])))

    return structures


def find_beads_to_remove(counts, lengths, ploidy, multiscale_factor=1,
                         threshold=0):
    """Determine beads for which no corresponding counts data exists.

    Identifies beads that should be removed (set to NaN) in the structure.
    If there aren't any counts in the rows/columns corresponding to a given
    bead, that bead should be removed.

    Parameters
    ----------
    counts : list of np.ndarray or scipy.sparse.coo_matrix
        Counts data, at the resolution specified by `multiscale_factor`.
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.

    Returns
    -------
    torm : array of bool of shape (nbeads,)
        Beads that should be removed (set to NaN) in the structure.
    """

    from .multiscale_optimization import decrease_lengths_res
    from .counts import CountsMatrix

    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
    nbeads = lengths_lowres.sum() * ploidy

    if not isinstance(counts, list):
        counts = [counts]
    if len(counts) == 0:
        raise ValueError("Counts is an empty list.")
    if all([isinstance(c, CountsMatrix) for c in counts]):
        # If counts are already formatted, they may contain multiple resolutions
        counts = [c for c in counts if c.multiscale_factor == multiscale_factor]
    if len(counts) == 0:
        raise ValueError(
            "Resolution of counts is not consistent with lengths at"
            f" multiscale_factor={multiscale_factor}.")

    inverse_torm = np.zeros(int(nbeads))
    for c in counts:
        if max(c.shape) not in (nbeads, nbeads / ploidy):
            raise ValueError(
                "Resolution of counts is not consistent with lengths at"
                f" multiscale_factor={multiscale_factor}. Counts shape is ("
                f"{', '.join(map(str, c.shape))}).")

        if isinstance(c, np.ndarray):
            axis0sum = np.tile(
                np.array(np.nansum(c, axis=0).flatten()).flatten(),
                int(nbeads / c.shape[1]))
            axis1sum = np.tile(
                np.array(np.nansum(c, axis=1).flatten()).flatten(),
                int(nbeads / c.shape[0]))
        else:
            axis0sum = np.tile(
                np.array(c.sum(axis=0).flatten()).flatten(),
                int(nbeads / c.shape[1]))
            axis1sum = np.tile(
                np.array(c.sum(axis=1).flatten()).flatten(),
                int(nbeads / c.shape[0]))
        inverse_torm += (axis0sum + axis1sum > threshold).astype(int)

    torm = ~ inverse_torm.astype(bool)
    # TODO it would save memory to return np.where(torm)[0]
    return torm


def _struct_replace_nan(struct, lengths, kind='linear', random_state=None):
    """Replace NaNs in structure via linear interpolation.
    """

    from scipy.interpolate import interp1d
    from warnings import warn
    from sklearn.utils import check_random_state

    if random_state is None:
        random_state = np.random.RandomState(seed=0)
    random_state = check_random_state(random_state)

    if isinstance(struct, str):
        struct = np.loadtxt(struct)
    else:
        struct = struct.copy()
    struct = struct.reshape(-1, 3)
    lengths = np.array(lengths).astype(int)

    ploidy = 1
    if len(struct) > lengths.sum():
        ploidy = 2

    if not np.isnan(struct).any():
        return(struct)
    else:
        nan_chroms = []
        mask = np.invert(np.isnan(struct[:, 0]))
        interpolated_struct = np.zeros(struct.shape)
        begin, end = 0, 0
        for j, length in enumerate(np.tile(lengths, ploidy)):
            end += length
            to_rm = mask[begin:end]
            if to_rm.sum() <= 1:
                interpolated_chr = (
                    1 - 2 * random_state.rand(length * 3)).reshape(-1, 3)
                if ploidy == 1:
                    nan_chroms.append(str(j + 1))
                else:
                    nan_chroms.append(
                        str(j + 1) + '_homo1' if j < lengths.shape[0] else str(j / 2 + 1) + '_homo2')
            else:
                m = np.arange(length)[to_rm]
                beads2interpolate = np.arange(m.min(), m.max() + 1, 1)

                interpolated_chr = np.full_like(struct[begin:end, :], np.nan)
                interpolated_chr[beads2interpolate, 0] = interp1d(
                    m, struct[begin:end, 0][to_rm], kind=kind)(beads2interpolate)
                interpolated_chr[beads2interpolate, 1] = interp1d(
                    m, struct[begin:end, 1][to_rm], kind=kind)(beads2interpolate)
                interpolated_chr[beads2interpolate, 2] = interp1d(
                    m, struct[begin:end, 2][to_rm], kind=kind)(beads2interpolate)

                # Fill in beads at start
                diff_beads_at_chr_start = interpolated_chr[beads2interpolate[
                    1], :] - interpolated_chr[beads2interpolate[0], :]
                how_far = 1
                for j in reversed(range(min(beads2interpolate))):
                    interpolated_chr[j, :] = interpolated_chr[
                        beads2interpolate[0], :] - diff_beads_at_chr_start * how_far
                    how_far += 1
                # Fill in beads at end
                diff_beads_at_chr_end = interpolated_chr[
                    beads2interpolate[-2], :] - interpolated_chr[beads2interpolate[-1], :]
                how_far = 1
                for j in range(max(beads2interpolate) + 1, length):
                    interpolated_chr[j, :] = interpolated_chr[
                        beads2interpolate[-1], :] - diff_beads_at_chr_end * how_far
                    how_far += 1

            interpolated_struct[begin:end, :] = interpolated_chr
            begin = end

        if len(nan_chroms) != 0:
            warn('The following chromosomes were all NaN: ' + ' '.join(nan_chroms))

        return(interpolated_struct)


@custom_jvp
def jax_max(x1: Array, x2: Array) -> Array:
    """Element-wise maximum of array elements.

    Compare two arrays and returns a new array containing the element-wise
    maxima. If one of the elements being compared is a NaN, then that element is
    returned. If both elements are NaNs then the first is returned. The latter
    distinction is important for complex NaNs, which are defined as at least one
    of the real or imaginary parts being a NaN. The net effect is that NaNs are
    propagated.

    Parameters
    ----------
    x1,x2 : array-like
        The arrays holding the elements to be compared. If `x1.shape !=
        x2.shape`, they must be broadcastable to a common shape (which becomes
        the shape of the output).

    Returns
    -------
    obj : array or scalar
        The maximum of x1 and x2, element-wise. This is a scalar if both x1 and
        x2 are scalars.
    """
    return jnp.maximum(x1, x2)


# FIXME double check this
jax_max.defjvps(
    lambda g1, ans, x1, x2: lax.select(x1 > x2, g1, lax.full_like(g1, 0)),
    lambda g2, ans, x1, x2: lax.select(x1 < x2, g2, lax.full_like(g2, 0)))


def subset_chrom(lengths_full, chrom_full, chrom_subset=None):
    """Return lengths, names, and indices for selected chromosomes only.

    If `chrom_subset` is None, return original lengths and chromosome names.
    Otherwise, only return lengths, chromosome names, and indices for
    chromosomes specified by `chrom_subset`.

    Parameters
    ----------
    lengths_full : array of int
        Number of beads per homolog of each chromosome in the full data.
    chrom_full : array of str
        Label for each chromosome in the full data, or file with chromosome
        labels (one label per line).
    chrom_subset : array of str, optional
        Label for each chromosome to be excised from the full data. If None,
        the full data will be returned..

    Returns
    -------
    lengths_subset : array of int
        Number of beads per homolog of each chromosome in the subsetted data
        for the specified chromosomes.
    chrom_subset : array of str
        Label for each chromosome in the subsetted data, in the order indicated
        by `chrom_full`.
    subset_index : array or None
        If `chrom_subset` is None or is equivalent to `chrom_full`, return None.
        Otherwise, return array with mask of subsetted chrom, of size
        `lengths_full.sum() * ploidy`.
    """

    if not isinstance(chrom_full, np.ndarray) or chrom_full.shape == ():
        chrom_full = np.array([chrom_full]).flatten()
    if not isinstance(lengths_full, np.ndarray) or lengths_full.shape == ():
        lengths_full = np.array([lengths_full]).flatten()
    lengths_full = lengths_full.astype(int)

    if chrom_subset is None:
        chrom_subset = chrom_full.copy()
    else:
        if not isinstance(chrom_subset, np.ndarray) or chrom_subset.shape == ():
            chrom_subset = np.array([chrom_subset]).flatten()
        missing_chrom = [x for x in chrom_subset if x not in chrom_full]
        if len(missing_chrom) > 0:
            raise ValueError("Chromosomes to be subsetted (%s) are not in full"
                             " list of chromosomes (%s)" %
                             (','.join(missing_chrom), ','.join(chrom_full)))
        # Make sure chrom_subset is sorted properly
        chrom_subset = np.array(
            [chrom for chrom in chrom_full if chrom in chrom_subset])

    if np.array_equal(chrom_subset, chrom_full):
        # Not subsetting chrom
        lengths_subset = lengths_full.copy()
        subset_index = None
    else:
        # Subsetting chrom
        lengths_subset = np.array([lengths_full[i] for i in range(
            len(chrom_full)) if chrom_full[i] in chrom_subset])
        subset_index = []
        for i in range(len(lengths_full)):
            subset_index.append(
                np.full((lengths_full[i],), chrom_full[i] in chrom_subset))
        subset_index = np.concatenate(subset_index)

    return lengths_subset, chrom_subset, subset_index


# TODO add on main branch - subset_chrom->subset_chrom_of_data & _get_chrom_subset_index->subset_chrom
def subset_chrom_of_data(ploidy, lengths_full, chrom_full, chrom_subset=None,
                         counts=None, structures=None, exclude_zeros=False):
    """Return data for selected chromosomes only.

    If `chrom_subset` is None, return original data. Otherwise, only return
    data for chromosomes specified by `chrom_subset`.

    Parameters
    ----------
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    lengths_full : array of int
        Number of beads per homolog of each chromosome in the full data.
    chrom_full : array of str
        Label for each chromosome in the full data, or file with chromosome
        labels (one label per line).
    chrom_subset : array of str, optional
        Label for each chromosome to be excised from the full data. If None,
        the full data will be returned.
    counts : list of array or coo_matrix, optional
        Full counts data.
    structures : array or list of array, optional
        Structure(s) with all chromosomes.

    Returns
    -------
    lengths_subset : array of int
        Number of beads per homolog of each chromosome in the subsetted data
        for the specified chromosomes.
    chrom_subset : array of str
        Label for each chromosome in the subsetted data, in the order indicated
        by `chrom_full`.
    counts : list of array or coo_matrix, or None
        If `counts` is inputted, subsetted counts data containing only the
        specified chromosomes. Otherwise, None.
    structures : array, list of array, or None
        If `structures` is inputted, subsetted structure(s) containing only
        the specified chromosomes. Otherwise, None.
    """

    from .counts import check_counts

    lengths_subset, chrom_subset, subset_index = subset_chrom(
        lengths_full=lengths_full, chrom_full=chrom_full,
        chrom_subset=chrom_subset)

    if subset_index is not None and ploidy == 2:
        subset_index = np.tile(subset_index, 2)

    if counts is not None:
        counts = check_counts(
            counts, lengths=lengths_full, ploidy=ploidy,
            exclude_zeros=exclude_zeros, chrom_subset_index=subset_index)

    if subset_index is not None and structures is not None:
        if isinstance(structures, list):
            for i in range(len(structures)):
                structures[i] = structures[i].reshape(-1, 3)[
                    subset_index].reshape(*structures[i].shape)
        else:
            structures = structures.reshape(-1, 3)[subset_index].reshape(
                *structures.shape)

    return lengths_subset, chrom_subset, counts, structures


def _intra_counts_mask(counts, lengths_counts):
    """Return mask of sparse COO data for intra-chromosomal counts.
    """

    if isinstance(counts, np.ndarray):
        counts = sparse.coo_matrix(counts)
    elif not sparse.issparse(counts):
        counts = counts.tocoo()
    bins_for_row = np.tile(
        lengths_counts, int(counts.shape[0] / lengths_counts.sum())).cumsum()
    bins_for_col = np.tile(
        lengths_counts, int(counts.shape[1] / lengths_counts.sum())).cumsum()
    row_binned = np.digitize(counts.row, bins_for_row)
    col_binned = np.digitize(counts.col, bins_for_col)

    if counts.shape[0] != counts.shape[1]:
        nchrom = lengths_counts.shape[0]
        row_binned[row_binned >= nchrom] -= nchrom
        col_binned[col_binned >= nchrom] -= nchrom

    return np.equal(row_binned, col_binned)


def _intra_counts(counts, lengths_counts, ploidy, exclude_zeros=False):
    """Return intra-chromosomal counts.
    """

    from .counts import _check_counts_matrix

    if isinstance(counts, np.ndarray):
        counts = counts.copy()
        counts[np.isnan(counts)] = 0
        counts = sparse.coo_matrix(counts)
    elif not sparse.issparse(counts):
        counts = counts.tocoo()

    counts = _check_counts_matrix(
        counts, lengths=lengths_counts, ploidy=ploidy, exclude_zeros=True)

    mask = _intra_counts_mask(counts=counts, lengths_counts=lengths_counts)
    counts_new = sparse.coo_matrix(
        (counts.data[mask], (counts.row[mask], counts.col[mask])),
        shape=counts.shape)

    if exclude_zeros:
        return counts_new
    else:
        counts_array = np.full(counts_new.shape, np.nan)
        counts_array[counts_new.row, counts_new.col] = counts_new.data
        return counts_array


def _inter_counts(counts, lengths_counts, ploidy, exclude_zeros=False):
    """Return inter-chromosomal counts.
    """

    from .counts import _check_counts_matrix

    if isinstance(counts, np.ndarray):
        counts = counts.copy()
        counts[np.isnan(counts)] = 0
        counts = sparse.coo_matrix(counts)
    elif not sparse.issparse(counts):
        counts = counts.tocoo()

    counts = _check_counts_matrix(
        counts, lengths=lengths_counts, ploidy=ploidy, exclude_zeros=True)

    mask = ~_intra_counts_mask(counts=counts, lengths_counts=lengths_counts)
    counts_new = sparse.coo_matrix(
        (counts.data[mask], (counts.row[mask], counts.col[mask])),
        shape=counts.shape)

    if exclude_zeros:
        return counts_new
    else:
        counts_array = np.full(counts_new.shape, np.nan)
        counts_array[counts_new.row, counts_new.col] = counts_new.data
        return counts_array


def _counts_near_diag(counts, lengths_counts, ploidy, nbins, exclude_zeros=False):
    """Return intra-chromosomal counts within `nbins` of diagonal.
    """

    from .counts import _check_counts_matrix

    if isinstance(counts, np.ndarray):
        counts = counts.copy()
        counts[np.isnan(counts)] = 0
        counts = sparse.coo_matrix(counts)
    elif not sparse.issparse(counts):
        counts = counts.tocoo()

    counts = _check_counts_matrix(
        counts, lengths=lengths_counts, ploidy=ploidy, exclude_zeros=True)

    mask_intra = _intra_counts_mask(counts=counts, lengths_counts=lengths_counts)

    row = counts.row.copy()
    col = counts.col.copy()
    if counts.shape[0] != counts.shape[1]:
        n = lengths_counts.sum()
        row[row >= n] -= n
        col[col >= n] -= n
    mask_near_diag = np.abs(row - col) <= nbins

    mask = mask_intra & mask_near_diag

    counts_new = sparse.coo_matrix(
        (counts.data[mask], (counts.row[mask], counts.col[mask])),
        shape=counts.shape)

    if exclude_zeros:
        return counts_new
    else:
        counts_array = np.full(counts_new.shape, np.nan)
        counts_array[counts_new.row, counts_new.col] = counts_new.data
        return counts_array
