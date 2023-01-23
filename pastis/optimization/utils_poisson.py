from __future__ import print_function

import numpy as np
import sys
import os
import textwrap
import pandas as pd
from scipy import sparse
from distutils.util import strtobool
from scipy.interpolate import interp1d
import warnings


def _setup_jax(debug_nan_inf=False):
    from absl import logging as absl_logging
    absl_logging.set_verbosity('error')
    from jax.config import config as jax_config
    jax_config.update("jax_platform_name", "cpu")
    jax_config.update("jax_enable_x64", True)
    # os.environ.update(
    #     XLA_FLAGS=(
    #         '--xla_cpu_multi_thread_eigen=false '
    #         'intra_op_parallelism_threads=1 '
    #         'inter_op_parallelism_threads=1 '
    #     ),
    #     XLA_PYTHON_CLIENT_PREALLOCATE='false',
    # )
    if debug_nan_inf:
        jax_config.update("jax_debug_nans", True)
        jax_config.update("jax_debug_infs", True)
    # jax_config.update("jax_check_tracer_leaks", True)


_setup_jax()

from typing import Any as Array
import jax.numpy as jnp
from jax import custom_jvp
from jax import lax
from jax.nn import relu
import jax


if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")


def _get_output_files(outdir, seed=None):
    """TODO"""
    if seed is None:
        seed_str = ''
    else:
        seed_str = f'.{seed:03d}'
    out_file = os.path.join(outdir, f'struct_inferred{seed_str}.coords')
    out_fail = os.path.join(
        outdir, f'struct_nonconverged{seed_str}.coords')
    init_file = os.path.join(outdir, f'struct_init{seed_str}.coords')
    history_file = os.path.join(outdir, f'history{seed_str}')
    infer_param_file = os.path.join(
        outdir, f'inference_params{seed_str}')
    reorient_file = os.path.join(
        outdir, f'reorient_inferred{seed_str}.coords')
    reorient_init_file = os.path.join(
        outdir, f'reorient_init{seed_str}.coords')
    outfiles = {
        'struct_infer': out_file, 'struct_nonconv': out_fail,
        'history': history_file, 'infer_param': infer_param_file,
        'init': init_file, 'dir': outdir, 'reorient': reorient_file,
        'reorient_init': reorient_init_file}
    return outfiles


def _euclidean_distance(struct, row, col):
    """Get euclidian distances between beads in given row and col of struct."""
    dis_sq = (jnp.square(struct[row] - struct[col])).sum(axis=1)
    return jnp.sqrt(dis_sq)


def _print_code_header(header, max_length=80, blank_lines=1, verbose=True):
    """Prints a header, for demarcation of output."""

    if not verbose:
        return
    if not isinstance(header, list):
        header = [header]
    if blank_lines is not None and blank_lines > 0:
        print('\n' * (blank_lines - 1), flush=True)
    print('=' * max_length, flush=True)
    for line_full in header:
        for line in textwrap.wrap(line_full, width=max_length - 4):
            print(f" {line} ".center(max_length, '='), flush=True)
    print('=' * max_length, flush=True)


def _load_infer_param(infer_param_file):
    """Loads a dictionary of inference variables, including alpha and beta.
    """

    infer_param = dict(pd.read_csv(
        infer_param_file, sep='\t', header=None, index_col=0,
        dtype=str).squeeze("columns"))

    for key in ['beta', 'est_hmlg_sep', 'orient', 'epsilon']:
        if key in infer_param:
            infer_param[key] = np.array(
                infer_param[key].strip('[]').split(), dtype=float)
    convert_type_fxns = {
        'alpha': [float], 'converged': [strtobool], 'seed': [float, int],
        'multiscale_variances': [float], 'obj': [float], 'time': [float],
        'alpha_converged': [strtobool], 'alpha_obj': [float],
        'alpha_loop': [int], 'rescale_by': [float]}
    for key, type_fxns in convert_type_fxns.items():
        if key in infer_param:
            for type_fxn in type_fxns:
                infer_param[key] = type_fxn(infer_param[key])

    if 'epsilon' in infer_param and infer_param['epsilon'].size == 1:
        infer_param['epsilon'] = infer_param['epsilon'][0]  # TODO just put epsilon above

    return infer_param


def _output_subdir(outdir, chrom_full=None, chrom_subset=None, null=False,
                   piecewise_step=None, lengths=None):
    """Returns subdirectory for inference output files."""
    from ..io.read import _get_chrom

    raise NotImplementedError('update me?')

    if outdir is None:
        return None

    if null:
        outdir = os.path.join(outdir, 'null')

    if chrom_subset is not None and chrom_full is not None:
        chrom_full = _get_chrom(chrom_full, lengths=lengths)
        chrom_subset = _get_chrom(chrom_subset)
        if chrom_subset.size != chrom_full.size:
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
    """Reformats and checks shape of structures."""

    # TODO this function was written by an idiot

    from .poisson import _format_X

    lengths = np.array(lengths, copy=False, ndmin=1, dtype=int).ravel()

    if isinstance(structures, list):
        if not all([isinstance(
                struct, (np.ndarray, jnp.ndarray)) for struct in structures]):
            raise ValueError("Individual structures must use numpy.ndarray"
                             "format.")
        try:
            structures = [struct.reshape(-1, 3) for struct in structures]
        except ValueError:
            raise ValueError("Structures should be composed of 3D coordinates")
    else:
        if not isinstance(structures, (np.ndarray, jnp.ndarray)):
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
    struct_nan : array of int
        Beads that should be removed (set to NaN) in the structure.
    """

    from .multiscale_optimization import decrease_lengths_res

    lengths_lowres = decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor)
    n = lengths_lowres.sum()
    nbeads = n * ploidy

    if not isinstance(counts, list):
        counts = [counts]
    if len(counts) == 0:
        raise ValueError("Counts is an empty list.")

    inverse_struct_nan_mask = np.zeros(int(nbeads))
    for c in counts:
        if set(c.shape) not in ({n}, {nbeads}, {nbeads, n}):
            raise ValueError(
                "Resolution of counts is not consistent with lengths at"
                f" {multiscale_factor=}... counts.shape={c.shape},"
                f" {lengths_lowres.sum()=}.")
        axis0sum = np.tile(
            np.array(c.sum(axis=0).ravel()).ravel(), int(nbeads / c.shape[1]))
        axis1sum = np.tile(
            np.array(c.sum(axis=1).ravel()).ravel(), int(nbeads / c.shape[0]))
        inverse_struct_nan_mask += (axis0sum + axis1sum > threshold).astype(int)

    struct_nan_mask = ~inverse_struct_nan_mask.astype(bool)
    struct_nan = np.where(struct_nan_mask)[0]
    return struct_nan


def _struct_replace_nan(struct, lengths, ploidy, kind='linear',
                        random_state=None, chromosomes=None):
    """Replace empty (NaN) beads in structure via linear interpolation."""
    from ..io.read import _get_chrom

    struct = struct.reshape(-1, 3)
    lengths = np.array(lengths, copy=False, ndmin=1, dtype=int).ravel()

    if struct.shape[0] != lengths.sum() * ploidy:
        raise ValueError(f"The structure must contain {lengths.sum() * ploidy}"
                         f" beads. It contains {struct.shape[0]} beads.")

    if not np.isnan(struct).any():
        return struct

    chromosomes = _get_chrom(chromosomes, lengths=lengths)

    if random_state is None:
        random_state = np.random.RandomState(seed=0)

    empty_chroms = {f"hmlg{i + 1}": [] for i in range(ploidy)}
    mask = np.invert(np.isnan(struct[:, 0]))
    struct_interp = struct.copy()
    begin, end = 0, 0
    for i, length in enumerate(np.tile(lengths, ploidy)):
        end += length
        mask_chrom = mask[begin:end]
        if (~mask_chrom).sum() == 0:   # No NaN beads in molecule
            pass
        elif mask_chrom.sum() == 0:  # 0 non-NaN beads in molecule
            random_chrom = random_state.uniform(
                low=-1, high=1, size=((~mask_chrom).sum(), 3))
            struct_interp[begin:end] = random_chrom
            if i < lengths.size:
                empty_chroms['hmlg1'].append(f'num{i + 1:g}')
            else:
                empty_chroms['hmlg2'].append(f'num{i + 1 - lengths.size:g}')
        elif mask_chrom.sum() == 1:  # Only 0-1 non-NaN beads in molecule
            struct_interp[begin:end][~mask_chrom] = random_state.normal(
                struct_interp[begin:end][mask_chrom], 1,
                ((~mask_chrom).sum(), 3))
        else:  # There are enough non-NaN beads in molecule to interpolate
            idx_orig = np.arange(length)[mask_chrom]
            idx_interp = np.arange(length)
            interp_chrom = np.full_like(struct[begin:end], np.nan)
            interp_chrom[idx_interp, 0] = interp1d(
                idx_orig, struct[begin:end, 0][mask_chrom], kind=kind,
                fill_value="extrapolate")(idx_interp)
            interp_chrom[idx_interp, 1] = interp1d(
                idx_orig, struct[begin:end, 1][mask_chrom], kind=kind,
                fill_value="extrapolate")(idx_interp)
            interp_chrom[idx_interp, 2] = interp1d(
                idx_orig, struct[begin:end, 2][mask_chrom], kind=kind,
                fill_value="extrapolate")(idx_interp)
            struct_interp[begin:end] = interp_chrom
        begin = end

    if any([len(x) != 0 for x in empty_chroms.values()]):
        if ploidy == 1 or set(empty_chroms['hmlg1']) == set(
                empty_chroms['hmlg2']):
            warnings.warn("All beads in the following chromosomes were NaN:"
                          " " + ', '.join(empty_chroms['hmlg1']))
        else:
            warnings.warn("All beads in the following molecules were NaN:"
                          f"\nHomolog1: {', '.join(empty_chroms['hmlg1'])}"
                          f"\nHomolog2: {', '.join(empty_chroms['hmlg2'])}")

    return struct_interp


def relu_min(x1, x2):
    """Returns min(x1, x2), jax-compatible."""
    return - (relu((-x1) - (-x2)) + (-x2))


def relu_max(x1, x2):
    """Returns max(x1, x2), jax-compatible."""
    return relu(x1 - x2) + x2


@custom_jvp
@jax.jit
def jax_max(x1: Array, x2: Array) -> Array:
    """Element-wise maximum of array elements.

    Compare two arrays and returns a new array containing the element-wise
    maxima. If one of the elements being compared is a NaN, then that element is
    returned. If both elements are NaNs then the first is returned. The latter
    distinction is important for complex NaNs, which are defined as at least one
    of the real or imaginary parts being a NaN. The net effect is that NaNs are
    propagated.

    Under differentiation, we take:

    .. math::
        \nabla \mathrm{max}(x, x) = 0

    Parameters
    ----------
    x1,x2 : array-like
        The arrays holding the elements to be compared. If `x1.shape !=
        x2.shape`, they must be broadcastable to a common shape (which becomes
        the shape of the output).

    Returns
    -------
    x : array or scalar
        The maximum of x1 and x2, element-wise. This is a scalar if both x1 and
        x2 are scalars.
    """
    return jnp.maximum(x1, x2)


jax_max.defjvps(
    lambda g1, ans, x1, x2: lax.select(x1 > x2, g1, lax.full_like(g1, 0)),
    lambda g2, ans, x1, x2: lax.select(x1 < x2, g2, lax.full_like(g2, 0)))


@custom_jvp
@jax.jit
def jax_min(x1: Array, x2: Array) -> Array:
    """Element-wise minimum of array elements.

    Compare two arrays and returns a new array containing the element-wise
    minima. If one of the elements being compared is a NaN, then that element is
    returned. If both elements are NaNs then the first is returned. The latter
    distinction is important for complex NaNs, which are defined as at least one
    of the real or imaginary parts being a NaN. The net effect is that NaNs are
    propagated.

    Under differentiation, we take:

    .. math::
        \nabla \mathrm{min}(x, x) = 0

    Parameters
    ----------
    x1,x2 : array-like
        The arrays holding the elements to be compared. If `x1.shape !=
        x2.shape`, they must be broadcastable to a common shape (which becomes
        the shape of the output).

    Returns
    -------
    x : array or scalar
        The minimum of x1 and x2, element-wise. This is a scalar if both x1 and
        x2 are scalars.
    """
    return jnp.minimum(x1, x2)


jax_min.defjvps(
    lambda g1, ans, x1, x2: lax.select(x1 < x2, g1, lax.full_like(g1, 0)),
    lambda g2, ans, x1, x2: lax.select(x1 > x2, g2, lax.full_like(g2, 0)))


def subset_chromosomes(lengths_full, chrom_full, chrom_subset=None):
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
    subset_idx : array or None
        If `chrom_subset` is None or is equivalent to `chrom_full`, return None.
        Otherwise, return array with mask of subsetted chrom, of size
        `lengths_full.sum() * ploidy`.
    """

    lengths_full = np.array(
        lengths_full, copy=False, ndmin=1, dtype=int).ravel()
    chrom_full = np.array(chrom_full, copy=False, ndmin=1, dtype=str).ravel()

    if chrom_full.size != np.unique(chrom_full).size:
        raise ValueError("Chromosome names may not contain duplicates.")
    if chrom_full.size != lengths_full.size:
        raise ValueError(
            f"Size of chromosome names ({chrom_full.size}) does not"
            f"match size of chromosome lengths ({lengths_full.size}).")

    if chrom_subset is None:
        chrom_subset = chrom_full.copy()
    else:
        chrom_subset = np.array(
            chrom_subset, copy=False, ndmin=1, dtype=str).ravel()

        if chrom_subset.size != np.unique(chrom_subset).size:
            raise ValueError(
                "List of chromosomes to subset may not contain duplicates.")
        missing_chrom = chrom_subset[~np.isin(chrom_subset, chrom_full)]
        if missing_chrom.size > 0:
            raise ValueError(
                f"Chromosomes to be subsetted ({', '.join(missing_chrom)}) are"
                f" not in full list of chromosomes ({', '.join(chrom_full)}).")
        if chrom_subset.size == 0:
            raise ValueError(f"No chromosomes selected, {chrom_subset.size=}.")

        # Make sure chrom_subset is sorted properly
        chrom_subset = chrom_full[np.isin(chrom_full, chrom_subset)]

    if np.array_equal(chrom_subset, chrom_full):
        # Not subsetting chrom
        lengths_subset = lengths_full.copy()
        subset_idx = None
    else:
        # Subsetting chrom
        lengths_subset = lengths_full[np.isin(chrom_full, chrom_subset)]
        subset_mask = []
        for i in range(len(lengths_full)):
            subset_mask.append(
                np.full((lengths_full[i],), chrom_full[i] in chrom_subset))
        subset_mask = np.concatenate(subset_mask)
        subset_idx = np.where(subset_mask)[0]

    return lengths_subset, chrom_subset, subset_idx


def subset_chrom_of_data(ploidy, lengths_full, chrom_full, chrom_subset=None,
                         counts=None, bias=None, structures=None):
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

    Returns  TODO update
    -------
    lengths_subset : array of int
        Number of beads per homolog of each chromosome in the subsetted data
        for the specified chromosomes.
    chrom_subset : array of str
        Label for each chromosome in the subsetted data, in the order indicated
        by `chrom_full`.
    counts : list of coo_matrix, or None
        If `counts` is inputted, subsetted counts data containing only the
        specified chromosomes. Otherwise, None.
    structures : array, list of array, or None
        If `structures` is inputted, subsetted structure(s) containing only
        the specified chromosomes. Otherwise, None.
    """

    from .counts import check_counts

    lengths_full = np.array(
        lengths_full, copy=False, ndmin=1, dtype=int).ravel()
    chrom_full = np.array(chrom_full, copy=False, ndmin=1, dtype=str).ravel()

    if bias is not None and bias.size != lengths_full.sum():
        raise ValueError("Bias size must be equal to the sum of the chromosome "
                         f"lengths ({lengths_full.sum()}). It is of size"
                         f" {bias.size}.")
    if structures is not None:
        if isinstance(structures, list):
            struct_list = structures
        else:
            struct_list = [structures]
        for i in range(len(struct_list)):
            if struct_list[i].size != lengths_full.sum() * ploidy * 3:
                raise ValueError(
                    f"Structure shape {struct_list[i].shape} is inconsistent"
                    f" with number of beads ({lengths_full.sum() * ploidy}).")

    lengths_subset, chrom_subset, subset_idx = subset_chromosomes(
        lengths_full=lengths_full, chrom_full=chrom_full,
        chrom_subset=chrom_subset)

    if subset_idx is not None and ploidy == 2:
        subset_idx = np.append(subset_idx, subset_idx + lengths_full.sum())

    if counts is not None:
        counts = check_counts(
            counts, lengths=lengths_full, ploidy=ploidy,
            chrom_subset_idx=subset_idx)

    if subset_idx is not None and bias is not None:
        bias = bias[subset_idx[subset_idx < bias.size]]

    if subset_idx is not None and structures is not None:
        if isinstance(structures, list):
            for i in range(len(structures)):
                structures[i] = structures[i].reshape(-1, 3)[subset_idx]
        else:
            structures = structures.reshape(-1, 3)[subset_idx]

    data_subset = {'counts': counts, 'bias': bias, 'struct': structures}
    return lengths_subset, chrom_subset, data_subset


def _intramol_mask(data, lengths_at_res):
    """Get mask of intra-molecular row/col for given counts/distance data."""

    if (isinstance(data, (tuple, list)) and len(data) == 2) or (
            isinstance(data, np.ndarray) and data.size == 2):
        shape = data
        row, col = (x.flatten() for x in np.indices(shape))
    else:
        row = data.row
        col = data.col
        shape = data.shape

    n = lengths_at_res.sum()
    if set(shape) not in ({n}, {n * 2}, {n, n * 2}):
        raise ValueError(
            f"Counts matrix shape {shape} is not consistent with"
            f" number of beads ({lengths_at_res.sum()=}).")

    # If only one chromosome, all bins are intra
    if lengths_at_res.size == 1 and set(shape) == {n}:
        return np.full(row.size, True)

    bins_for_row = np.tile(
        lengths_at_res, int(shape[0] / lengths_at_res.sum())).cumsum()
    bins_for_col = np.tile(
        lengths_at_res, int(shape[1] / lengths_at_res.sum())).cumsum()
    row_binned = np.digitize(row, bins_for_row)
    col_binned = np.digitize(col, bins_for_col)

    if shape[0] != shape[1]:
        nchrom = lengths_at_res.shape[0]
        row_binned[row_binned >= nchrom] -= nchrom
        col_binned[col_binned >= nchrom] -= nchrom

    return np.equal(row_binned, col_binned)


def _get_counts_sections(counts, sections, lengths_at_res, ploidy, nbins=None):
    """Set counts outside the given section to zero."""  # TODO move to counts.py
    from .counts import _check_counts_matrix

    sections = sections.lower()
    options = ['inter', 'intra', 'near diag']
    if sections not in options:
        raise ValueError(f"Options: {', '.join(options)}.")
    if sections == 'near diag' and nbins is None:
        raise ValueError("Must input nbins.")

    counts = _check_counts_matrix(counts, lengths=lengths_at_res, ploidy=ploidy)

    mask_intra = _intramol_mask(counts, lengths_at_res=lengths_at_res)
    if sections == 'intra':
        mask = mask_intra
    elif sections == 'inter':
        mask = ~mask_intra
    elif sections == 'near diag':
        row = counts.row
        col = counts.col
        if counts.shape[0] != counts.shape[1]:
            row = row.copy()
            col = col.copy()
            n = lengths_at_res.sum()
            row[row >= n] -= n
            col[col >= n] -= n
        mask_near_diag = np.abs(row - col) <= nbins
        mask = mask_intra & mask_near_diag
    # elif sections == 'lowres nghbr':
    #     row = counts.row; col = counts.col
    #     row = row.copy()
    #     col = col.copy()
    #     if counts.shape[0] != counts.shape[1]:
    #         n = lengths_at_res.sum()
    #         row[row >= n] -= n
    #         col[col >= n] -= n
    #     lengths_tiled = np.tile(lengths_at_res, ploidy)
    #     for i in np.flip(np.arange(1, lengths_tiled.size)):
    #         l = lengths_tiled[:i].sum()
    #         row[row >= l] -= lengths_tiled[i - 1]
    #         col[col >= l] -= lengths_tiled[i - 1]
    #     mask_lowres_nghbr = np.abs(np.floor(row / nbins) - np.floor(
    #         col / nbins)) == 1
    #     mask = mask_intra & mask_lowres_nghbr

    return sparse.coo_matrix(
        (counts.data[mask], (counts.row[mask], counts.col[mask])),
        shape=counts.shape)


def _intramol_counts(counts, lengths_at_res, ploidy):
    """Return intra-molecular counts."""  # TODO move to counts.py
    return _get_counts_sections(
        counts=counts, sections='intra', lengths_at_res=lengths_at_res,
        ploidy=ploidy)


def _intermol_counts(counts, lengths_at_res, ploidy):
    """Return inter-molecular counts."""  # TODO move to counts.py
    return _get_counts_sections(
        counts=counts, sections='inter', lengths_at_res=lengths_at_res,
        ploidy=ploidy)


def _counts_near_diag(counts, lengths_at_res, ploidy, nbins):
    """Return intra-molecular counts within nbins of diagonal."""  # TODO move to counts.py
    return _get_counts_sections(
        counts=counts, sections='near diag', lengths_at_res=lengths_at_res,
        ploidy=ploidy, nbins=nbins)
