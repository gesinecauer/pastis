import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import numpy as np
from scipy import sparse
import warnings
import re
from copy import deepcopy
import pandas as pd

from iced.filter import filter_low_counts
from iced.normalization import ICE_normalization

from .utils_poisson import _setup_jax
_setup_jax()
import jax.numpy as jnp

from .utils_poisson import find_beads_to_remove
from .utils_poisson import _intramol_counts, _intermol_counts, _counts_near_diag
from .utils_poisson import _dict_is_equal, _dict_to_hash
from .multiscale_optimization import decrease_lengths_res
from .multiscale_optimization import _group_counts_multiscale
from .multiscale_optimization import _get_fullres_counts_index
from .multiscale_optimization import _count_fullres_per_lowres_bead


def _best_counts_dtype(counts):
    """Choose most memory-efficient dtype for counts matrix"""

    if sparse.issparse(counts):
        data = counts.data
    else:
        data = np.asarray(counts)

    if not (np.issubdtype(data.dtype, np.floating) or np.issubdtype(
            data.dtype, np.integer)):
        raise ValueError(f"Counts dtype is {data.dtype}, must be float or int.")
    if (~np.isfinite(data)).sum() > 0:
        raise ValueError(f"Counts may not contain {data.sum()}.")
    if np.any(data < 0):
        raise ValueError("Counts must not be < 0.")

    max_val = data.max()
    if np.issubdtype(data.dtype, np.floating):
        if np.array_equal(data, data.round()):
            max_val = int(max_val)
        else:
            warnings.warn("Counts matrix should only contain integers.")
            max_val = data.sum()  # Otherwise, sum can cause overflow to inf
            return np.promote_types(np.min_scalar_type(max_val), np.float64)

    return np.min_scalar_type(max_val)


def ambiguate_counts(counts, lengths, ploidy):
    """Convert diploid counts to ambiguous & aggregate counts across matrices.

    If diploid, convert unambiguous and partially ambiguous counts to ambiguous
    and aggregate list of counts into a single counts matrix. If haploid,
    check format of and return the inputted counts matrix.

    Parameters
    ----------
    counts : list containing: array, coo_matrix, or CountsMatrix instances
        Counts data.
    lengths : array_like of int
        Number of beads per homolog of each chromosome.

    Returns
    -------
    coo_matrix or ndarray
        Aggregated and ambiguated contact counts matrix.
    """

    lengths = np.array(lengths, copy=False, ndmin=1, dtype=int).ravel()
    n = lengths.sum()
    if not isinstance(counts, (list, tuple)):
        counts = [counts]

    if all([isinstance(c, CountsMatrix) for c in counts]):
        counts_ambig = sum(counts).ambiguate()
        return counts_ambig

    counts = check_counts(counts, lengths=lengths, ploidy=ploidy)

    if len(counts) == 1 and (ploidy == 1 or counts[0].shape == (n, n)):
        return counts[0]

    counts_ambig = sparse.csr_matrix(np.zeros((n, n)))
    for i in range(len(counts)):
        counts[i] = counts[i].tocsr()
        if set(counts[i].shape) == {n}:
            counts_ambig += counts[i]
        elif set(counts[i].shape) == {ploidy * n}:
            tmp = counts[i][:n, :] + counts[i][n:, :]
            tmp = tmp.tocsc()
            tmp = tmp[:, :n] + tmp[:, n:]
            counts_ambig += sparse.triu(tmp + tmp.T, k=1)
        else:
            tmp = counts[i][:n, :] + counts[i][n:, :]
            counts_ambig += sparse.triu(tmp + tmp.T, k=1)
        counts[i] = counts[i].tocoo()

    counts_ambig.sort_indices()

    return _check_counts_matrix(
        counts_ambig.tocoo(), lengths=lengths, ploidy=ploidy)


def _ambiguate_beta(beta, counts, lengths, ploidy):
    """Sum betas to be consistent with ambiguated counts.
    """

    if beta is None:
        return None

    if not isinstance(beta, (list, tuple, np.ndarray, jnp.ndarray)):
        beta = [beta]
    if ploidy == 1:
        return beta[0]

    if not isinstance(counts, (list, tuple)):
        counts = [counts]
    if len(counts) != len(beta):
        raise ValueError(f"Inconsistent number of betas ({len(beta)}) and"
                         f" counts matrices ({len(counts)}).")

    beta_ambig = 0
    for i in range(len(beta)):
        if counts[i].shape[0] == counts[i].shape[1]:
            beta_ambig += beta[i]
        else:  # Adjust for partially ambiguous: multiply by 2
            beta_ambig += beta[i] * 2
    return beta_ambig


def _disambiguate_beta(beta_ambig, counts, lengths, ploidy, bias=None):
    """Derive beta for each counts matrix from ambiguated beta.
    """
    if beta_ambig is None:
        return None
    if ploidy == 1:
        return [beta_ambig]

    if bias is not None and bias.size != lengths.sum():
        raise ValueError("Bias size must be equal to the sum of the chromosome "
                         f"lengths ({lengths.sum()}). It is of size"
                         f" {bias.size}.")

    if not isinstance(counts, (list, tuple)):
        counts = [counts]

    # Get sum of each (normalized) counts matrix
    counts_sums = np.zeros(len(counts))
    for i in range(len(counts)):
        if counts[i].sum() == 0:
            continue
        counts_ambig_i = ambiguate_counts(
            counts[i], lengths=lengths, ploidy=ploidy)
        if bias is not None and not np.all(bias == 1):  # Divide by locus biases
            bias_per_bin = bias[counts_ambig_i.row] * bias[counts_ambig_i.col]
            counts_sums[i] = np.sum(counts_ambig_i.data / bias_per_bin)
        else:
            counts_sums[i] = counts_ambig_i.sum()

    beta = counts_sums / counts_sums.sum() * beta_ambig

    # Adjust beta for partially ambiguous: divide by 2
    for i in range(len(counts)):
        if counts[i].shape[0] != counts[i].shape[1]:
            beta[i] /= 2

    beta = beta.tolist()
    return beta


def _get_included_counts_bins(counts, lengths, ploidy, check_counts=True,
                              remove_diag=True, exclude_zeros=False):
    """TODO
    """
    if check_counts:
        counts = _check_counts_matrix(
            counts, lengths=lengths, ploidy=ploidy, remove_diag=remove_diag)

    if exclude_zeros:
        return sparse.coo_matrix(
            (np.ones(counts.data.size, dtype=int), (counts.row, counts.col)),
            shape=counts.shape).toarray().astype(bool)

    n = lengths.sum()
    included = np.ones(counts.shape, dtype=np.uint8)

    # Remove values on/below diagonal, as appropriate
    if counts.shape[0] == counts.shape[1]:
        if remove_diag:
            included = np.triu(included, k=1)
        else:
            included = np.triu(included, k=0)
    elif remove_diag:
        np.fill_diagonal(included[:n, :], 0)
        np.fill_diagonal(included[n:, :], 0)

    # Remove empty loci
    struct_nan = find_beads_to_remove(counts, lengths=lengths, ploidy=ploidy)
    included[struct_nan[struct_nan < included.shape[0]], :] = 0
    included[:, struct_nan[struct_nan < included.shape[1]]] = 0

    return included.astype(bool)


def _check_counts_matrix(counts, lengths, ploidy, chrom_subset_idx=None,
                         remove_diag=True, copy=False):
    """Check counts dimensions, reformat, & excise selected chromosomes.
    """

    lengths = np.array(lengths, copy=False, ndmin=1, dtype=int).ravel()
    n = lengths.sum()

    # Check input
    if not (isinstance(counts, np.ndarray) or sparse.issparse(counts)):
        raise ValueError("Counts must be ndarray or sparse.coo_matrix.")
    if counts.ndim != 2:
        raise ValueError(
            f"Counts must be 2D matrix, current shape = {counts.shape}.")
    if set(counts.shape) not in ({n}, {n * ploidy}, {n, n * ploidy}):
        raise ValueError(
            f"Counts matrix shape {counts.shape} is not consistent with"
            f" number of beads ({lengths.sum() * ploidy}).")

    if copy:
        counts = counts.copy()

    # Get rid of NaNs and Inf, select best dtype, convert to sparse coo matrix
    if isinstance(counts, np.ndarray):
        counts[~np.isfinite(counts)] = 0
        counts = sparse.coo_matrix(
            counts, dtype=_best_counts_dtype(counts))
    else:  # Is a sparse matrix
        counts = sparse.coo_matrix(counts)
        mask = np.isfinite(counts.data)
        if np.invert(mask).sum() > 0:
            counts = sparse.coo_matrix(
                (counts.data[mask], (counts.row[mask], counts.col[mask])),
                shape=counts.shape,
                dtype=_best_counts_dtype(counts.data[mask]))
        else:
            counts = counts.astype(_best_counts_dtype(counts))

    # Remove values on/below diagonal, as appropriate
    if counts.shape[0] == counts.shape[1]:
        if remove_diag:
            counts = sparse.triu(counts, k=1)
        else:
            counts = sparse.triu(counts, k=0)
    else:
        if counts.shape[0] == n:  # Hmlgs were horizontally concat: (n, 2n)
            counts = counts.T
        counts = counts.tocsr()
        hmlg1 = counts[:n, :].tocoo()
        hmlg2 = counts[n:, :].tocoo()
        if remove_diag:
            hmlg1.setdiag(0)
            hmlg2.setdiag(0)
            hmlg1.eliminate_zeros()
            hmlg2.eliminate_zeros()
        counts = sparse.vstack([hmlg1, hmlg2])  # Vertical concat: (2n, n)

    if chrom_subset_idx is not None:
        counts = counts.tocsr()[
            chrom_subset_idx[chrom_subset_idx < counts.shape[0]], :]
        counts = counts.tocsc()[
            :, chrom_subset_idx[chrom_subset_idx < counts.shape[1]]].tocoo()

    counts = counts.tocsr()
    counts.sort_indices()
    counts = counts.tocoo()
    counts.eliminate_zeros()

    return counts


def check_counts(counts, lengths, ploidy, chrom_subset_idx=None):
    """Check counts dimensions and reformat data.

    Check dimensions of each counts matrix, exclude appropriate values,
    and (if applicable) make sure partially ambiguous diploid counts are
    vertically oriented (one matrix above the other).

    Parameters
    ----------
    counts : list of array or coo_matrix
        Counts data.
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.

    Returns
    -------
    counts : list of coo_matrix
        Checked and reformatted counts data.
    """

    lengths = np.array(lengths, copy=False, ndmin=1, dtype=int).ravel()
    if not isinstance(counts, (list, tuple)):
        counts = [counts]

    # Check that there aren't multiple counts matrices of the same type
    ambiguities = [_get_counts_ambiguity(
        c.shape, lengths.sum() * ploidy) for c in counts]
    if len(ambiguities) != len(set(ambiguities)):
        raise ValueError("Can't input multiple counts matrices of the same"
                         f" type. Inputs: {', '.join(map(str, ambiguities))}")

    return [_check_counts_matrix(
        c, lengths=lengths, ploidy=ploidy,
        chrom_subset_idx=chrom_subset_idx) for c in counts]


def preprocess_counts(counts, lengths, ploidy, multiscale_factor=1,
                      beta=None, bias=None, excluded_counts=None,
                      exclude_zeros=False, multiscale_reform=True,
                      input_weight=None, verbose=True, mods=[]):
    """Check counts, reformat.

    Counts are also checked and reformatted
    for inference. Final matrices are stored as CountsMatrix subclass instances.

    Parameters
    ----------
    counts : list of coo_matrix
        Counts data without normalization or filtering.
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
    beta : array_like of float, optional
        Scaling parameter that determines the size of the structure, relative to
        each counts matrix. There should be one beta per counts matrix. If None,
        the optimal beta will be estimated.
    excluded_counts : {"inter", "intra"}, optional
        Whether to exclude inter- or intra-chromosomal counts from optimization. # TODO update

    Returns
    -------
    counts : list of CountsMatrix subclass instances
        Preprocessed counts data.
    bias : array of float
        Hi-C biases
    struct_nan : array of int
        Beads that should be removed (set to NaN) in the structure.
    """

    # Beads to remove from full-res structure
    if multiscale_factor == 1:
        fullres_struct_nan = None
    else:
        fullres_struct_nan = find_beads_to_remove(
            counts, lengths=lengths, ploidy=ploidy, multiscale_factor=1)

    # Format counts as CountsMatrix objects
    counts = _format_counts(
        counts, beta=beta, bias=bias, input_weight=input_weight,
        lengths=lengths, ploidy=ploidy, exclude_zeros=exclude_zeros,
        multiscale_factor=multiscale_factor,
        multiscale_reform=multiscale_reform)

    # Identify beads to be removed from the final structure
    struct_nan = find_beads_to_remove(
        counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor)

    # Optionally exclude certain counts
    if excluded_counts is not None:
        if isinstance(excluded_counts, float):
            if excluded_counts != int(excluded_counts):
                raise ValueError("excluded_counts must be an integer.")
            excluded_counts = int(excluded_counts)
        if isinstance(excluded_counts, str) and re.match(
                r'[0-9]+(\.0){0,1}', excluded_counts):
            excluded_counts = int(excluded_counts)

        lengths_lowres = decrease_lengths_res(
            lengths, multiscale_factor=multiscale_factor)

        if ploidy == 1 or any([c.ambiguity == 'ua' for c in counts]):
                raise NotImplementedError

        if isinstance(excluded_counts, int):
            counts = [_counts_near_diag(
                c, lengths_at_res=lengths_lowres, ploidy=ploidy,
                nbins=excluded_counts, copy=False) for c in counts]
        elif excluded_counts.lower() == 'intra':
            counts = [_intermol_counts(
                c, lengths_at_res=lengths_lowres, ploidy=ploidy,
                copy=False) for c in counts]
        elif excluded_counts.lower() == 'inter':
            counts = [_intramol_counts(
                c, lengths_at_res=lengths_lowres, ploidy=ploidy,
                copy=False) for c in counts]
            # n = lengths_lowres.sum()
            # print(counts[0].tocoo().toarray()[:n, n:].sum()); exit(0)
        else:
            raise ValueError(
                "excluded_counts must be an integer, 'inter', 'intra' or None.")

    return counts, struct_nan, fullres_struct_nan


def _get_counts_ambiguity(shape, nbeads):
    """TODO"""
    return {1: 'ambig', 1.5: 'pa', 2: 'ua'}[sum(shape) / nbeads]


def _prep_counts(counts, lengths, ploidy, filter_threshold=0.04, normalize=True,
                 bias=None, verbose=True):
    """Check counts, filter, and compute bias.
    """

    counts = check_counts(counts, lengths=lengths, ploidy=ploidy)

    if (filter_threshold is None or filter_threshold == 0) and (
            not normalize) and (bias is None):
        return counts, bias

    # Optionally filter counts
    if filter_threshold is not None and filter_threshold > 0:
        # If there are multiple counts matrices, filter them together.
        # Counts will be ambiguated for deciding which beads to remove.
        # For diploid, any beads that are filtered out will be removed from both
        # homologs.
        if verbose:
            print("FILTERING LOW COUNTS: manually filtering counts"
                  f" by {filter_threshold * 100:g}%", flush=True)

        # Ambiguate counts
        counts_ambig = ambiguate_counts(counts, lengths=lengths, ploidy=ploidy)

        # How many beads are initially zero?
        num0_initial = find_beads_to_remove(
            counts_ambig, lengths=lengths, ploidy=1).size
        perc0_initial = num0_initial / lengths.sum()

        # Filter ambiguated counts, and get mask of beads that are NaN
        counts_ambig = filter_low_counts(
            counts_ambig, sparsity=False,
            percentage=filter_threshold + perc0_initial).tocoo()
        if verbose:
            if num0_initial > 0:
                print(f"{' ' * 22}{num0_initial} loci (per homolog) are",
                      " already empty", flush=True)
            num0_final = find_beads_to_remove(
                counts_ambig, lengths=lengths, ploidy=1).size
            print(f"{' ' * 22}Removed {num0_final - num0_initial} loci (per",
                  " homolog)", flush=True)

        # Remove the NaN beads from all counts matrices
        for i in range(len(counts)):
            struct_nan = find_beads_to_remove(
                counts_ambig, lengths=lengths, ploidy=ploidy)
            mask = ~np.isin(counts[i].row, struct_nan) & ~np.isin(
                counts[i].col, struct_nan)
            counts[i] = sparse.coo_matrix(
                (counts[i].data[mask],
                    (counts[i].row[mask], counts[i].col[mask])),
                shape=counts[i].shape)

    # Optionally normalize counts
    # Normalization is done on ambiguated counts
    if normalize and bias is None:
        if verbose:
            print('COMPUTING HI-C BIASES', flush=True)
        counts_ambig = ambiguate_counts(counts, lengths=lengths, ploidy=ploidy)
        bias = ICE_normalization(
            counts_ambig, max_iter=300, output_bias=True)[1].flatten()
    elif bias is not None:
        if verbose:
            print('USING USER-PROVIDED HI-C BIAS VECTOR', flush=True)
        if bias.size != lengths.sum():
            raise ValueError("Bias size must be equal to the sum of the"
                             f" chromosome lengths ({lengths.sum()}). It is of"
                             f" size {bias.size}.")

    # In each counts matrix, zero out counts for which bias is NaN
    if bias is not None:
        for i in range(len(counts)):
            # How many beads are initially zero?
            counts_ambig_i = ambiguate_counts(
                counts[i], lengths=lengths, ploidy=ploidy)
            num0_initial = find_beads_to_remove(
                counts_ambig_i, lengths=lengths, ploidy=1).size

            # Set zeros in bias vector to NaN
            bias[bias == 0] = np.nan

            # Remove beads with bias=NaN from counts
            bias_is_finite = np.where(np.isfinite(np.tile(bias, ploidy)))[0]
            mask = np.isin(counts[i].row, bias_is_finite) & np.isin(
                counts[i].col, bias_is_finite)
            counts[i] = sparse.coo_matrix(
                (counts[i].data[mask],
                    (counts[i].row[mask], counts[i].col[mask])),
                shape=counts[i].shape)

            if verbose:
                num0_final = find_beads_to_remove(
                    counts_ambig_i, lengths=lengths, ploidy=1).size
                if num0_final - num0_initial > 0:
                    ambiguity = _get_counts_ambiguity(
                        counts[i].shape, nbeads=lengths.sum() * ploidy)
                    print(f"{' ' * 22}Removing {num0_final - num0_initial}"
                          f" additional loci (per homolog) from {ambiguity}",
                          " counts", flush=True)

    counts = check_counts(counts, lengths=lengths, ploidy=ploidy)
    return counts, bias


def _set_initial_beta(counts, lengths, ploidy, bias=None, exclude_zeros=False,
                      neighboring_beads_only=True):
    """Estimate compatible betas for each counts matrix.

    Set the mean (distance ** alpha) using either
      (1) distances between all bead pairs (if neighboring_beads_only=False), or
      (2) only the distances between neighboring beads (default)
    ...to be equal to 1
    """

    from .constraints import _neighboring_bead_indices

    if bias is not None and bias.size != lengths.sum():
        raise ValueError("Bias size must be equal to the sum of the chromosome "
                         f"lengths ({lengths.sum()}). It is of size"
                         f" {bias.size}.")

    # Get relevant counts
    counts_ambig = ambiguate_counts(counts, lengths=lengths, ploidy=ploidy)
    if neighboring_beads_only:
        row_nghbr = _neighboring_bead_indices(
            lengths=lengths, ploidy=1, multiscale_factor=1,
            counts=counts_ambig, include_struct_nan_beads=False)
        mask = np.isin(counts_ambig.row, row_nghbr) & (
            counts_ambig.col == counts_ambig.row + 1)
        counts_ambig = sparse.coo_matrix(
            (counts_ambig.data[mask],
                (counts_ambig.row[mask], counts_ambig.col[mask])),
            shape=counts_ambig.shape)

    # Get indices of distance bins associated with relevant counts
    if neighboring_beads_only:
        # We assume that the contribution of inter-hmlg counts to nghbr counts
        # is negligible
        if exclude_zeros:
            row3d = np.tile(counts_ambig.row, ploidy) + np.repeat(
                np.arange(ploidy) * lengths.sum(), counts_ambig.row.size)
        else:
            row3d = np.tile(row_nghbr, ploidy) + np.repeat(
                np.arange(ploidy) * lengths.sum(), row_nghbr.size)
        col3d = row3d + 1
    else:
        row3d, col3d = _counts_indices_to_3d_indices(
            counts_ambig, lengths_at_res=lengths, ploidy=ploidy,
            exclude_zeros=exclude_zeros)
    if row3d.size == 0:
        raise ValueError("Cannot compute beta -- no counts bins are included.")

    # Get denominator: sum of normalized distance bins
    if bias is not None and not np.all(bias == 1):
        bias_tiled = np.tile(bias.ravel(), ploidy)
        denominator = (bias_tiled[col3d] * bias_tiled[col3d]).sum()
    else:
        denominator = row3d.size

    # Get universal/ambiguated beta
    beta_ambig = counts_ambig.sum() / denominator
    if (not np.isfinite(beta_ambig)) or beta_ambig <= 0:
        raise ValueError(f"Beta for ambiguated counts is {beta_ambig}.")

    # Assign separate betas to each counts matrix
    beta = _disambiguate_beta(
        beta_ambig, counts=counts, lengths=lengths, ploidy=ploidy, bias=bias)

    return beta_ambig, beta


def _format_counts(counts, lengths, ploidy, beta=None, bias=None,
                   input_weight=None, exclude_zeros=False, multiscale_factor=1,
                   multiscale_reform=True):
    """Format each counts matrix as a CountsMatrix subclass instance.
    """

    # Check input
    if not isinstance(counts, (tuple, list)):
        counts = [counts]
    if input_weight is None:
        input_weight = [1] * len(counts)
    else:
        if not isinstance(input_weight, (list, tuple, np.ndarray)):
            input_weight = [input_weight]
        if len(input_weight) != len(counts):
            raise ValueError("input_weights needs to contain as many values"
                             f" as there are counts matrices ({len(counts)})."
                             f" It is of size {len(input_weight)}.")
        input_weight = np.array(input_weight)
        if input_weight.sum() not in (0, 1):
            input_weight *= len(input_weight) / input_weight.sum()
    if beta is not None:
        if not isinstance(beta, (list, tuple, np.ndarray)):
            beta = [beta]
        if len(beta) != len(counts):
            raise ValueError(
                "Beta needs to contain as many values as there are counts"
                f" matrices ({len(counts)}). It is of size {len(beta)}.")

    # If counts are already formatted as CountsMatrix instances
    if all([isinstance(c, CountsMatrix) for c in counts]):
        if any([c.multiscale_factor != multiscale_factor for c in counts]):
            raise ValueError(f"CountsMatrix does not match {multiscale_factor=}.")
        if any([c.multires_naive != (not multiscale_reform) for c in counts]):
            raise ValueError(f"CountsMatrix does not match {multiscale_reform=}.")
        if any([c.exclude_zeros != exclude_zeros for c in counts]):
            raise ValueError(f"CountsMatrix does not match {exclude_zeros=}.")
        if any([c.ploidy != ploidy for c in counts]):
            raise ValueError(f"CountsMatrix does not match {ploidy=}.")
        if any([not np.array_equal(c.lengths, lengths) for c in counts]):
            raise ValueError(f"CountsMatrix does not match {lengths=}.")
        if beta is not None:
            counts = [counts[i].update_beta(beta[i]) for i in range(len(beta))]
        return tuple(counts)

    # Prepare for formatting as CountsMatrix instance
    counts = check_counts(counts, lengths=lengths, ploidy=ploidy)
    if beta is None:
        _, beta = _set_initial_beta(
            counts, lengths=lengths, ploidy=ploidy, bias=bias,
            exclude_zeros=exclude_zeros)

    # Reformat counts as CountsMatrix instance
    counts_reformatted = []
    for i in range(len(counts)):
        counts_matrix = CountsMatrix(
            lengths=lengths, ploidy=ploidy, counts=counts[i],
            multiscale_factor=multiscale_factor, beta=beta[i],
            weight=input_weight[i], exclude_zeros=exclude_zeros,
            multires_naive=not multiscale_reform)
        counts_reformatted.append(counts_matrix)

    return tuple(counts_reformatted)


def _counts_indices_to_3d_indices(data, lengths_at_res, ploidy,
                                  exclude_zeros=None):
    """Return distance matrix indices associated with counts matrix data.
    """

    if isinstance(data, tuple) and len(data) == 3:
        row3d, col3d, shape = data
    else:
        shape = data.shape
        if exclude_zeros is None:
            raise ValueError("Must input exclude_zeros")
        elif exclude_zeros:
            row3d = data.row
            col3d = data.col
        else:
            included = _get_included_counts_bins(
                data, lengths=lengths_at_res, ploidy=ploidy)
            row3d, col3d = np.where(included)

    if row3d.size == 0:
        return np.array([]), np.array([])

    nbeads_lowres = int(lengths_at_res.sum() * ploidy)
    if shape[0] != nbeads_lowres or shape[1] != nbeads_lowres:
        nnz = len(row3d)

        map_factor_rows = int(nbeads_lowres / shape[0])
        map_factor_cols = int(nbeads_lowres / shape[1])
        map_factor = map_factor_rows * map_factor_cols

        x, y = np.indices((map_factor_rows, map_factor_cols))
        x = x.flatten()
        y = y.flatten()

        row3d = np.repeat(
            x, int(nnz * map_factor / x.shape[0])) * min(shape) + np.tile(
            row3d, map_factor)
        col3d = np.repeat(
            y, int(nnz * map_factor / y.shape[0])) * min(shape) + np.tile(
            col3d, map_factor)

    row3d = np.asarray(row3d, dtype=np.min_scalar_type(row3d.max()), order='C')
    col3d = np.asarray(col3d, dtype=np.min_scalar_type(col3d.max()), order='C')

    return row3d, col3d


def _get_bias_per_bin(ploidy, bias, row, col, multiscale_factor=1, lengths=None,
                      multires_naive=False):
    """Determines bias corresponding to each bin of the distance matrix."""
    if bias is None or (isinstance(bias, np.ndarray) and np.all(bias == 1)):
        return None

    if lengths is not None:
        if multiscale_factor > 1 and multires_naive:
            lengths_lowres = decrease_lengths_res(
                lengths, multiscale_factor=multiscale_factor)
            if bias.size != lengths_lowres.sum():
                raise ValueError("Bias size must be equal to the sum of low-res"
                                 f" chromosome lengths ({lengths_lowres.sum()})."
                                 f" It is of size {bias.size}.")
        elif bias.size != lengths.sum():
            raise ValueError("Bias size must be equal to the sum of the"
                             f" chromosome lengths ({lengths.sum()}). It is of"
                             f" size {bias.size}.")

    # Output type should be the same as type(bias)
    if isinstance(bias, jnp.ndarray):
        tmp_np = jnp
    else:
        tmp_np = np

    bias = tmp_np.tile(bias.ravel(), ploidy)
    if multiscale_factor == 1 or multires_naive:
        return bias[row] * bias[col]
    else:
        (row_fullres, col_fullres, bad_idx), _ = _get_fullres_counts_index(
            multiscale_factor=multiscale_factor, lengths=lengths,
            ploidy=ploidy, lowres_idx=(row, col))
        bias_per_bin = bias[row_fullres] * bias[col_fullres]
        bias_per_bin = tmp_np.where(
            bad_idx | (~tmp_np.isfinite(bias_per_bin)), 0, bias_per_bin)
        bias_per_bin = bias_per_bin.reshape(multiscale_factor ** 2, -1)
        return bias_per_bin


# _isin_2d_good = np.vectorize(
#     lambda arr1, arr2: np.any((arr2[:, 0] == arr1[0]) & (arr2[:, 1] == arr1[1])),
#     signature='(n),(m,n)->()',
#     doc='A 2-dimensional version of numpy.isin')


# def _idx_isin_good(idx1, idx2):
#     """Whether each (row, col) pair in idx1 (row1, col1) is in idx2 (row2, col2)
#     """

#     if isinstance(idx1, (list, tuple)):
#         idx1 = np.stack(idx1, axis=1)
#     if isinstance(idx2, (list, tuple)):
#         idx2 = np.stack(idx2, axis=1)

#     mask2 = np.isin(idx2[:, 0], idx1[:, 0]) & np.isin(idx2[:, 1], idx1[:, 1])
#     # print(f"\n\tISIN {(~mask2).sum()=}/{mask2.size}", flush=True) # TODO remove
#     if mask2.sum() == 0:
#         return np.full(idx1.shape[0], False)
#     idx2 = idx2[mask2]

#     mask1 = np.isin(idx1[:, 0], idx2[:, 0]) & np.isin(idx1[:, 1], idx2[:, 1])
#     if mask1.sum() == 0:
#         return mask1
#     # print(f"\tISIN {(~mask1).sum()=}/{mask1.size}", flush=True) # TODO remove
#     mask1[mask1] = _isin_2d(idx1[mask1], idx2)
#     return mask1
#     # return _isin_2d(idx1, idx2) # TODO remove?


def _idx_isin(idx1, idx2):
    """Whether each (row, col) pair in idx1 (row1, col1) is in idx2 (row2, col2)
    """

    if isinstance(idx1, (list, tuple)):
        idx1 = np.stack(idx1, axis=1)
    type1 = np.min_scalar_type(idx1.max()).str
    idx1 = idx1.astype(type1)

    if isinstance(idx2, (list, tuple)):
        idx2 = np.stack(idx2, axis=1)
    if idx2.size == 0:
        return np.full(idx1.shape[0], False)
    type2 = np.min_scalar_type(idx2.max()).str
    idx2 = idx2.astype(type2)
    idx2 = np.unique(idx2, axis=0)

    mask2 = np.isin(idx2[:, 0], idx1[:, 0]) & np.isin(idx2[:, 1], idx1[:, 1])
    if mask2.sum() == 0:
        return np.full(idx1.shape[0], False)
    idx2 = idx2[mask2]

    mask1 = np.isin(idx1[:, 0], idx2[:, 0]) & np.isin(idx1[:, 1], idx2[:, 1])
    if mask1.sum() == 0:
        return mask1

    idx1 = np.array(list((map(tuple, idx1[mask1]))), dtype=f"{type1},{type1}")
    idx2 = np.array(list((map(tuple, idx2))), dtype=f"{type2},{type2}")

    mask1[mask1] = np.isin(idx1, idx2)
    return mask1


def _get_nonzero_mask(multiscale_factor, lengths, ploidy, row, col,
                      empty_idx_fullres=None):
    """At low res: create mask for counts bins that are included in inference.

    Used to create .bins_nonzero.mask attribute for a CountsMatrix instance"""

    if multiscale_factor == 1:
        return None

    idx_fullres_all, idx_lowres_all = _get_fullres_counts_index(
        multiscale_factor=multiscale_factor, lengths=lengths, ploidy=ploidy,
        counts_fullres_shape=(lengths.sum(), lengths.sum()))

    # Only include low-res bins associated for relevant row & col
    idx_all_mask = _idx_isin(idx_lowres_all, (row, col))
    row_fullres, col_fullres, bad_idx_fullres = [x.reshape(
        multiscale_factor ** 2, -1)[:, idx_all_mask] for x in idx_fullres_all]

    # mask is False for full-res loci that have no data in any row/col
    nonzero_mask = ~bad_idx_fullres
    if empty_idx_fullres is not None:
        nonzero_mask = nonzero_mask & ~np.isin(
            row_fullres, empty_idx_fullres) & ~np.isin(
            col_fullres, empty_idx_fullres)

    if np.all(nonzero_mask):
        nonzero_mask = None

    return nonzero_mask


class CountsMatrix(object):
    """Stores counts data, indices, beta, weight, distance matrix indices, etc.

    Counts data and information associated with this counts matrix.

    Parameters
    ----------
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    beta : float
        Scaling parameter that determines the size of the structure, relative to
        each counts matrix.
    counts : coo_matrix
        Preprocessed counts data.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
    multires_naive : bool, optional
        Whether to apply the naive multi-resolution model.

    Attributes
    ----------
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    multiscale_factor : int
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
    multires_naive : bool
        Whether to apply the naive multi-resolution model.
    null : bool
        Whether the counts data should be excluded from the primary objective
        function. Counts are still used in the calculation of the constraints.
    beta : float
        Scaling parameter that determines the size of the structure, relative to
        each counts matrix.
    ambiguity : {"ambig", "pa", "ua"}
        The ambiguity level of the counts data. "ambig" indicates ambiguous,
        "pa" indicates partially ambiguous, and "ua" indicates unambiguous
        or haploid.
    shape : tuple of int
        Shape of the matrix.
    bins_nonzero : CountsBins instance
        Data for counts bins with >0 counts
    bins_zero : CountsBins instance or None
        Data for counts bins with 0 counts
    bins : list
        Contains bins_nonzero and bins_zero
    """

    def __init__(self, lengths, ploidy, beta, counts=None, multiscale_factor=1,
                 weight=1, exclude_zeros=False, multires_naive=False):
        self.lengths = lengths
        self.ploidy = ploidy
        self.multiscale_factor = multiscale_factor
        self.multires_naive = multires_naive
        self.null = False  # If True, exclude counts data from primary obj
        self.beta = float(beta)
        self.weight = (1. if weight is None else float(weight))
        if np.isnan(self.weight) or np.isinf(self.weight) or self.weight == 0:
            raise ValueError(f"Counts weight may not be {self.weight}.")

        self.ambiguity = None
        self.shape = None
        self._empty_idx_fullres = None  # Full-res loci without any data
        self.bins_nonzero = None
        self.bins_zero = None
        self.exclude_zeros = None

        if counts is not None:
            self.ambiguity = _get_counts_ambiguity(
                counts.shape, nbeads=self.lengths.sum() * self.ploidy)

            if self.multiscale_factor > 1:
                self._empty_idx_fullres = find_beads_to_remove(
                    counts, lengths=self.lengths, ploidy=self.ploidy)

            data, (row, col), self.shape, mask = _group_counts_multiscale(
                counts, lengths=self.lengths, ploidy=self.ploidy,
                multiscale_factor=self.multiscale_factor,
                exclude_zeros=exclude_zeros,
                multiscale_reform=not multires_naive)
            self.bins_nonzero = CountsBins(
                meta=self, row=row, col=col, data=data, mask=mask)

            self.exclude_zeros = exclude_zeros
            if not exclude_zeros:
                self._create_bins_zero()

    @property
    def bins(self):
        """List containing all available CountsBins instances."""
        bins = []
        if self.bins_nonzero is not None:
            bins.append(self.bins_nonzero)
        if self.bins_zero is not None:
            bins.append(self.bins_zero)
        return bins

    @property
    def nbins(self):
        """Number of bins at the current resolution."""
        return sum([x.nbins for x in self.bins])

    def filter(self, row, col, copy=True):
        """Filter data for the given indices."""
        if copy:
            filtered = self.copy()
        else:
            filtered = self

        if self.bins_nonzero is None and self.bins_zero is None:
            return filtered
        if self.bins_nonzero is not None:
            filtered.bins_nonzero = filtered.bins_nonzero.filter(
                row=row, col=col, copy=False)
        if self.bins_zero is not None:
            filtered.bins_zero = filtered.bins_zero.filter(
                row=row, col=col, copy=False)

        if self.bins_nonzero is None and self.bins_zero is None:
            raise ValueError("All counts bins have been filtered out.")
        return filtered

    def tocoo(self):
        """Convert counts matrix to scipy sparse COO format."""
        if self.bins_nonzero is None:
            return None
        data = self.bins_nonzero.data
        if len(data.shape) > 1:
            data = data.sum(axis=0)
        coo = sparse.coo_matrix(
            (data, (self.bins_nonzero.row, self.bins_nonzero.col)),
            shape=self.shape)
        return coo

    def sum(self, axis=None, dtype=None, out=None):
        """Sum of current counts matrix."""
        if self.bins_nonzero is None:
            return 0
        if (not isinstance(axis, int)) and (
                axis is None or set(list(axis)) == {0, 1}):
            return self.bins_nonzero._sum
        return self.tocoo().sum(axis=axis, dtype=dtype, out=out)

    def copy(self):  # XXX
        """Copy counts matrix."""
        return deepcopy(self)

    def update_beta(self, beta):
        """Update beta."""
        if isinstance(beta, dict):
            beta = beta[self.ambiguity]
        if isinstance(beta, jnp.ndarray):
            beta = beta._value
        if isinstance(beta, np.ndarray):
            if beta.size > 1:
                raise ValueError("Beta must be a float.")
            beta = float(beta)
        self.beta = beta
        if self.bins_nonzero is not None:
            self.bins_nonzero.beta = self.beta
        if self.bins_zero is not None:
            self.bins_zero.beta = self.beta
        return self

    def as_null(self):
        """
        Exclude the counts data from the primary objective function.

        Counts are still used in the calculation of the constraints.
        """
        self = self.ambiguate(copy=False)  # Null counts need not be phased
        self.null = True
        if self.bins_nonzero is not None:
            self.bins_nonzero.null = True
        if self.bins_zero is not None:
            self.bins_zero.null = True
        return self

    def _create_bins_zero(self):
        if self.bins_nonzero is None:
            raise ValueError(
                "Must create bins_nonzero before creating bins_zero.")

        _empty_idx_lowres = find_beads_to_remove(
            self, lengths=self.lengths, ploidy=self.ploidy,
            multiscale_factor=self.multiscale_factor)
        lengths_lowres = decrease_lengths_res(
            self.lengths, multiscale_factor=self.multiscale_factor)

        if self.multiscale_factor == 1 or self.multires_naive:
            zero_mask = None
            dummy = sparse.coo_matrix(_get_included_counts_bins(
                np.ones(self.shape, dtype=np.uint8), lengths=lengths_lowres,
                ploidy=self.ploidy, check_counts=False).astype(np.uint8))

            # Exclude loci that have no data in any row/col bin
            idx_all_mask = np.isin(
                dummy.row, _empty_idx_lowres, invert=True) & np.isin(
                dummy.col, _empty_idx_lowres, invert=True)

            # Exclude bins that have nonzero counts
            if idx_all_mask.sum() > 0:
                idx_all_mask[idx_all_mask] = ~_idx_isin(
                    (dummy.row[idx_all_mask], dummy.col[idx_all_mask]),
                    (self.bins_nonzero.row, self.bins_nonzero.col))

            zero_row = dummy.row[idx_all_mask]
            zero_col = dummy.col[idx_all_mask]
        else:
            idx_fullres_all, idx_lowres_all = _get_fullres_counts_index(
                multiscale_factor=self.multiscale_factor, lengths=self.lengths,
                ploidy=self.ploidy,
                counts_fullres_shape=(self.lengths.sum(), self.lengths.sum()))
            row_lowres_all, col_lowres_all = idx_lowres_all

            # Exclude low-res loci that have no data in any low-res row/col bin
            idx_all_mask = np.isin(
                row_lowres_all, _empty_idx_lowres, invert=True) & np.isin(
                col_lowres_all, _empty_idx_lowres, invert=True)

            # Exclude low-res bins that contain any nonzero counts
            if idx_all_mask.sum() > 0:
                idx_all_mask[idx_all_mask] = ~_idx_isin(
                    (row_lowres_all[idx_all_mask], col_lowres_all[idx_all_mask]),
                    (self.bins_nonzero.row, self.bins_nonzero.col))

            zero_row = row_lowres_all[idx_all_mask]
            zero_col = col_lowres_all[idx_all_mask]
            row_fullres, col_fullres, bad_idx_fullres = [x.reshape(
                self.multiscale_factor ** 2,
                -1)[:, idx_all_mask] for x in idx_fullres_all]

            # mask=False for full-res loci that have no data in any row/col
            zero_mask = ~bad_idx_fullres & ~np.isin(
                row_fullres, self._empty_idx_fullres) & ~np.isin(
                col_fullres, self._empty_idx_fullres)
            if np.all(zero_mask):
                zero_mask = None

        if zero_row.size == 0:
            self.bins_zero = None
        else:
            self.bins_zero = CountsBins(
                meta=self, row=zero_row, col=zero_col, mask=zero_mask)
        self.exclude_zeros = False

    def ambiguate(self, copy=True):
        """Convert diploid counts to ambiguous.

        If  unambiguous or partially ambiguous diploid, convert to ambiguous. If
        haploid or ambiguous diploid, return current counts matrix.

        Returns
        -------
        CountsMatrix object
            Ambiguated contact counts matrix.
        """

        if self.bins_nonzero is None and self.bins_zero is None:
            raise ValueError("Add data to CountsMatrix before ambiguating.")
        elif self.bins_nonzero is None:
            raise NotImplementedError(
                "CountsMatrix with only zero bins is not currently supported.")

        if self.ploidy == 1 or self.ambiguity == 'ambig':
            if copy:
                return self.copy()
            else:
                return self

        lengths_lowres = decrease_lengths_res(
            self.lengths, multiscale_factor=self.multiscale_factor)
        n = lengths_lowres.sum()

        if copy:
            ambig = CountsMatrix(
                lengths=self.lengths, ploidy=self.ploidy,
                multiscale_factor=self.multiscale_factor, beta=self.beta,
                weight=self.weight, multires_naive=self.multires_naive)
        else:
            ambig = self
        if self.ambiguity == 'pa':
            ambig.beta *= 2
        ambig.shape = (n, n)
        ambig.ambiguity = 'ambig'

        row = self.bins_nonzero.row.copy()
        col = self.bins_nonzero.col.copy()
        row[row >= n] -= n
        col[col >= n] -= n
        row_ambig = np.minimum(row, col)
        col_ambig = np.maximum(row, col)
        swap = row != row_ambig

        data = self.bins_nonzero.data.T
        if self.multiscale_factor > 1 and (
                not self.multires_naive) and swap.sum() > 0:
            data[swap] = data[swap].reshape(
                swap.sum(), self.multiscale_factor,
                self.multiscale_factor).reshape(swap.sum(), -1, order='f')
        data = pd.DataFrame(data)
        data['row'] = row_ambig
        data['col'] = col_ambig
        data = data.groupby(['row', 'col']).sum().reset_index()
        data = data[data.row != data.col]
        nonzero_data = data[[c for c in data.columns if c not in (
            'row', 'col')]].values.T
        if self.multiscale_factor == 1 or self.multires_naive:
            nonzero_data = nonzero_data.ravel()
        nonzero_row = data.row.values
        nonzero_col = data.col.values

        if self.multiscale_factor == 1 or self.multires_naive:
            nonzero_mask = None
        else:
            tmp_unique, tmp_counts = np.unique(
                np.append(self._empty_idx_fullres,
                          self._empty_idx_fullres - self.lengths.sum()),
                return_counts=True)
            _empty_idx_fullres = tmp_unique[tmp_counts == 2]
            ambig._empty_idx_fullres = np.append(
                _empty_idx_fullres, _empty_idx_fullres + self.lengths.sum())

            nonzero_mask = _get_nonzero_mask(
                multiscale_factor=self.multiscale_factor, lengths=self.lengths,
                ploidy=self.ploidy, row=data.row.values, col=data.col.values,
                empty_idx_fullres=ambig._empty_idx_fullres)

        ambig.bins_nonzero = CountsBins(
            meta=ambig, row=nonzero_row, col=nonzero_col, data=nonzero_data,
            mask=nonzero_mask)
        if not self.exclude_zeros:
            ambig._create_bins_zero()

        return ambig

    def __add__(self, other):
        if self.ploidy != other.ploidy:
            raise ValueError("Mismatch in ploidy")
        if self.multiscale_factor != other.multiscale_factor:
            raise ValueError("Mismatch in multiscale_factor")
        if not np.array_equal(self.lengths, other.lengths):
            raise ValueError("Mismatch in number of beads per chromosome")
        if self.null != other.null:
            raise ValueError("Mismatch in null attribute")

        first = self
        second = other
        if self.shape != other.shape and self.ploidy == 2:
            first = first.ambiguate(copy=True)
            second = second.ambiguate(copy=True)
        if first.shape != second.shape:
            raise ValueError("Mismatch in shape")
        if first.ambiguity != second.ambiguity:
            raise ValueError("Mismatch in ambiguity")

        first_empty = first.bins_nonzero is None and first.bins_zero is None
        second_empty = second.bins_nonzero is None and second.bins_zero is None
        if first_empty and second_empty:
            return self.copy()
        elif first_empty:
            return second.copy()
        elif second_empty:
            return first.copy()
        elif first.bins_nonzero is None:
            raise NotImplementedError(
                "CountsMatrix with only zero bins is not currently supported.")
        elif second.bins_nonzero is None:
            raise NotImplementedError(
                "CountsMatrix with only zero bins is not currently supported.")

        combo = CountsMatrix(
            lengths=self.lengths, ploidy=self.ploidy,
            multiscale_factor=self.multiscale_factor,
            beta=first.beta + second.beta, multires_naive=self.multires_naive,
            weight=(
                first.weight * first.nbins + second.weight * second.nbins) / (
                first.nbins + second.nbins))
        combo.shape = first.shape
        combo.ambiguity = first.ambiguity
        data = pd.DataFrame(np.concatenate(
            [first.bins_nonzero.data.T, second.bins_nonzero.data.T]))
        data['row'] = np.concatenate(
            [first.bins_nonzero.row, second.bins_nonzero.row])
        data['col'] = np.concatenate(
            [first.bins_nonzero.col, second.bins_nonzero.col])
        data = data.groupby(['row', 'col']).sum().reset_index()
        nonzero_row = data.row.values
        nonzero_col = data.col.values
        nonzero_data = data[[c for c in data.columns if c not in (
            'row', 'col')]].values.T
        if self.multiscale_factor == 1 or self.multires_naive:
            nonzero_data = nonzero_data.ravel()

        if self.multiscale_factor == 1 or self.multires_naive:
            nonzero_mask = None
        else:
            # Update _empty_idx_fullres
            tmp_unique, tmp_counts = np.unique(
                np.append(first._empty_idx_fullres,
                          second._empty_idx_fullres),
                return_counts=True)
            combo._empty_idx_fullres = tmp_unique[tmp_counts == 2]

            nonzero_mask = _get_nonzero_mask(
                multiscale_factor=self.multiscale_factor, lengths=self.lengths,
                ploidy=self.ploidy, row=data.row.values, col=data.col.values,
                empty_idx_fullres=combo._empty_idx_fullres)

        combo.bins_nonzero = CountsBins(
            meta=combo, row=nonzero_row, col=nonzero_col, data=nonzero_data,
            mask=nonzero_mask)
        if not (self.exclude_zeros and other.exclude_zeros):
            combo._create_bins_zero()

        return combo

    def __radd__(self, other):
        if other == 0:
            return self.copy()
        else:
            return self.__add__(other)

    def __eq__(self, other):
        if type(other) is type(self):
            if not _dict_is_equal(self.__dict__, other.__dict__):
                return False
            return True
        return NotImplemented

    def __hash__(self):
        return _dict_to_hash(self.__dict__)

    # def __str__(self):

    #     if all([x is None for x in (self.ambiguity, self.shape,
    #                                 self.bins_nonzero, self.exclude_zeros)]):
    #         pass
    #     elif any([x is None for x in (self.ambiguity, self.shape,
    #                                   self.bins_nonzero, self.exclude_zeros)]):
    #         pass
    #     else:
    #         if self.ploidy == 2:
    #             ambiguity = {"ambig": "ambiguous", "pa": "partially ambiguous",
    #                          "ua": "unambiguous"}[self.ambiguity]
    #         self.shape
    #         self.bins_nonzero
    #         self.bins_zero
    #         self.exclude_zeros

    #     ploidy = {1: "haploid", 2: "diploid"}[self.ploidy]
    #     to_print = f""

    #     to_print += f"at {self.multiscale_factor}x resolution"
    #     if self.multires_naive:
    #         to_print += " (formatted for naive multiresolution approach)"

    #     self.lengths
    #     self.null
    #     self.beta
    #     self.weight


class CountsBins(object):
    """Stores counts data, indices, beta, weight, distance matrix indices, etc.

    Counts data and information associated with this counts matrix.

    Parameters
    ----------
    TODO update

    Attributes
    ----------
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
    multires_naive : bool
        Whether to apply the naive multi-resolution model.
    null : bool
        Whether the counts data should be excluded from the poisson component
        of the objective function. The indices of the counts are still used to
        compute the constraint components of the objective function.
    beta : float, optional
        Scaling parameter that determines the size of the structure, relative to
        each counts matrix.
    ambiguity : {"ambig", "pa", "ua"}
        The ambiguity level of the counts data. "ambig" indicates ambiguous,
        "pa" indicates partially ambiguous, and "ua" indicates unambiguous
        or haploid.
    name : {"ambig", "pa", "ua", "ambig0", "pa0", "ua0"}
        For nonzero counts data, this is the same as `ambiguity`. Otherwise,
        it is `amiguity` + "0".
    shape : tuple of int
        Shape of the matrix.
    row : array of int
        Row index array of the matrix (COO format).
    col : array of int
        Column index array of the matrix (COO format).
    row3d : array of int
        Distance matrix rows associated with counts matrix rows.
    col3d : array of int
        Distance matrix columns associated with counts matrix columns.
    mask : array of bool or None
        For multiscale: returns mask for full-res counts data. If None, all
        full-res bins in data are included.
    """

    def __init__(self, meta, row, col, data=None, mask=None):
        self.lengths = meta.lengths
        self.ploidy = meta.ploidy
        self.multiscale_factor = meta.multiscale_factor
        self.multires_naive = meta.multires_naive
        self.null = meta.null  # If True, exclude counts data from primary obj
        self.beta = meta.beta
        self.weight = meta.weight
        self.shape = meta.shape
        if meta.ambiguity is None:
            self.ambiguity = _get_counts_ambiguity(
                self.shape, nbeads=self.lengths.sum() * self.ploidy)
        else:
            self.ambiguity = meta.ambiguity
        self._empty_idx_fullres = meta._empty_idx_fullres

        self.row = np.asarray(  # C-contiguous for hashing
            row, dtype=np.min_scalar_type(row.max()), order='C')
        self.col = np.asarray(  # C-contiguous for hashing
            col, dtype=np.min_scalar_type(col.max()), order='C')
        if mask is None or np.all(mask):
            self.mask = None
        else:
            self.mask = np.asarray(mask, order='C')  # C-contiguous for hashing
        if data is None:
            self.name = f"{self.ambiguity}0"
            self.data = None
            self._sum = 0
        else:
            self.name = self.ambiguity
            self.data = np.asarray(  # C-contiguous for hashing
                data, dtype=_best_counts_dtype(data), order='C')
            self._sum = self.data.sum()

        lengths_lowres = decrease_lengths_res(
            self.lengths, multiscale_factor=self.multiscale_factor)
        self.row3d, self.col3d = _counts_indices_to_3d_indices(
            (self.row, self.col, self.shape), lengths_at_res=lengths_lowres,
            ploidy=self.ploidy)

    @property
    def nbins(self):
        """Number of bins at the current resolution.
        """
        return self.row.size

    def copy(self):
        """Copy counts matrix."""
        return deepcopy(self)

    def filter(self, row, col, copy=True):
        """Filter data for the given indices."""
        if copy:
            filtered = self.copy()
        else:
            filtered = self

        filter_mask = _idx_isin((self.row, self.col), (row, col))
        if filter_mask.sum() == 0:  # All counts bins have been filtered out
            filtered = None
            return filtered

        filtered.row = self.row[filter_mask]
        filtered.col = self.col[filter_mask]
        if self.data is not None:
            if self.multiscale_factor == 1 or self.multires_naive:
                filtered.data = self.data[filter_mask]
            else:
                filtered.data = self.data[:, filter_mask]
            filtered._sum = filtered.data.sum()
        if self.mask is not None:
            filtered.mask = self.mask[:, filter_mask]

        lengths_lowres = decrease_lengths_res(
            self.lengths, multiscale_factor=self.multiscale_factor)
        filtered.row3d, filtered.col3d = _counts_indices_to_3d_indices(
            (filtered.row, filtered.col, self.shape),
            lengths_at_res=lengths_lowres, ploidy=self.ploidy)

        return filtered

    def bias_per_bin(self, bias):
        """Determines bias corresponding to each bin of the distance matrix.

        Returns bias for each bin of the distance matrix by multiplying
        the bias for the bin's row and column.

        Parameters
        ----------
        bias : array of float, optional
            Biases computed by ICE normalization.

        Returns
        -------
        array of float
            Bias for each bin of the distance matrix.
        """
        return _get_bias_per_bin(
            ploidy=self.ploidy, bias=bias, row=self.row, col=self.col,
            multiscale_factor=self.multiscale_factor, lengths=self.lengths,
            multires_naive=self.multires_naive)

    def sum(self, axis=None, dtype=None, out=None):
        """TODO"""
        if (not isinstance(axis, int)) and (
                axis is None or set(list(axis)) == {0, 1}):
            return self._sum
        elif self._sum == 0:
            if axis in (0, 1, -1):
                output = np.zeros((self.shape[int(not axis)]), dtype=dtype)
                if out is not None:
                    output = output.reshape(out.shape)
                return output
            else:
                raise ValueError(f"Axis ({axis}) not understood")
        else:
            return self.tocoo().sum(axis=axis, dtype=dtype, out=out)

    @property
    def fullres_per_lowres_dis(self):
        """
        For multiscale: return number of full-res bins per bin at current res.

        Returns the number of full-resolution counts bins corresponding to each
        low-resolution distance bin.
        """
        if self.multiscale_factor == 1:
            return 1
        elif self.multires_naive:
            fullres_per_lowres_bead = _count_fullres_per_lowres_bead(
                multiscale_factor=self.multiscale_factor, lengths=self.lengths,
                ploidy=self.ploidy, fullres_struct_nan=self._empty_idx_fullres)
            return fullres_per_lowres_bead[self.row] * fullres_per_lowres_bead[
                self.col]
        elif self.mask is not None:
            return self.mask.sum(axis=0)
        else:
            return self.multiscale_factor ** 2

    def __eq__(self, other):
        if type(other) is type(self):
            if not _dict_is_equal(self.__dict__, other.__dict__):
                return False
            return True
        return NotImplemented

    def __hash__(self):
        return _dict_to_hash(self.__dict__)
