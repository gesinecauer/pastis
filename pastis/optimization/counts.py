import numpy as np
from scipy import sparse
from warnings import warn
import re
import copy
import pandas as pd

from iced.filter import filter_low_counts
from iced.normalization import ICE_normalization

from .utils_poisson import find_beads_to_remove
from .utils_poisson import _intra_counts, _inter_counts, _counts_near_diag

from .multiscale_optimization import decrease_lengths_res
from .multiscale_optimization import decrease_counts_res
from .multiscale_optimization import _group_counts_multiscale
from .multiscale_optimization import _get_fullres_counts_index


def ambiguate_counts(counts, lengths, ploidy, exclude_zeros=False):
    """Convert diploid counts to ambiguous & aggregate counts across matrices.

    If diploid, convert unambiguous and partially ambiguous counts to ambiguous
    and aggregate list of counts into a single counts matrix. If haploid,
    check format of and return the inputted counts matrix.

    Parameters
    ----------
    counts : list of array or coo_matrix or CountsMatrix instances
        Counts data.
    lengths : array_like of int
        Number of beads per homolog of each chromosome.

    Returns
    -------
    coo_matrix or ndarray
        Aggregated and ambiguated contact counts matrix.
    """

    lengths = np.asarray(lengths)
    n = lengths.sum()
    if not isinstance(counts, list):
        counts = [counts]

    if all([isinstance(c, CountsMatrix) for c in counts]):
        counts_ambig = sum(counts).ambiguate()
        if exclude_zeros:
            counts_ambig.bins_zero = None
        elif counts_ambig.bins_zero is None:
            counts_ambig._create_bins_zero()
        return counts_ambig

    if len(counts) == 1 and (ploidy == 1 or counts[0].shape == (n, n)):
        c = counts[0]
        return _check_counts_matrix(
            c, lengths=lengths, ploidy=ploidy, exclude_zeros=exclude_zeros)

    counts_ambig = np.zeros((n, n))
    for c in counts:
        counts_sum = c.sum()
        counts_sum = counts_sum if not np.isnan(counts_sum) else np.nansum(c)
        if np.isnan(counts_sum) or counts_sum == 0:
            continue
        if not isinstance(c, np.ndarray):
            c = c.toarray()
        c = _check_counts_matrix(
            c, lengths=lengths, ploidy=ploidy, exclude_zeros=True).toarray()
        if c.shape[0] > c.shape[1]:
            c_ambig = np.nansum(
                [c[:n, :], c[n:, :], c[:n, :].T, c[n:, :].T], axis=0)
        elif c.shape[0] < c.shape[1]:
            c_ambig = np.nansum(
                [c[:, :n].T, c[:, n:].T, c[:, :n], c[:, n:]], axis=0)
        elif c.shape[0] == n:
            c_ambig = c
        else:
            c_ambig = np.nansum(
                [c[:n, :n], c[:n, n:], c[:n, n:].T, c[n:, n:]], axis=0)
        counts_ambig[~np.isnan(c_ambig)] += c_ambig[~np.isnan(c_ambig)]

    counts_ambig = np.triu(counts_ambig, 1)
    return _check_counts_matrix(
        counts_ambig, lengths=lengths, ploidy=ploidy,
        exclude_zeros=exclude_zeros)


def _ambiguate_beta(beta, counts, lengths, ploidy):
    """Sum betas to be consistent with ambiguated counts.
    """

    if beta is None:
        return beta
    if ploidy == 1:
        return beta[0]

    if not isinstance(counts, list):
        counts = [counts]
    if not isinstance(beta, list):
        beta = [beta]
    if len(counts) != len(beta):
        raise ValueError(f"Inconsistent number of betas ({len(beta)}) and"
                         f" counts matrices ({len(counts)}).")

    beta_ambig = 0.
    for i in range(len(beta)):
        if counts[i].shape[0] == counts[i].shape[1]:
            beta_ambig += beta[i]
        else:
            beta_ambig += beta[i] * 2
    return beta_ambig


def _disambiguate_beta(beta_ambig, counts, lengths, ploidy, bias=None):
    """Derive beta for each counts matrix from ambiguated beta.
    """
    if beta_ambig is None or ploidy == 1:
        return [beta_ambig]

    if not isinstance(counts, list):
        counts = [counts]

    total_counts = 0
    for i in range(len(counts)):
        if isinstance(counts[i], np.ndarray):
            total_counts += np.nansum(counts[i])
        else:
            total_counts += counts[i].sum()

    beta = []
    for i in range(len(counts)):
        if isinstance(counts[i], np.ndarray):
            counts_sum = np.nansum(counts[i])
        else:
            counts_sum = counts[i].sum()
        if counts_sum == 0:
            continue
        if counts[i].shape[0] == counts[i].shape[1]:
            beta.append(beta_ambig * counts_sum / total_counts)
        else:
            beta.append(beta_ambig * counts_sum / total_counts / 2)  # FIXME double check, make unit tests
    return beta


def _check_counts_matrix(counts, lengths, ploidy, exclude_zeros=False,
                         chrom_subset_index=None, remove_diag=True,
                         copy=True):
    """Check counts dimensions, reformat, & excise selected chromosomes.
    """

    if copy:
        counts = counts.copy()

    if chrom_subset_index is not None and len(
            chrom_subset_index) / max(counts.shape) not in (1, 2):
        raise ValueError("chrom_subset_index size (%d) does not fit counts"
                         " shape (%d, %d)." %
                         (len(chrom_subset_index), counts.shape[0],
                             counts.shape[1]))
    if len(counts.shape) != 2:
        raise ValueError(
            "Counts matrix must be two-dimensional, current shape = (%s)"
            % ', '.join([str(x) for x in counts.shape]))
    if any([x > lengths.sum() * ploidy for x in counts.shape]):
        raise ValueError("Counts matrix shape (%d, %d) is greater than number"
                         " of beads (%d) in %s genome." %
                         (counts.shape[0], counts.shape[1],
                             lengths.sum() * ploidy,
                             {1: "haploid", 2: "diploid"}[ploidy]))
    if any([x / lengths.sum() not in (1, 2) for x in counts.shape]):
        raise ValueError("Counts matrix shape (%d, %d) does not match lenghts"
                         " (%s)"
                         % (counts.shape[0], counts.shape[1],
                             ", ".join(map(str, lengths))))

    empty_val = 0
    struct_nan_mask = np.full((max(counts.shape)), False)
    if not exclude_zeros:
        empty_val = np.nan
        struct_nan = find_beads_to_remove(
            counts, lengths=lengths,
            ploidy=int(max(counts.shape) / lengths.sum()))
        struct_nan_mask[struct_nan] = True
        counts = counts.astype(float)

    if sparse.issparse(counts) or isinstance(counts, CountsMatrix):
        counts = counts.toarray()
    if not isinstance(counts, np.ndarray):
        counts = np.array(counts)

    if not np.array_equal(counts[~np.isnan(counts)],
                          counts[~np.isnan(counts)].round()):
        warn("Counts matrix must only contain integers or NaN")

    if counts.shape[0] == counts.shape[1]:
        if remove_diag:
            counts[np.tril_indices(counts.shape[0])] = empty_val
        else:
            counts[np.tril_indices(counts.shape[0], -1)] = empty_val
        counts[struct_nan_mask, :] = empty_val
        counts[:, struct_nan_mask] = empty_val
        if chrom_subset_index is not None:
            counts = counts[chrom_subset_index[:counts.shape[0]], :][
                :, chrom_subset_index[:counts.shape[1]]]
    elif min(counts.shape) * 2 == max(counts.shape):
        hmlg1 = counts[:min(counts.shape), :min(counts.shape)]
        hmlg2 = counts[counts.shape[0] -
                       min(counts.shape):, counts.shape[1] - min(counts.shape):]
        if counts.shape[0] == min(counts.shape):
            hmlg1 = hmlg1.T
            hmlg2 = hmlg2.T
        if remove_diag:
            np.fill_diagonal(hmlg1, empty_val)
            np.fill_diagonal(hmlg2, empty_val)
        hmlg1[:, struct_nan_mask[:min(counts.shape)] | struct_nan_mask[
            min(counts.shape):]] = empty_val
        hmlg2[:, struct_nan_mask[:min(counts.shape)] | struct_nan_mask[
            min(counts.shape):]] = empty_val
        # axis=0 is vertical concat
        counts = np.concatenate([hmlg1, hmlg2], axis=0)
        counts[struct_nan_mask, :] = empty_val
        if chrom_subset_index is not None:
            counts = counts[chrom_subset_index[:counts.shape[0]], :][
                :, chrom_subset_index[:counts.shape[1]]]
    else:
        raise ValueError("Input counts matrix is - %d by %d. Counts must be"
                         " n-by-n or n-by-2n or 2n-by-2n." %
                         (counts.shape[0], counts.shape[1]))

    if exclude_zeros:
        counts[np.isnan(counts)] = 0
        counts = sparse.coo_matrix(counts)

    return counts


def check_counts(counts, lengths, ploidy, exclude_zeros=False,
                 chrom_subset_index=None):
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

    Returns
    -------
    counts : list of array or coo_matrix
        Checked and reformatted counts data.
    """

    lengths = np.array(lengths)
    if not isinstance(counts, list):
        counts = [counts]
    return [_check_counts_matrix(
        c, lengths=lengths, ploidy=ploidy, exclude_zeros=exclude_zeros,
        chrom_subset_index=chrom_subset_index) for c in counts]


def preprocess_counts(counts_raw, lengths, ploidy, multiscale_factor=1,
                      beta=None, excluded_counts=None, mixture_coefs=None,
                      exclude_zeros=False, input_weight=None, verbose=True,
                      simple_diploid=False, mods=[]):
    """Check counts, reformat, reduce resolution, filter, and compute bias.

    Optionally reduce resolution. Counts are also checked and reformatted
    for inference. Final matrices are stored as CountsMatrix subclass instances.

    Parameters
    ----------
    counts_raw : list of array or coo_matrix
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
        Biases computed by ICE normalization.
    struct_nan : array of int
        Beads that should be removed (set to NaN) in the structure.
    """

    # FIXME what about if betas are different?

    if simple_diploid:
        if ploidy != 2:
            raise ValueError("Ploidy is not 2, but simple_diploid specified.")
        counts_raw = check_counts(
            counts_raw, lengths=lengths, ploidy=2, exclude_zeros=exclude_zeros)
        beta = 2 * _ambiguate_beta(
            beta, counts=counts_raw, lengths=lengths, ploidy=2)
        counts_raw = [ambiguate_counts(
            counts=counts_raw, lengths=lengths, ploidy=2,
            exclude_zeros=exclude_zeros)]
        ploidy = 1

    # # Filter counts and compute bias
    # counts_raw, bias = _prep_counts(
    #     counts_raw, lengths=lengths, ploidy=ploidy, normalize=normalize,
    #     filter_threshold=filter_threshold, exclude_zeros=exclude_zeros,
    #     verbose=verbose)

    # Beads to remove from full-res structure
    if multiscale_factor == 1:
        fullres_struct_nan = None
    else:
        fullres_struct_nan = find_beads_to_remove(
            counts_raw, lengths=lengths, ploidy=ploidy, multiscale_factor=1)

    # Optionally exclude certain counts
    if excluded_counts is not None:
        if isinstance(excluded_counts, float):
            if excluded_counts != int(excluded_counts):
                raise ValueError("excluded_counts must be an integer.")
            excluded_counts = int(excluded_counts)
        if isinstance(excluded_counts, str) and re.match(
                r'[0-9]+(\.0){0,1}', excluded_counts):
            excluded_counts = int(excluded_counts)
        if isinstance(excluded_counts, int):
            counts_raw = [_counts_near_diag(
                c, lengths_at_res=lengths, ploidy=ploidy, nbins=excluded_counts,
                exclude_zeros=exclude_zeros) for c in counts_raw]
            # print(np.array2string((~np.isnan(counts_raw[0])).astype(int)[:80, :80], max_line_width=np.inf, threshold=np.inf).replace('0', '□').replace('1', '■')); exit(1) # TODO
        elif excluded_counts.lower() == 'intra':
            counts_raw = [_inter_counts(
                c, lengths_at_res=lengths, ploidy=ploidy,
                exclude_zeros=exclude_zeros) for c in counts_raw]
        elif excluded_counts.lower() == 'inter':
            counts_raw = [_intra_counts(
                c, lengths_at_res=lengths, ploidy=ploidy,
                exclude_zeros=exclude_zeros) for c in counts_raw]
        else:
            raise ValueError(
                "`excluded_counts` must be an integer, 'inter', 'intra' or None.")

    # Format counts as CountsMatrix objects
    counts = _format_counts(
        counts_raw, beta=beta, input_weight=input_weight,
        lengths=lengths, ploidy=ploidy, exclude_zeros=exclude_zeros,
        multiscale_factor=multiscale_factor)

    # Identify beads to be removed from the final structure
    struct_nan = find_beads_to_remove(
        counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor)
    if mixture_coefs is not None and len(mixture_coefs) > 1:
        lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
        struct_nan_mask = np.full(lengths_lowres.sum() * ploidy, False)
        struct_nan_mask[struct_nan] = True
        struct_nan = np.where(np.tile(struct_nan_mask, len(mixture_coefs)))[0]

    return counts, struct_nan, fullres_struct_nan


def _prep_counts(counts_list, lengths, ploidy=1, multiscale_factor=1,
                 filter_threshold=0.04, normalize=True, bias=None,
                 exclude_zeros=True, verbose=True):
    """Copy counts, check matrix, reduce resolution, filter, and compute bias.
    """

    if not isinstance(counts_list, list):
        counts_list = [counts_list]

    # Copy counts
    counts_list = [c.copy() for c in counts_list]

    # Check counts
    counts_list = check_counts(
        counts_list, lengths=lengths, ploidy=ploidy, exclude_zeros=True)

    # Determine ambiguity
    nbeads = lengths.sum() * ploidy
    counts_dict = [('haploid' if ploidy == 1 else {
        1: 'ambig', 1.5: 'pa', 2: 'ua'}[
        sum(c.shape) / nbeads], c) for c in counts_list]
    if len(counts_dict) != len(dict(counts_dict)):
        raise ValueError("Can't input multiple counts matrices of the same"
                         f" type. Inputs ({len(counts_dict)}) = "
                         f"{', '.join([x[0] for x in counts_dict])}")
    counts_dict = dict(counts_dict)

    # Reduce resolution
    lengths_lowres = decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor)
    for counts_type, counts in counts_dict.items():
        if multiscale_factor != 1:
            counts_dict[counts_type] = decrease_counts_res(
                counts, multiscale_factor=multiscale_factor, lengths=lengths,
                ploidy=ploidy)

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
        counts_ambig = ambiguate_counts(
            list(counts_dict.values()), lengths=lengths_lowres, ploidy=ploidy,
            exclude_zeros=True)

        # How many beads are initially zero?
        initial_zero_num = find_beads_to_remove(
            counts_ambig, lengths=lengths, ploidy=1,
            multiscale_factor=multiscale_factor).size
        initial_zero_perc = initial_zero_num / lengths.sum()

        # Filter ambiguated counts, and get mask of beads that are NaN
        all_counts_filtered = filter_low_counts(
            sparse.coo_matrix(counts_ambig), sparsity=False,
            percentage=filter_threshold + initial_zero_perc).tocoo()
        struct_nan = find_beads_to_remove(
            all_counts_filtered, lengths=lengths, ploidy=1,
            multiscale_factor=multiscale_factor)
        struct_nan_mask = np.full(lengths_lowres.sum(), False)
        struct_nan_mask[struct_nan] = True
        if verbose:
            if initial_zero_num > 0:
                print(f"{' ' * 22}{initial_zero_num} beads are already zero",
                      flush=True)
            num_removed = struct_nan.size - initial_zero_num
            print(f"{' ' * 22}Removed {num_removed} bead(s)", flush=True)

        # Remove the NaN beads from all counts matrices
        for counts_type, counts in counts_dict.items():
            if sparse.issparse(counts):
                counts = counts.toarray()
            counts[np.tile(
                struct_nan_mask,
                int(counts.shape[0] / struct_nan_mask.size)), :] = 0
            counts[:, np.tile(
                struct_nan_mask,
                int(counts.shape[1] / struct_nan_mask.size))] = 0
            counts = sparse.coo_matrix(counts)
            counts_dict[counts_type] = counts

    # Optionally normalize counts
    # Normalization is done on ambiguated counts
    if normalize and bias is None:
        if verbose:
            print('COMPUTING BIAS', flush=True)
        counts_ambig = ambiguate_counts(
            list(counts_dict.values()), lengths=lengths_lowres,
            ploidy=ploidy, exclude_zeros=True)
        bias = ICE_normalization(
            counts_ambig, max_iter=300, output_bias=True)[1].flatten()

    # In each counts matrix, zero out counts for which bias is NaN
    if bias is not None:
        for counts_type, counts in counts_dict.items():
            # How many beads are initially zero?
            counts_ambig = ambiguate_counts(
                counts, lengths=lengths_lowres, ploidy=ploidy)
            initial_zero_num = find_beads_to_remove(
                counts_ambig, lengths=lengths, ploidy=1,
                multiscale_factor=multiscale_factor).size

            # Remove beads with bias=NaN from counts
            if sparse.issparse(counts):
                counts = counts.toarray()
            counts[np.tile(
                np.isnan(bias), int(counts.shape[0] / bias.size)), :] = 0
            counts[:, np.tile(
                np.isnan(bias), int(counts.shape[1] / bias.size))] = 0
            counts = sparse.coo_matrix(counts)
            counts_dict[counts_type] = counts

            if verbose:
                struct_nan = find_beads_to_remove(
                    counts_ambig, lengths=lengths, ploidy=1,
                    multiscale_factor=multiscale_factor)
                num_removed = struct_nan.size - initial_zero_num
                if num_removed > 0:
                    print(f"{' ' * 22}Removing {num_removed} additional "
                          f"bead(s) from {counts_type}", flush=True)

    output_counts = check_counts(
        list(counts_dict.values()), lengths=lengths_lowres, ploidy=ploidy,
        exclude_zeros=exclude_zeros)
    return output_counts, bias


def _set_initial_beta(counts, lengths, ploidy, bias=None, exclude_zeros=False,
                      neighboring_beads_only=True):
    """Estimate compatible betas for each counts matrix."""

    # Set the mean (distance ** alpha) for either
    #   (1) the whole structure, or
    #   (2) distances between neighboring beads (default)
    # ...to be equal to 1

    from .constraints import _neighboring_bead_indices

    # Get reelvant counts
    counts_ambig = ambiguate_counts(
        counts, lengths=lengths, ploidy=ploidy, exclude_zeros=exclude_zeros)
    if neighboring_beads_only:
        row_nghbr = _neighboring_bead_indices(
            lengths=lengths, ploidy=1, multiscale_factor=1)
        if exclude_zeros:
            mask = np.isin(counts_ambig.row, row_nghbr) & (
                counts_ambig.col == counts_ambig.row + 1)
            counts_ambig = sparse.coo_matrix(
                (counts_ambig.data[mask],
                    (counts_ambig.row[mask], counts_ambig.col[mask])),
                shape=counts_ambig.shape)
        else:
            counts_diag = np.diagonal(counts_ambig, 1)
            counts_ambig = np.full_like(counts_ambig, np.nan)
            counts_ambig[row_nghbr, row_nghbr + 1] = counts_diag[row_nghbr]

    # Get number of distance bins associated with revant counts
    num_dis_bins = _counts_indices_to_3d_indices(
        counts_ambig, nbeads_lowres=lengths.sum() * ploidy)[0].size
    if neighboring_beads_only:
        # Intentionally dividing num_dis_bins / 2 for diploid: we assume that
        # the contribution of inter-hmlg counts to nghbr counts is negligible
        num_dis_bins /= ploidy

    # Normalize counts
    if bias is not None:
        if exclude_zeros:
            bias_per_bin = bias[counts_ambig.row] * bias[counts_ambig.col]
            counts_ambig = sparse.coo_matrix(
                (counts_ambig.data * bias_per_bin,
                    (counts_ambig.row, counts_ambig.col)),
                shape=counts_ambig.shape)
        else:
            bias = bias.reshape(-1, 1)
            counts_ambig = counts_ambig / bias / bias.T

    # Get universal/ambiguated beta
    if exclude_zeros:
        beta_ambig = counts_ambig.sum() / num_dis_bins
    else:
        beta_ambig = np.nansum(counts_ambig) / num_dis_bins

    # Assign separate betas to each counts matrix
    beta = _disambiguate_beta(
        beta_ambig, counts=counts, lengths=lengths, ploidy=ploidy, bias=bias)

    return beta_ambig, beta


def _format_counts(counts, lengths, ploidy, beta=None, bias=None,
                   input_weight=None, exclude_zeros=False, multiscale_factor=1):
    """Format each counts matrix as a CountsMatrix subclass instance.
    """

    # Check input
    counts = check_counts(
        counts, lengths=lengths, ploidy=ploidy, exclude_zeros=exclude_zeros)

    if beta is not None:
        if not isinstance(beta, (list, np.ndarray)):
            beta = [beta]
        if len(beta) != len(counts):
            raise ValueError(
                "Beta needs to contain as many scaling factors as there are "
                f"datasets ({len(counts)}). It is of length ({len(beta)})")
    else:
        _, beta = _set_initial_beta(
            counts, lengths=lengths, ploidy=ploidy, bias=bias,
            exclude_zeros=exclude_zeros)

    if input_weight is not None:
        if not isinstance(input_weight, (list, np.ndarray)):
            input_weight = [input_weight]
        if len(input_weight) != len(counts):
            raise ValueError("input_weights needs to contain as many weighting"
                             " factors as there are datasets (%d). It is of"
                             " length (%d)" % (len(counts), len(input_weight)))
        input_weight = np.array(input_weight)
        if input_weight.sum() not in (0, 1):
            input_weight *= len(input_weight) / input_weight.sum()
    else:
        input_weight = [1.] * len(counts)

    # Reformat counts as CountsMatrix instance
    counts_reformatted = []
    for i in range(len(counts)):
        counts_matrix = CountsMatrix(
            lengths=lengths, ploidy=ploidy, counts=counts[i],
            multiscale_factor=multiscale_factor, beta=beta[i],
            weight=input_weight[i])
        if not exclude_zeros:
            counts_matrix._create_bins_zero()
        counts_reformatted.append(counts_matrix)

    return counts_reformatted


def _row_and_col(data):
    """Return row and column indices of non-excluded counts data.
    """

    if isinstance(data, np.ndarray):
        return np.where(~np.isnan(data))
    else:
        return data.row, data.col


def _counts_indices_to_3d_indices(data, nbeads_lowres):
    """Return distance matrix indices associated with counts matrix data.
    """

    if isinstance(data, tuple) and len(data) == 3:
        row3d, col3d, shape = data
    else:
        row3d, col3d = _row_and_col(data)
        shape = data.shape

    nbeads_lowres = int(nbeads_lowres)
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

    return row3d, col3d


def _idx_isin(idx1, idx2):
    """Whether each (row, col) pair in idx1 (row1, col1) is in idx2 (row2, col2)
    """

    if isinstance(idx1, (list, tuple)):
        idx1 = np.stack(idx1, axis=1)
    if isinstance(idx2, (list, tuple)):
        idx2 = np.stack(idx2, axis=1)
    return (idx1 == idx2[:, None]).all(axis=2).any(axis=0)


def _get_nonzero_mask(multiscale_factor, lengths, ploidy, row, col,
                      empty_idx_fullres=None):
    """TODO"""

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

    return nonzero_mask


class CountsMatrix(object):
    """Stores counts data, indices, beta, weight, distance matrix indices, etc.

    Counts data and information associated with this counts matrix.

    Parameters
    ----------
    counts : array or coo_matrix
        Preprocessed counts data.
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
    beta : float, optional
        Scaling parameter that determines the size of the structure, relative to
        each counts matrix.

    Attributes
    ----------
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
    null : bool
        Whether the counts data should be excluded from the primary objective
        function. Counts are still used in the calculation of the constraints.
    beta : float, optional
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

    def __init__(self, lengths, ploidy, counts=None, multiscale_factor=1,
                 beta=1, weight=1):
        self.lengths = lengths
        self.ploidy = ploidy
        self.multiscale_factor = multiscale_factor
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

        if counts is not None:
            self.ambiguity = {1: 'ambig', 1.5: 'pa', 2: 'ua'}[
                sum(counts.shape) / (self.lengths.sum() * self.ploidy)]

            if self.multiscale_factor > 1:
                self._empty_idx_fullres = find_beads_to_remove(
                    counts, lengths=self.lengths, ploidy=self.ploidy)

            data, (row, col), self.shape, mask = _group_counts_multiscale(
                counts, lengths=self.lengths, ploidy=self.ploidy,
                multiscale_factor=self.multiscale_factor)
            if np.array_equal(data, data.round()):  # TODO Use Mozes function for this
                data = data.astype(
                    sparse.sputils.get_index_dtype(maxval=data.max()))
            self.bins_nonzero = CountsBins(
                meta=self, row=row, col=col, data=data, mask=mask)

    @property
    def bins(self):
        """TODO"""
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
        if self.bins_nonzero is not None:
            filtered.bins_nonzero = filtered.bins_nonzero.filter(
                row=row, col=col, copy=False)
        if self.bins_zero is not None:
            filtered.bins_zero = filtered.bins_zero.filter(
                row=row, col=col, copy=False)
        return filtered

    def toarray(self):
        """Convert counts matrix to numpy array format."""
        lengths_lowres = decrease_lengths_res(
            self.lengths, multiscale_factor=self.multiscale_factor)
        return _check_counts_matrix(
            self.tocoo().toarray(), lengths=lengths_lowres, ploidy=self.ploidy,
            exclude_zeros=False)

    def tocoo(self):
        """Convert counts matrix to scipy sparse COO format."""
        data = self.bins_nonzero.data
        if len(data.shape) > 1:
            data = data.sum(axis=0)
        coo = sparse.coo_matrix(
            (data, (self.bins_nonzero.row, self.bins_nonzero.col)),
            shape=self.shape)
        return coo

    def sum(self, axis=None, dtype=None, out=None):
        """Sum of current counts matrix."""
        if (not isinstance(axis, int)) and (
                axis is None or set(list(axis)) == {0, 1}):
            return self.bins_nonzero._sum
        return self.tocoo().sum(axis=axis, dtype=dtype, out=out)

    def copy(self):  # XXX
        """Copy counts matrix."""
        return copy.deepcopy(self)

    def update_beta(self, beta):
        """Update beta."""
        if isinstance(beta, dict):
            beta = beta[self.ambiguity]
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
        _empty_idx_lowres = find_beads_to_remove(
            self, lengths=self.lengths, ploidy=self.ploidy,
            multiscale_factor=self.multiscale_factor)

        if self.multiscale_factor == 1:
            zero_mask = None
            dummy = _check_counts_matrix(
                np.ones(self.shape), lengths=self.lengths, ploidy=self.ploidy,
                exclude_zeros=True)

            # Exclude low-res bins that contain any nonzero counts
            # Exclude low-res loci that have no data in any low-res row/col bin
            idx_all_mask = ~_idx_isin(
                (dummy.row, dummy.col),
                (self.bins_nonzero.row, self.bins_nonzero.col)) & ~np.isin(
                dummy.row, _empty_idx_lowres) & ~np.isin(
                dummy.col, _empty_idx_lowres)
            zero_row = dummy.row[idx_all_mask]
            zero_col = dummy.col[idx_all_mask]
        else:
            idx_fullres_all, idx_lowres_all = _get_fullres_counts_index(
                multiscale_factor=self.multiscale_factor, lengths=self.lengths,
                ploidy=self.ploidy,
                counts_fullres_shape=(self.lengths.sum(), self.lengths.sum()))
            row_lowres_all, col_lowres_all = idx_lowres_all

            # Exclude low-res bins that contain any nonzero counts
            # Exclude low-res loci that have no data in any low-res row/col bin
            idx_all_mask = ~_idx_isin(
                idx_lowres_all,
                (self.bins_nonzero.row, self.bins_nonzero.col)) & ~np.isin(
                row_lowres_all, _empty_idx_lowres) & ~np.isin(
                col_lowres_all, _empty_idx_lowres)
            zero_row = row_lowres_all[idx_all_mask]
            zero_col = col_lowres_all[idx_all_mask]
            row_fullres, col_fullres, bad_idx_fullres = [x.reshape(
                self.multiscale_factor ** 2,
                -1)[:, idx_all_mask] for x in idx_fullres_all]

            # mask=False for full-res loci that have no data in any row/col
            zero_mask = ~bad_idx_fullres & ~np.isin(
                row_fullres, self._empty_idx_fullres) & ~np.isin(
                col_fullres, self._empty_idx_fullres)

        if zero_row.size == 0:
            self.bins_zero = None
        else:
            self.bins_zero = CountsBins(
                meta=self, row=zero_row, col=zero_col, mask=zero_mask)

    def ambiguate(self, copy=True):
        """Convert diploid counts to ambiguous.

        If  unambiguous or partially ambiguous diploid, convert to ambiguous. If
        haploid or ambiguous diploid, return current counts matrix.

        Returns
        -------
        CountsMatrix object
            Ambiguated contact counts matrix.
        """

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
                weight=self.weight)
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
        if self.multiscale_factor > 1:
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
        if self.multiscale_factor == 1:
            nonzero_data = nonzero_data.ravel()
        if np.array_equal(nonzero_data, nonzero_data.round()):  # TODO Use Mozes function for this
            nonzero_data = nonzero_data.astype(
                sparse.sputils.get_index_dtype(maxval=nonzero_data.max()))
        nonzero_row = data.row.values
        nonzero_col = data.col.values

        if self.multiscale_factor == 1:
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
        if self.bins_zero is not None:
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

        combo = CountsMatrix(
            lengths=self.lengths, ploidy=self.ploidy,
            multiscale_factor=self.multiscale_factor,
            beta=first.beta + second.beta,
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
        if self.multiscale_factor == 1:
            nonzero_data = nonzero_data.ravel()
        if np.array_equal(nonzero_data, nonzero_data.round()):  # TODO Use Mozes function for this
            nonzero_data = nonzero_data.astype(
                sparse.sputils.get_index_dtype(maxval=nonzero_data.max()))

        if self.multiscale_factor == 1:
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
        if self.bins_zero is not None or other.bins_zero is not None:
            combo._create_bins_zero()

        return combo

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __eq__(self, other):  # TODO remove print statements
        if type(other) is type(self):
            # return self.__dict__ == other.__dict__
            if self.__dict__.keys() != other.__dict__.keys():
                print('~~~~~~ __dict__ keys mismatch')
                return False
            for key in self.__dict__.keys():
                if type(self.__dict__[key]) is not type(other.__dict__[key]):
                    print(f'~~~~~~ type(__dict__[key]) mismatch: {key}')
                    print(f"{type(self.__dict__[key])=}")
                    print(f"{type(other.__dict__[key])=}")
                    return False
                if isinstance(self.__dict__[key], (list, np.ndarray)):
                    if ((self.__dict__[key].dtype.char in np.typecodes[
                            'AllInteger']) and (other.__dict__[
                            key].dtype.char in np.typecodes['AllInteger'])):
                        if not np.array_equal(
                                self.__dict__[key], other.__dict__[key]):
                            print(f'~~~~~~ __dict__ array not equal int: {key}')
                            return False
                    elif not np.allclose(
                            self.__dict__[key], other.__dict__[key]):
                        print(f'~~~~~~ __dict__ array not equal: {key}')
                        return False
                elif self.__dict__[key] != other.__dict__[key]:
                    print(f'~~~~~~ __dict__ value not equal: {key}')
                    return False
            return True
        return NotImplemented

    def __hash__(self):  # XXX
        __dict__ = []
        for x in self.__dict__.items():
            if isinstance(x, np.ndarray):
                x = x.tolist()
            if isinstance(x, list):
                x = tuple(x)
            __dict__.append(x)
        return hash(tuple(sorted(__dict__)))



class CountsBins(object):
    """Stores counts data, indices, beta, weight, distance matrix indices, etc.

    Counts data and information associated with this counts matrix.

    Parameters
    ----------
    counts : array or coo_matrix
        Preprocessed counts data.
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
    beta : float, optional
        Scaling parameter that determines the size of the structure, relative to
        each counts matrix.

    Attributes
    ----------
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
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
    mask : array of bool
        For multiscale: returns mask for full-res counts data
    """

    def __init__(self, meta, row, col, data=None, mask=None):
        self.lengths = meta.lengths
        self.ploidy = meta.ploidy
        self.multiscale_factor = meta.multiscale_factor
        self.null = meta.null  # If True, exclude counts data from primary obj
        self.beta = meta.beta
        self.weight = meta.weight
        self.ambiguity = meta.ambiguity
        self.shape = meta.shape

        self.row = row
        self.col = col
        self.data = data
        self.mask = mask
        if self.data is None:
            self._sum = 0
            self.name = f"{self.ambiguity}0"
        else:
            self._sum = self.data.sum()
            self.name = self.ambiguity

        n = decrease_lengths_res(
            self.lengths, multiscale_factor=self.multiscale_factor).sum()
        self.row3d, self.col3d = _counts_indices_to_3d_indices(
            (self.row, self.col, self.shape), nbeads_lowres=n * self.ploidy)

    @property
    def nbins(self):
        """Number of bins at the current resolution.
        """
        return self.row.size

    def copy(self):  # TODO don't include this method?
        """Copy counts matrix."""
        return copy.deepcopy(self)

    def filter(self, row, col, copy=True):
        """Filter data for the given indices."""
        if copy:
            filtered = self.copy()
        else:
            filtered = self

        filter_mask = _idx_isin((self.row, self.col), (row, col))
        filtered.row = self.row[filter_mask]
        filtered.col = self.col[filter_mask]
        if self.multiscale_factor == 1:
            filtered.data = self.data[filter_mask]
            filtered.mask = None
        else:
            if self.data is not None:
                filtered.data = self.data[:, filter_mask]
            filtered.mask = self.mask[:, filter_mask]

        n = decrease_lengths_res(
            self.lengths, multiscale_factor=self.multiscale_factor).sum()
        filtered.row3d, filtered.col3d = _counts_indices_to_3d_indices(
            (filtered.row, filtered.col, self.shape),
            nbeads_lowres=n * self.ploidy)

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
        if bias is None or np.all(bias == 1):
            return 1
        else:
            bias = bias.flatten()
            bias = np.tile(bias, int(min(self.shape) * self.ploidy / len(bias)))
            return bias[self.row3d] * bias[self.col3d]

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
        else:
            return self.mask.sum(axis=0)

    def __eq__(self, other):  # TODO remove print statements
        if type(other) is type(self):
            # return self.__dict__ == other.__dict__
            if self.__dict__.keys() != other.__dict__.keys():
                print('~~~~~~ __dict__ keys mismatch')
                return False
            for key in self.__dict__.keys():
                if type(self.__dict__[key]) is not type(other.__dict__[key]):
                    print(f'~~~~~~ type(__dict__[key]) mismatch: {key}')
                    print(f"{type(self.__dict__[key])=}")
                    print(f"{type(other.__dict__[key])=}")
                    return False
                if isinstance(self.__dict__[key], (list, np.ndarray)):
                    if ((self.__dict__[key].dtype.char in np.typecodes[
                            'AllInteger']) and (other.__dict__[
                            key].dtype.char in np.typecodes['AllInteger'])):
                        if not np.array_equal(
                                self.__dict__[key], other.__dict__[key]):
                            print(f'~~~~~~ __dict__ array not equal int: {key}')
                            return False
                    elif not np.allclose(
                            self.__dict__[key], other.__dict__[key]):
                        print(f'~~~~~~ __dict__ array not equal: {key}')
                        return False
                elif self.__dict__[key] != other.__dict__[key]:
                    print(f'~~~~~~ __dict__ value not equal: {key}')
                    return False
            return True
        return NotImplemented

    def __hash__(self):
        __dict__ = []
        for x in self.__dict__.items():
            if isinstance(x, np.ndarray):
                x = x.tolist()
            if isinstance(x, list):
                x = tuple(x)
            __dict__.append(x)
        return hash(tuple(sorted(__dict__)))
