import numpy as np
from scipy import sparse
from warnings import warn
import re

from iced.filter import filter_low_counts
from iced.normalization import ICE_normalization

from .utils_poisson import find_beads_to_remove
from .utils_poisson import _intra_counts, _inter_counts, _counts_near_diag

from .multiscale_optimization import decrease_lengths_res
from .multiscale_optimization import decrease_counts_res
from .multiscale_optimization import _group_counts_multiscale


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

    counts = [c for c in counts if c.sum() != 0]

    if len(counts) == 1 and (ploidy == 1 or counts[0].shape == (n, n)):
        c = counts[0]
        if not isinstance(c, np.ndarray):
            c = c.toarray()
        return _check_counts_matrix(
            c, lengths=lengths, ploidy=ploidy, exclude_zeros=exclude_zeros)

    output = np.zeros((n, n))
    for c in counts:
        if isinstance(c, ZeroCountsMatrix):
            continue
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
        output[~np.isnan(c_ambig)] += c_ambig[~np.isnan(c_ambig)]

    output = np.triu(output, 1)
    return _check_counts_matrix(
        output, lengths=lengths, ploidy=ploidy, exclude_zeros=exclude_zeros)


def _ambiguate_beta(beta, counts, lengths, ploidy):
    """Sum betas to be consistent with ambiguated counts.
    """

    if beta is None:
        return beta
    if ploidy == 1:
        return beta[0]

    if not isinstance(counts, list):
        counts = [counts]
    counts = [c for c in counts if (
        isinstance(c, np.ndarray) and np.nansum(c) != 0) or c.sum() != 0]
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
    counts = [c for c in counts if (
        isinstance(c, np.ndarray) and np.nansum(c) != 0) or c.sum() != 0]

    total_counts = np.sum([c.sum() for c in counts])
    beta = []
    for i in range(len(counts)):
        if counts[i].shape[0] == counts[i].shape[1]:
            beta.append(beta_ambig * counts[i].sum() / total_counts)
        else:
            beta.append(beta_ambig * counts[i].sum() / total_counts / 2)  # TODO double check, make unit tests
    return beta


def _included_dis_indices(counts, lengths, ploidy, multiscale_factor):
    """Return row & col of distance matrix bins associated with counts data.
    """

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

    lengths_lowres = decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor)
    nbeads_lowres = lengths_lowres.sum() * ploidy

    row = []
    col = []
    for i in range(len(counts)):
        if max(counts[i].shape) not in (nbeads_lowres, nbeads_lowres / ploidy):
            raise ValueError(
                "Resolution of counts is not consistent with lengths at"
                f" multiscale_factor={multiscale_factor}. Counts shape is ("
                f"{', '.join(map(str, counts[i].shape))}).")
        if isinstance(counts[i], CountsMatrix):
            row_i = counts[i].row3d
            col_i = counts[i].col3d
        else:
            row_i, col_i = _counts_indices_to_3d_indices(
                counts[i], nbeads_lowres=nbeads_lowres)
        row.append(np.minimum(row_i, col_i))
        col.append(np.maximum(row_i, col_i))

    if len(counts) == 1:
        return row[0], col[0]

    row = np.atleast_2d(np.concatenate(row))
    col = np.atleast_2d(np.concatenate(col))
    indices = np.unique(np.concatenate([row, col], axis=0), axis=1)
    row = indices[0]
    col = indices[1]

    return row, col


def _create_unambig_dummy_counts(counts, lengths, ploidy, multiscale_factor=1):
    """Create sparse matrix of 1's with same row and col as distance matrix.
    """

    rows, cols = _included_dis_indices(
        counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor)

    lengths_lowres = decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor)
    nbeads = lengths_lowres.sum() * ploidy

    dummy_counts = sparse.coo_matrix(
        (np.ones_like(rows), (rows, cols)), shape=(nbeads, nbeads)).toarray()
    dummy_counts = sparse.coo_matrix(np.triu(
        np.maximum(dummy_counts + dummy_counts.T, 1), 1))

    return dummy_counts


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
        struct_nan_mask = find_beads_to_remove(
            counts, lengths=lengths,
            ploidy=int(max(counts.shape) / lengths.sum()))
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
        homo1 = counts[:min(counts.shape), :min(counts.shape)]
        homo2 = counts[counts.shape[0] -
                       min(counts.shape):, counts.shape[1] - min(counts.shape):]
        if counts.shape[0] == min(counts.shape):
            homo1 = homo1.T
            homo2 = homo2.T
        if remove_diag:
            np.fill_diagonal(homo1, empty_val)
            np.fill_diagonal(homo2, empty_val)
        homo1[:, struct_nan_mask[:min(counts.shape)] | struct_nan_mask[
            min(counts.shape):]] = empty_val
        homo2[:, struct_nan_mask[:min(counts.shape)] | struct_nan_mask[
            min(counts.shape):]] = empty_val
        # axis=0 is vertical concat
        counts = np.concatenate([homo1, homo2], axis=0)
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
    # if all([isinstance(c, CountsMatrix) for c in counts]):

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
    # counts_prepped, bias = _prep_counts(
    #     counts_raw, lengths=lengths, ploidy=ploidy, normalize=normalize,
    #     filter_threshold=filter_threshold, exclude_zeros=exclude_zeros,
    #     verbose=verbose)

    # Beads to remove from full-res structure
    if multiscale_factor == 1:
        fullres_struct_nan = None
    else:
        fullres_struct_nan = find_beads_to_remove(
            counts_prepped, lengths=lengths, ploidy=ploidy, multiscale_factor=1)

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
            counts_prepped = [_counts_near_diag(
                c, lengths_at_res=lengths, ploidy=ploidy, nbins=excluded_counts,
                exclude_zeros=exclude_zeros) for c in counts_prepped]
            # print(np.array2string((~np.isnan(counts_prepped[0])).astype(int)[:80, :80], max_line_width=np.inf, threshold=np.inf).replace('0', '□').replace('1', '■')); exit(1) # TODO
        elif excluded_counts.lower() == 'intra':
            counts_prepped = [_inter_counts(
                c, lengths_at_res=lengths, ploidy=ploidy,
                exclude_zeros=exclude_zeros) for c in counts_prepped]
        elif excluded_counts.lower() == 'inter':
            counts_prepped = [_intra_counts(
                c, lengths_at_res=lengths, ploidy=ploidy,
                exclude_zeros=exclude_zeros) for c in counts_prepped]
        else:
            raise ValueError(
                "`excluded_counts` must be an integer, 'inter', 'intra' or None.")

    # Format counts as CountsMatrix objects
    counts = _format_counts(
        counts_prepped, beta=beta, input_weight=input_weight,
        lengths=lengths, ploidy=ploidy, exclude_zeros=exclude_zeros,
        multiscale_factor=multiscale_factor)

    # Identify beads to be removed from the final structure
    struct_nan = find_beads_to_remove(
        counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor)
    if mixture_coefs is not None and len(mixture_coefs) > 1:
        lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
        struct_nan_mask = np.full(lengths_lowres * ploidy, False)
        struct_nan_mask[struct_nan] = True
        struct_nan = np.where(np.tile(struct_nan_mask, len(mixture_coefs)))[0]

    return counts, struct_nan, fullres_struct_nan


def _percent_nan_beads(counts):
    """Return percent of beads that would be NaN for current counts matrix.
    """

    struct_nan = find_beads_to_remove(
        counts, lengths=np.array([max(counts.shape)]), ploidy=1)
    return struct_nan.shape / max(counts.shape)


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
            print("FILTERING LOW COUNTS: manually filtering all counts together"
                  f" by {filter_threshold}", flush=True)
        all_counts_ambiguated = ambiguate_counts(
            list(counts_dict.values()), lengths=lengths_lowres, ploidy=ploidy,
            exclude_zeros=True)
        initial_zero_beads = find_beads_to_remove(
            all_counts_ambiguated, lengths=lengths, ploidy=1,
            multiscale_factor=multiscale_factor).sum()
        all_counts_filtered = filter_low_counts(
            sparse.coo_matrix(all_counts_ambiguated), sparsity=False,
            percentage=filter_threshold + _percent_nan_beads(
                all_counts_ambiguated)).tocoo()
        struct_nan = find_beads_to_remove(
            all_counts_filtered, lengths=lengths, ploidy=1,
            multiscale_factor=multiscale_factor)
        struct_nan_mask = np.full(lengths_lowres * ploidy, False)
        struct_nan_mask[struct_nan] = True
        if verbose:
            print('                      removing %d beads' %
                  (struct_nan_mask.sum() - initial_zero_beads), flush=True)
        for counts_type, counts in counts_dict.items():
            if sparse.issparse(counts):
                counts = counts.toarray()
            counts[np.tile(
                struct_nan_mask,
                int(counts.shape[0] / struct_nan_mask.shape[0])), :] = 0.
            counts[:, np.tile(
                struct_nan_mask,
                int(counts.shape[1] / struct_nan_mask.shape[0]))] = 0.
            counts = sparse.coo_matrix(counts)
            counts_dict[counts_type] = counts

    # Optionally normalize counts
    if normalize and bias is None:
        if verbose:
            print('COMPUTING BIAS: all counts together', flush=True)
        bias = ICE_normalization(
            ambiguate_counts(
                list(counts_dict.values()), lengths=lengths_lowres,
                ploidy=ploidy, exclude_zeros=True),
            max_iter=300, output_bias=True)[1].flatten()

    # In each counts matrix, zero out counts for which bias is NaN
    if bias is not None
        for counts_type, counts in counts_dict.items():
            initial_zero_beads = find_beads_to_remove(
                ambiguate_counts(counts, lengths=lengths_lowres, ploidy=ploidy),
                lengths=lengths, ploidy=1,
                multiscale_factor=multiscale_factor).sum()
            if sparse.issparse(counts):
                counts = counts.toarray()
            counts[np.tile(np.isnan(bias), int(counts.shape[0] /
                                               bias.shape[0])), :] = 0.
            counts[:, np.tile(np.isnan(bias), int(counts.shape[1] /
                                                  bias.shape[0]))] = 0.
            counts = sparse.coo_matrix(counts)
            counts_dict[counts_type] = counts
            struct_nan = find_beads_to_remove(
                ambiguate_counts(counts, lengths=lengths_lowres, ploidy=ploidy),
                lengths=lengths, ploidy=1, multiscale_factor=multiscale_factor)
            if verbose and struct_nan.shape - initial_zero_beads > 0:
                print('                removing %d additional beads from %s' %
                      (struct_nan.shape - initial_zero_beads, counts_type),
                      flush=True)

    output_counts = check_counts(
        list(counts_dict.values()), lengths=lengths_lowres, ploidy=ploidy,
        exclude_zeros=exclude_zeros)
    return output_counts, bias


def _set_initial_beta(counts, lengths, ploidy, bias=None, exclude_zeros=False,
                      neighboring_beads_only=True):
    """Estimate compatible betas for each counts matrix."""

    # Set the mean (distance ** alpha) of either
    #   (1) the whole structure, or
    #   (2) distances between beads (default)
    # ...to be equal to 1

    from .constraints import _neighboring_bead_indices

    counts_ambig = ambiguate_counts(
        counts, lengths=lengths, ploidy=ploidy, exclude_zeros=exclude_zeros)

    if neighboring_beads_only:
        row_nghbr = _neighboring_bead_indices(
            lengths=lengths, ploidy=1, multiscale_factor=1)  # ambig so ploidy=1
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

    num_dis_bins = _counts_indices_to_3d_indices(
        counts_ambig, nbeads_lowres=lengths.sum() * ploidy)[0].size
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
    if exclude_zeros:
        beta_ambig = counts_ambig.sum() / num_dis_bins
    else:
        beta_ambig = np.nansum(counts_ambig) / num_dis_bins

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

    # Reformat counts as SparseCountsMatrix or ZeroCountsMatrix instance
    counts_reformatted = []
    for i in range(len(counts)):
        counts_reformatted.append(SparseCountsMatrix(
            counts[i], lengths=lengths, ploidy=ploidy,
            multiscale_factor=multiscale_factor, beta=beta[i],
            weight=input_weight[i]))
        if not exclude_zeros and (counts[i] == 0).sum() > 0:
            zero_counts_maps = ZeroCountsMatrix(
                counts[i], lengths=lengths, ploidy=ploidy,
                multiscale_factor=multiscale_factor, beta=beta[i],
                weight=input_weight[i])
            if zero_counts_maps.nnz > 0:
                counts_reformatted.append(zero_counts_maps)

    return counts_reformatted


def _row_and_col(data):
    """Return row and column indices of non-excluded counts data.
    """

    if isinstance(data, np.ndarray):
        return np.where(~np.isnan(data))
    else:
        return data.row, data.col


def _counts_indices_to_3d_indices(counts, nbeads_lowres):
    """Return distance matrix indices associated with counts matrix data.
    """

    nbeads_lowres = int(nbeads_lowres)

    row3d, col3d = _row_and_col(counts)

    if counts.shape[0] != nbeads_lowres or counts.shape[1] != nbeads_lowres:
        nnz = len(row3d)

        map_factor_rows = int(nbeads_lowres / counts.shape[0])
        map_factor_cols = int(nbeads_lowres / counts.shape[1])
        map_factor = map_factor_rows * map_factor_cols

        x, y = np.indices((map_factor_rows, map_factor_cols))
        x = x.flatten()
        y = y.flatten()

        row3d = np.repeat(
            x, int(nnz * map_factor / x.shape[0])) * min(
                counts.shape) + np.tile(row3d, map_factor)
        col3d = np.repeat(
            y, int(nnz * map_factor / y.shape[0])) * min(
                counts.shape) + np.tile(col3d, map_factor)

    return row3d, col3d


def _update_betas_in_counts_matrices(counts, beta):
    """Updates betas in list of CountsMatrix instances with provided values.
    """

    for counts_maps in counts:
        counts_maps.beta = beta[counts_maps.ambiguity]
    return counts


class CountsMatrix(object):
    """Stores counts data, indices, beta, weight, distance matrix indices, etc.

    Counts data and information associated with this counts matrix.

    Parameters
    ----------
    counts : list of CountsMatrix subclass instances
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
    row : array of int
        Row index array of the matrix (COO format).
    col : array of int
        Column index array of the matrix (COO format).
    shape : tuple of int
        Shape of the matrix.
    input_sum : int
        Sum of the nonzero counts in the input.
    ambiguity : {"ambig", "pa", "ua"}
        The ambiguity level of the counts data. "ambig" indicates ambiguous,
        "pa" indicates partially ambiguous, and "ua" indicates unambiguous
        or haploid.
    name : {"ambig", "pa", "ua", "ambig0", "pa0", "ua0"}
        For nonzero counts data, this is the same as `ambiguity`. Otherwise,
        it is `amiguity` + "0".
    null : bool
        Whether the counts data should be excluded from the poisson component
        of the objective function. The indices of the counts are still used to
        compute the constraint components of the objective function.
    fullres_per_lowres_bead : None or array of int
        For multiscale optimization, this is the number of full-res beads
        corresponding to each low-res bead.
    row3d : array of int
        Distance matrix rows associated with counts matrix rows.
    col3d : array of int
        Distance matrix columns associated with counts matrix columns.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    TODO
    """

    def __init__(self):
        if self.__class__.__name__ in ('CountsMatrix', 'AtypicalCountsMatrix'):
            raise ValueError("This class is not intended"
                             " to be instantiated directly.")
        self.input_sum = None
        self.ambiguity = None
        self.name = None
        self.beta = None
        self.weight = None
        self.null = None
        self.row3d = None
        self.col3d = None

    @property
    def nnz(self):
        """Number of stored values, including explicit zeros.
        """

        pass

    @property
    def data(self):
        """Data array of the matrix (COO format).
        """

        pass

    def toarray(self):
        """Convert counts matrix to numpy array format.
        """

        pass

    def tocoo(self):
        """Convert counts matrix to scipy sparse COO format.
        """

        pass

    def copy(self):
        """Copy counts matrix.
        """

        pass

    def sum(self, axis=None, dtype=None, out=None):
        """Sum of current counts matrix.
        """

        pass

    def bias_per_bin(self, bias):
        """Determines bias corresponding to each bin of the matrix.

        Returns bias for each bin of the contact counts matrix by multiplying
        the bias for the bin's row and column.

        Parameters
        ----------
        bias : array of float, optional
            Biases computed by ICE normalization.

        Returns
        -------
        array of float
            Bias for each bin of the contact counts matrix.
        """

        pass

    @property
    def fullres_per_lowres_dis(self):
        """
        For multiscale: return number of full-res bins per bin at current res.

        Returns the number of full-resolution counts bins corresponding to each
        low-resolution distance bin.
        """

        pass


class SparseCountsMatrix(CountsMatrix):
    """Stores data for non-zero counts bins.
    """

    def __init__(self, counts, lengths, ploidy, multiscale_factor=1, beta=1,
                 weight=1):
        _counts = counts.copy()
        if sparse.issparse(_counts):
            _counts = _counts.toarray()
        self.input_sum = np.nansum(_counts)
        _counts[np.isnan(_counts)] = 0
        if np.array_equal(_counts[~np.isnan(_counts)],
                          _counts[~np.isnan(_counts)].round()):
            _counts = _counts.astype(
                sparse.sputils.get_index_dtype(maxval=_counts.max()))
        _counts = sparse.coo_matrix(_counts)

        self.ambiguity = {1: 'ambig', 1.5: 'pa', 2: 'ua'}[
            sum(_counts.shape) / (lengths.sum() * ploidy)]
        self.name = self.ambiguity
        self.beta = beta
        self.weight = (1. if weight is None else weight)
        if np.isnan(self.weight) or np.isinf(self.weight) or self.weight == 0:
            raise ValueError(f"Counts weight may not be {self.weight}.")
        self.type = 'sparse'
        self.null = False
        self.lengths = lengths
        self.ploidy = ploidy
        self.multiscale_factor = multiscale_factor

        tmp = _group_counts_multiscale(
            _counts, lengths=lengths, ploidy=ploidy,
            multiscale_factor=multiscale_factor)
        self._data, indices, indices3d, self.shape, self.mask = tmp
        self.row, self.col = indices
        self.row3d, self.col3d = indices3d

    @property
    def nnz(self):
        return self.row.shape[0]

    @property
    def data(self):
        return self._data

    def toarray(self):
        # TODO decide what this fxn should actually do (esp wrt reform), and make AtypicalCM match
        lengths_lowres = decrease_lengths_res(
            self.lengths, multiscale_factor=self.multiscale_factor)
        return _check_counts_matrix(
            self.tocoo().toarray(), lengths=lengths_lowres, ploidy=self.ploidy,
            exclude_zeros=False)

    def tocoo(self):
        # TODO decide what this fxn should actually do (esp wrt reform), and make AtypicalCM match
        data = self.data
        if len(data.shape) > 1:
            data = data.sum(axis=0)
        coo = sparse.coo_matrix(
            (data, (self.row, self.col)), shape=self.shape)
        return coo

    def copy(self):
        # TODO update (also AtypicalCM)
        self.row3d = self.row3d.copy()
        self.col3d = self.col3d.copy()
        return self

    def sum(self, axis=None, dtype=None, out=None):
        # TODO update (also AtypicalCM)
        return self.tocoo().sum(axis=axis, dtype=dtype, out=out)

    def bias_per_bin(self, bias):
        if bias is None or np.all(bias == 1):
            return 1
        else:
            if self.multiscale_factor != 1:
                raise NotImplementedError
            bias = bias.flatten()
            bias = np.tile(bias, int(min(self.shape) * self.ploidy / len(bias)))
            return bias[self.row] * bias[self.col]

    def fullres_per_lowres_dis(self):
        if self.multiscale_factor == 1:
            return 1
        else:
            return self.mask.sum(axis=0)


class AtypicalCountsMatrix(CountsMatrix):
    """Stores null counts data or data for zero counts bins.
    """

    def __init__(self):
        CountsMatrix.__init__(self)

    @property
    def nnz(self):
        return self.row.shape[0]

    @property
    def data(self):
        return np.zeros(self._data_grouped_shape, dtype=np.uint16)

    def toarray(self):
        array = np.full(self.shape, np.nan)
        array[self.row, self.col] = 0
        return array

    def tocoo(self):
        return sparse.coo_matrix(self.toarray())

    def copy(self):
        self.row = self.row.copy()
        self.col = self.col.copy()
        self.row3d = self.row3d.copy()
        self.col3d = self.col3d.copy()
        return self

    def sum(self, axis=None, dtype=None, out=None):
        if axis is None or axis == (0, 1) or axis == (1, 0):
            output = 0
        elif axis in (0, 1, -1):
            output = np.zeros((self.shape[int(not axis)]))
        else:
            raise ValueError("Axis %s not understood" % axis)
        if out is not None:
            output = np.array(output).reshape(out.shape)
        if dtype is not None:
            if isinstance(output, np.ndarray):
                output = output.astype(dtype)
            else:
                output = dtype(output)
        return output

    def bias_per_bin(self, bias=None):
        if bias is None or np.all(bias == 1):
            return 1
        else:
            if self.multiscale_factor != 1:
                raise NotImplementedError
            bias = bias.flatten()
            bias = np.tile(bias, int(min(self.shape) * self.ploidy / len(bias)))
            return bias[self.row] * bias[self.col]

    def fullres_per_lowres_dis(self):
        if self.multiscale_factor == 1:
            return 1
        else:
            return self.mask.sum(axis=0)


class ZeroCountsMatrix(AtypicalCountsMatrix):
    """Stores data for zero counts bins.
    """

    def __init__(self, counts, lengths, ploidy, multiscale_factor=1, beta=1,
                 weight=1):
        # counts = counts.copy()
        if sparse.issparse(counts):
            # FIXME I think this should be _check_counts_matrix with exclude_zeros=False, right?
            # because you don't want beads that are all zero to be included - they should be nan
            counts = counts.toarray()
        self.input_sum = np.nansum(counts)
        mask_non0 = (counts != 0)
        # counts[counts != 0] = np.nan
        dummy_counts = counts + 1
        # dummy_counts[np.isnan(dummy_counts)] = 0
        dummy_counts[mask_non0] = 0
        dummy_counts = sparse.coo_matrix(dummy_counts)

        self.ambiguity = {1: 'ambig', 1.5: 'pa', 2: 'ua'}[
            sum(counts.shape) / (lengths.sum() * ploidy)]
        self.name = f'{self.ambiguity}0'
        self.beta = beta
        self.weight = (1. if weight is None else weight)
        if np.isnan(self.weight) or np.isinf(self.weight) or self.weight == 0:
            raise ValueError(f"Counts weight may not be {self.weight}.")
        self.type = 'zero'
        self.null = False
        self.lengths = lengths
        self.ploidy = ploidy
        self.multiscale_factor = multiscale_factor

        tmp = _group_counts_multiscale(
            dummy_counts, lengths=lengths, ploidy=ploidy,
            multiscale_factor=multiscale_factor, dummy=True,
            exclude_each_fullres_zero=True)
        data_grouped, indices, indices3d, self.shape, self.mask = tmp
        self._data_grouped_shape = data_grouped.shape
        self.row, self.col = indices
        self.row3d, self.col3d = indices3d


class NullCountsMatrix(AtypicalCountsMatrix):
    """Stores null counts data.
    """

    def __init__(self, counts, lengths, ploidy, multiscale_factor=1, beta=1,
                 weight=1):

        # TODO make sure all of this is still valid, given multiscale

        if not isinstance(counts, list):
            counts = [counts]

        # Dummy counts need to be inputted because if a row/col is all 0 it is
        # excluded from calculations of constraints.
        # Dummy counts should be "unambiguous" for diploid organisms.
        # All non-zero data in dummy counts is set to 1.
        # (multiscale_factor=1 because counts are always high-resolution)
        dummy_counts = _create_unambig_dummy_counts(
            counts=counts, lengths=lengths, ploidy=ploidy,
            multiscale_factor=1)

        self.ambiguity = 'ua'
        self.name = '%s0' % self.ambiguity
        self.beta = 0.
        self.weight = 0.
        self.type = 'null'
        self.null = True
        self.lengths = lengths
        self.ploidy = ploidy
        self.multiscale_factor = multiscale_factor

        self.input_sum = 0.
        for counts_maps in counts:
            if isinstance(counts_maps, np.ndarray):
                self.input_sum += np.nansum(counts_maps)
            else:
                self.input_sum += counts_maps.sum()

        lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
        tmp = _group_counts_multiscale(
            dummy_counts, lengths=lengths_lowres, ploidy=ploidy,
            multiscale_factor=1, dummy=True)
        data_grouped, indices, indices3d, self.shape, self.mask = tmp
        self._data_grouped_shape = data_grouped.shape
        self.row, self.col = indices
        self.row3d, self.col3d = indices3d
