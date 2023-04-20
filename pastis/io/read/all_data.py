import numpy as np
import os
from scipy import sparse
from iced.io import load_lengths
from .hiclib import load_hiclib_counts
from ...optimization.utils_poisson import subset_chrom_of_data
from ...optimization.counts import _prep_counts, _check_counts_matrix
from ...optimization.counts import CountsMatrix


def _get_lengths(lengths):
    """Parse chromosome lengths, load from file if necessary."""

    if lengths is None:
        raise ValueError("Must input chromosome lengths.")

    if isinstance(lengths, (list, np.ndarray)) and np.asarray(
            lengths).size == 1 and isinstance(lengths[0], str):
        lengths = lengths[0]

    if isinstance(lengths, str):
        try:
            lengths = int(lengths)
        except ValueError:
            if os.path.isfile(lengths):
                lengths = load_lengths(lengths)
            else:
                raise ValueError(
                    f"Chromosome lengths file not found: {lengths}.")

    lengths = np.array(lengths, copy=False, ndmin=1, dtype=int).ravel()
    return lengths


def _get_bias(bias, lengths):
    """Parse bias, load from file if necessary."""

    if bias is None:
        return None
    lengths = _get_lengths(lengths)

    if isinstance(bias, (list, np.ndarray)) and np.asarray(
            bias).size == 1 and isinstance(bias[0], str):
        bias = bias[0]

    if isinstance(bias, str):
        if not os.path.isfile(bias):
            raise ValueError(f"Bias file not found: {bias}.")
        bias = np.loadtxt(bias)

    bias = np.array(bias, copy=False, ndmin=1, dtype=float).ravel()

    if bias.size != lengths.sum():
        raise ValueError("Bias size must be equal to the sum of the chromosome "
                         f"lengths ({lengths.sum()}). It is of size"
                         f" {bias.size}.")
    return bias


def _get_struct(struct, lengths, ploidy):
    """Parse structure, load from file if necessary."""

    if struct is None:
        return None
    lengths = _get_lengths(lengths)

    if isinstance(struct, (list, np.ndarray)) and np.asarray(
            struct).size == 1 and isinstance(struct[0], str):
        struct = struct[0]

    if isinstance(struct, str):
        if not os.path.isfile(struct):
            raise ValueError(f"Structure file not found: {struct}.")
        struct = np.loadtxt(struct)

    struct = np.array(struct, copy=False, dtype=float)

    try:
        struct = struct.reshape(-1, 3)
    except ValueError:
        raise ValueError("Structure should be composed of 3D bead coordinates,"
                         f" {struct.shape=}")
    if struct.shape[0] != lengths.sum() * ploidy:
        raise ValueError(f"The structure must contain {lengths.sum() * ploidy}"
                         f" beads. It contains {struct.shape[0]} beads.")
    return struct


def _get_chrom(chrom, lengths=None):
    """Parse chromosome names, load from file if necessary."""

    if isinstance(chrom, (list, np.ndarray)) and np.asarray(
            chrom).size == 1 and isinstance(chrom[0], str):
        chrom = chrom[0]

    if isinstance(chrom, str):
        if os.path.isfile(chrom):
            chrom = np.loadtxt(chrom, dtype='str')
        elif '/' in chrom or chrom.endswith('.txt'):
            raise ValueError(f"Chromosome names file not found: {chrom}.")
        elif lengths is not None and lengths.size == 1:
            chrom = [chrom]
        elif len(chrom) < 10:  # TODO decide what to do here
            chrom = [chrom]
        else:
            raise ValueError(f"Chromosome names file not found: {chrom}.")

    if lengths is not None:
        lengths = _get_lengths(lengths)

    if chrom is None:  # Create chromosome names: num1, num2, num3... etc
        if lengths is None:
            raise ValueError("Must supply chromosome lengths.")
        chrom = np.array([f'num{i}' for i in range(1, lengths.size + 1)])

    chrom = np.array(chrom, copy=False, ndmin=1, dtype=str).ravel()

    if chrom.size != np.unique(chrom).size:
        raise ValueError("Chromosome names may not contain duplicates.")
    if lengths is not None and chrom.size != lengths.size:
        raise ValueError(f"Size of chromosome names ({chrom.size}) does not"
                         f"match size of chromosome lengths ({lengths.size}).")
    return chrom


def _get_counts(counts, lengths, ploidy):
    """Parse counts matrix, load from file if necessary."""

    if not isinstance(counts, (list, tuple, np.ndarray)):
        counts = [counts]
    counts = list(counts)
    lengths = _get_lengths(lengths)
    if all([isinstance(c, str) for c in counts]):
        counts = counts[:]

    for i in range(len(counts)):
        if isinstance(counts[i], CountsMatrix):
            continue  # Don't check counts matrix

        if isinstance(counts[i], np.ndarray) or sparse.issparse(counts[i]):
            pass
        elif not os.path.isfile(counts[i]):
            raise ValueError(f"Counts file does not exist: {counts[i]}")
        elif counts[i].endswith(".npy") or counts[i].endswith(".npz"):
            counts[i] = np.load(counts[i])
        elif counts[i].endswith(".matrix") or counts[i].endswith(".matrix.gz"):
            counts[i] = load_hiclib_counts(counts[i], lengths=lengths)
        else:
            raise ValueError(  # TODO ask Nelle about this...
                "Counts must be saved as a numpy matrix in binary file format"
                " (.npy/.npz) or as a hiclib file (.matrix/.matrix.gz).")

        counts[i] = _check_counts_matrix(
            counts[i], lengths=lengths, ploidy=ploidy, copy=False)

    return counts


def load_data(counts, lengths_full, ploidy, chrom_full=None,
              chrom_subset=None, filter_threshold=0.04, normalize=True,
              bias=None, struct_true=None, verbose=False):
    """Load all input data from files, and/or reformat data objects.

    If files are provided, load data from files. Also reformats data objects.

    Parameters
    ----------
    counts : list of str or list of array or list of coo_matrix
        Counts data files in the hiclib format or as numpy ndarrays.
    lengths_full : str or list
        Number of beads per homolog of each chromosome in the inputted data, or
        hiclib .bed file with lengths data.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    chrom_full : str or list of str, optional
        Label for each chromosome in the in the inputted data, or file with
        chromosome labels (one label per line).
    chrom_subset : list of str, optional
        Label for each chromosome to be excised from the full data; labels of
        chromosomes for which inference should be performed.
    normalize : bool, optional
        Perform ICE normalization on the counts prior to optimization.
        Normalization is reccomended.
    filter_threshold : float, optional
        Ratio of non-zero beads to be filtered out. Filtering is
        reccomended.

    Returns
    -------
    counts : coo_matrix of int or ndarray or int
        Counts data. If `chrom_subset` is not None, only counts data for the
        specified chromosomes are returned.
    lengths_subset : array of int
        Number of beads per homolog of each chromosome in the returned data. If
        `chrom_subset` is not None, only chromosome lengths for the specified
        chromosomes are returned.
    chrom_subset : array of str
        Label for each chromosome in the returned data; labels of chromosomes
        for which inference should be performed.
    lengths_full : array of int
        Number of beads per homolog of each chromosome in the inputted data.
    chrom_full : array of str
        Label for each chromosome in the inputted data.
    struct_true : None or array of float
        The true structure. If `chrom_subset` is not None, only beads for the
        specified chromosomes are returned.
    """

    # Load
    lengths_full = _get_lengths(lengths_full)
    chrom_full = _get_chrom(chrom_full, lengths=lengths_full)
    if chrom_subset is not None:
        chrom_subset = _get_chrom(chrom_subset)
    counts = _get_counts(counts, lengths=lengths_full, ploidy=ploidy)
    bias = _get_bias(bias, lengths=lengths_full)
    struct_true = _get_struct(struct_true, lengths=lengths_full, ploidy=ploidy)

    # Optionally limit the data to the specified chromosomes
    lengths_subset, chrom_subset, data_subset = subset_chrom_of_data(
        ploidy=ploidy, lengths_full=lengths_full, chrom_full=chrom_full,
        chrom_subset=chrom_subset, counts=counts, bias=bias,
        structures=struct_true)
    counts = data_subset['counts']
    bias = data_subset['bias']
    struct_true = data_subset['struct']

    # Filter counts and compute bias
    counts_are_prepped = False
    counts_tmp = counts
    if all([isinstance(c, CountsMatrix) and (
            c.multiscale_factor == 1) for c in counts]):
        if filter_threshold is None or filter_threshold == 0:
            if bias is not None:
                counts_are_prepped = True
                bias[bias == 0] = np.nan
            elif not normalize:
                counts_are_prepped = True
            else:
                counts_tmp = [c.tocoo() for c in counts]
    if not counts_are_prepped:
        counts, bias = _prep_counts(
            counts_tmp, lengths=lengths_subset, ploidy=ploidy,
            filter_threshold=filter_threshold, normalize=normalize, bias=bias,
            verbose=verbose)

    return (counts, bias, lengths_subset, chrom_subset, lengths_full,
            chrom_full, struct_true)
