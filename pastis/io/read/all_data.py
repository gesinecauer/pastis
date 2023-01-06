import numpy as np
import os
from scipy import sparse
from iced.io import load_lengths
from .hiclib import load_hiclib_counts
from ...optimization.utils_poisson import subset_chrom_of_data
from ...optimization.counts import _prep_counts, _check_counts_matrix


def _get_lengths(lengths):
    """Load chromosome lengths from file, or reformat lengths object.
    """

    if lengths is None:
        raise ValueError("Must input chromosome lengths")
    if isinstance(lengths, str) and os.path.exists(lengths):
        lengths = load_lengths(lengths)
    elif isinstance(lengths, (list, np.ndarray)):
        if len(lengths) == 1 and isinstance(lengths[0], str) and os.path.exists(
                lengths[0]):
            lengths = load_lengths(lengths[0])
    lengths = np.array(lengths, dtype=int, ndmin=1, copy=False).flatten()
    return lengths


def _get_bias(bias, lengths):
    """Load bias from file, or reformat lengths object."""

    if bias is None:
        return None
    lengths = _get_lengths(lengths)
    if isinstance(bias, str) and os.path.exists(bias):
        bias = np.loadtxt(bias)
    elif isinstance(bias, (list, np.ndarray)):
        if len(bias) == 1 and isinstance(bias[0], str) and os.path.exists(
                bias[0]):
            bias = np.loadtxt(bias[0])
    bias = np.array(bias, dtype=float, ndmin=1, copy=False).flatten()
    if bias.size != lengths.sum():
        raise ValueError("Bias size must be equal to the sum of the chromosome "
                         f"lengths ({lengths.sum}). It is of size {bias.size}.")
    return bias


def _get_chrom(chrom, lengths=None):
    """Load chromosome names from file, or reformat chromosome names object.
    """

    if isinstance(chrom, str) and os.path.exists(chrom):
        chrom = np.loadtxt(chrom, dtype='str')
    elif chrom is not None and isinstance(chrom, (list, np.ndarray)):
        if len(chrom) == 1 and isinstance(chrom[0], str) and os.path.exists(
                chrom[0]):
            chrom = np.loadtxt(chrom[0], dtype='str')
        chrom = np.array(chrom, dtype=str, ndmin=1, copy=False).flatten()
    else:
        if lengths is None:
            raise ValueError("Must supply chromosome lengths")
        lengths = _get_lengths(lengths)
        chrom = np.array([f'num{i}' for i in range(1, lengths.size + 1)])
    return chrom


def _get_counts(counts, lengths, ploidy):
    """Load counts from file, or reformat counts object.
    """

    if not isinstance(counts, list):
        counts = [counts]
    lengths = _get_lengths(lengths)

    for i in range(len(counts)):
        if isinstance(counts[i], np.ndarray) or sparse.issparse(counts[i]):
            pass
        elif counts[i].endswith(".npy") or counts[i].endswith(".npz"):
            counts[i] = np.load(counts[i])
        elif counts[i].endswith(".matrix") or counts[i].endswith(".matrix.gz"):
            counts[i] = load_hiclib_counts(counts[i], lengths=lengths)
        else:
            raise ValueError(
                "Counts must be saved as a numpy binary file (.npy/.npz) or"
                " a hiclib file (.matrix/.matrix.gz).")

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
    counts = _get_counts(counts, lengths=lengths_full, ploidy=ploidy)
    bias = _get_bias(bias, lengths=lengths_full)
    if struct_true is not None and isinstance(struct_true, str):
        struct_true = np.loadtxt(struct_true).reshape(-1, 3)

    # Optionally limit the data to the specified chromosomes
    lengths_subset, chrom_subset, data_subset = subset_chrom_of_data(
        ploidy=ploidy, lengths_full=lengths_full, chrom_full=chrom_full,
        chrom_subset=chrom_subset, counts=counts, bias=bias,
        structures=struct_true)
    counts = data_subset['counts']
    bias = data_subset['bias']
    struct_true = data_subset['struct']

    # Filter counts and compute bias
    counts, bias = _prep_counts(
        counts, lengths=lengths_subset, ploidy=ploidy,
        filter_threshold=filter_threshold, normalize=normalize, bias=bias,
        verbose=verbose)

    return (counts, bias, lengths_subset, chrom_subset, lengths_full,
            chrom_full, struct_true)
