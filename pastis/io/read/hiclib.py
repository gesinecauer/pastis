import warnings
import numpy as np
from scipy import sparse
import pandas as pd
from ...optimization.counts import _best_counts_dtype


def _get_counts_shape(row_max, col_max, lengths=None):
    """
    Return shape of counts matrix.
    """

    if lengths is None:
        if round((row_max + 1) / (col_max + 1)) == 2:
            n = max(row_max + 1, (col_max + 1) * 2)
        elif round((col_max + 1) / (row_max + 1)) == 2:
            n = max((row_max + 1) * 2, col_max + 1)
        else:
            n = max(row_max, col_max) + 1
    else:
        lengths = np.array(lengths)
        n = lengths.sum()

    nrows = row_max + 1
    ncols = col_max + 1

    # Round up to nearest (n/2)
    row_factor = np.ceil(nrows / (n / 2)) / 2
    col_factor = np.ceil(ncols / (n / 2)) / 2
    if {row_factor, col_factor} not in ({2}, {1}, {1, 2}):
        raise ValueError(
            f"Counts matrix shape does not match chrom lengths.\nThere are {n}"
            f" beads per homolog, and counts matrix has at least {nrows:g} rows"
            f" and {ncols:g} columns.")
    nrows = int(n * row_factor)
    ncols = int(n * col_factor)

    return (nrows, ncols)


def load_hiclib_counts(filename, lengths=None):
    """
    Fast loading of a raw interaction counts file

    Parameters
    ----------
    filename : str
        path to the file to load. The file should be of the following format:
        i, j, counts

    lengths : ndarray
        lengths of each chromosomes

    Returns
    --------
    X : the interaction counts file
    """

    n = None
    if lengths is not None:
        n = lengths.sum()

    # This is the interaction count files
    dataframe = pd.read_csv(filename, sep="\t", comment="#", header=None)
    row, col, data = dataframe.values.T[:3]

    # If there are NAs remove them
    mask = np.isnan(data)
    if np.any(mask):
        warnings.warn(
            f"NaN detected in {filename}. Removing NaN and replacing with 0.")
        row = row[np.invert(mask)]
        col = col[np.invert(mask)]
        data = data[np.invert(mask)]

    # XXX We need to deal with the fact that we should not duplicate entries
    # for the diagonal.
    # XXX what if n doesn't exist?
    if (col.min() >= 1 and row.min() >= 1) and \
       ((n is None) or (max(col.max(), row.max()) == n)):
        # This is a hack to deal with the fact that sometimes, the files are
        # indexed at 1 and not 0
        col -= 1
        row -= 1

    shape = _get_counts_shape(
        row_max=row.max(), col_max=col.max(), lengths=lengths)

    data = data.astype(_best_counts_dtype(data, warn_on_float=False))
    counts = sparse.coo_matrix((data, (row, col)), shape=shape)
    return counts
