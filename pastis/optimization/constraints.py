import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import numpy as np
from scipy import sparse

use_jax = True
if use_jax:
    from absl import logging as absl_logging
    absl_logging.set_verbosity('error')
    from jax.config import config as jax_config
    jax_config.update("jax_platform_name", "cpu")
    jax_config.update("jax_enable_x64", True)

    import jax.numpy as ag_np #import autograd.numpy as ag_np
    SequenceBox = list #from autograd.builtins import SequenceBox
    from jax import lax
    from jax.nn import relu
else:
    import autograd.numpy as ag_np
    from autograd.builtins import SequenceBox

from .multiscale_optimization import decrease_lengths_res
from .multiscale_optimization import _count_fullres_per_lowres_bead
from .utils_poisson import find_beads_to_remove, _intra_counts_mask


class Constraints(object):
    """Compute objective constraints.

    Prepares constraints and computes the negative log likelhood of each
    constraint.

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
    constraint_lambdas : dict, optional
        Lambdas for each constraint. Keys should match `constraint_params`
        when applicable.
    constraint_params : dict, optional
        Any parameters used for the calculation of each constraint. Keys should
        be in `constraint_lambdas`.

    Attributes
    ----------
    lambdas : dict
        Lambdas for each constraint.
    params : dict
        Any parameters used for the calculation of each constraint.
    row : array of int
        Rows of the distance matrix to be used in calculation of constraints.
    col : array of int
        Columns of the distance matrix to be used in calculation of constraints.
    row_adj : array of int
        Rows of the distance matrix indicating adjacent beads, to be used in
        calculation of the bead-chain-connectivity constraint.
    col_adj : array of int
        Columns of the distance matrix indicating adjacent beads, to be used in
        calculation of the bead-chain-connectivity constraint.
    lengths : array of int
        Number of beads per homolog of each chromosome in the full-resolution
        data.
    lengths_lowres : array of int
        Number of beads per homolog of each chromosome in the data at the
        current resolution (defined by `multiscale_factor`).
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    multiscale_factor : int
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.

    """

    def __init__(self, counts, lengths, ploidy, multiscale_factor=1,
                 constraint_lambdas=None, constraint_params=None,
                 fullres_torm=None, verbose=True, mods=[]):

        self.lengths = np.asarray(lengths).astype(np.int32)
        self.lengths_lowres = decrease_lengths_res(
            lengths, multiscale_factor=multiscale_factor)
        self.ploidy = ploidy
        self.multiscale_factor = multiscale_factor
        if constraint_lambdas is None:
            self.lambdas = {}
        elif isinstance(constraint_lambdas, dict):
            self.lambdas = constraint_lambdas
        else:
            raise ValueError("Constraint lambdas must be inputted as dict.")
        if constraint_params is None:
            self.params = {}
        elif isinstance(constraint_params, dict):
            for hsc in ("hsc", "mhs"):
                if hsc in constraint_params and constraint_params[hsc] is not None:
                    constraint_params[hsc] = np.array(
                        constraint_params[hsc]).flatten().astype(float)
            self.params = constraint_params
        else:
            raise ValueError("Constraint params must be inputted as dict.")
        if mods is None:  # TODO remove
            self.mods = []
        elif isinstance(mods, str):
            self.mods = mods.lower().split('.')
        else:
            self.mods = [x.lower() for x in mods]

        self.check(verbose=verbose)

        self.bead_weights = None
        if self.lambdas["hsc"] or self.lambdas["mhs"]:
            if multiscale_factor != 1:
                fullres_per_lowres_bead = _count_fullres_per_lowres_bead(
                    multiscale_factor=multiscale_factor, lengths=lengths,
                    ploidy=ploidy, fullres_torm=fullres_torm)
                bead_weights = fullres_per_lowres_bead / multiscale_factor
            else:
                bead_weights = np.ones((self.lengths_lowres.sum() * ploidy,))
            torm = find_beads_to_remove(
                counts, lengths=lengths, ploidy=ploidy,
                multiscale_factor=multiscale_factor)
            bead_weights[torm] = 0.
            n = self.lengths_lowres.sum()
            begin = end = 0
            for i in range(len(self.lengths_lowres)):
                end = end + self.lengths_lowres[i]
                bead_weights[:n][begin:end] /= np.sum(
                    bead_weights[:n][begin:end])
                bead_weights[n:][begin:end] /= np.sum(
                    bead_weights[n:][begin:end])
                begin = end
            self.bead_weights = bead_weights.reshape(-1, 1)

        self.subtracted = None
        if self.lambdas["mhs"]:
            lambda_intensity = np.ones((self.lengths.shape[0],))
            self.subtracted = (lambda_intensity.sum() - (
                1 * np.log(lambda_intensity)).sum())

        self.row_nghbr = self.col_nghbr = None
        if self.lambdas["bcc"]:
            self.row_nghbr, self.col_nghbr = _neighboring_bead_indices(
                counts=counts, lengths=lengths, ploidy=ploidy,
                multiscale_factor=multiscale_factor)

        self.laplacian = None
        self.rho = None
        if self.lambdas["shn"]:
            self.laplacian, self.affinity = _get_laplacian_matrix(
                counts=counts, lengths=lengths, ploidy=ploidy,
                multiscale_factor=multiscale_factor, sigma=self.params["shn"])
            nonzero_entries = total_entries = 0
            for i in range(len(counts)):
                if counts[i].sum() > 0:
                    nonzero_entries += counts[i].nnz
                total_entries += counts[i].nnz
            nbeads = self.lengths_lowres.sum() * ploidy
            self.rho = max(
                (1 - nonzero_entries / total_entries) * np.sqrt(nbeads),
                min(3, 0.2 * np.sqrt(nbeads)))

    def check(self, verbose=True):
        """Check constraints object.

        Check that lambdas are greater than zero, and that necessary parameters
        are supplied. Optionally print summary of constraints.

        Parameters
        ----------
        verbose : bool
            Verbosity.
        """

        # Set defaults
        lambda_defaults = {"bcc": 0., "hsc": 0., "shn": 0., "mhs": 0.}
        lambda_all = lambda_defaults
        if self.lambdas is not None:
            for k, v in self.lambdas.items():
                if k not in lambda_all:
                    raise ValueError(
                        "constraint_lambdas key not recognized - %s" % k)
                elif v is not None:
                    lambda_all[k] = float(v) # np.float32(v) # FIXME revert
        self.lambdas = lambda_all

        params_defaults = {"hsc": None, "shn": None, 'bcc': 1, "mhs": None}
        params_all = params_defaults
        if self.params is not None:
            for k, v in self.params.items():
                if k not in params_all:
                    raise ValueError('params key not recognized - %s' % k)
                elif v is not None:
                    # if isinstance(v, int) or isinstance(v, float):  # FIXME revert
                    #     v = np.float32(v)  # FIXME revert
                    if isinstance(v, list) or isinstance(v, np.ndarray):
                        v = np.asarray(v) #.astype(np.float32)  # FIXME revert
                    params_all[k] = v
        self.params = params_all

        # Check constraints
        for k, v in self.lambdas.items():
            if v != lambda_defaults[k]:
                if v < 0:
                    raise ValueError("Lambdas must be >= 0. Lambda for"
                                     " %s is %g" % (k, v))
                if k in self.params and self.params[k] is None:
                    raise ValueError("Lambda for %s is supplied,"
                                     " but constraint is not" % k)
            elif k in self.params and not np.array_equal(self.params[k],
                                                         params_defaults[k]):
                # print(self.params[k], type(self.params[k])) # TODO remove on main branch
                raise ValueError("Constraint for %s is supplied, but lambda is"
                                 " 0" % k)

        if (self.lambdas["hsc"] or self.lambdas["mhs"]) and self.ploidy == 1:
            raise ValueError("Homolog-separating constraint can not be"
                             " applied to haploid genome.")

        # Print constraints
        constraint_names = {"bcc": "bead chain connectivity",
                            "hsc": "homolog-separating",
                            "shn": "ShNeigh"}
        lambda_to_print = {k: v for k, v in self.lambdas.items() if v != 0}
        if verbose and len(lambda_to_print) > 0:
            for constraint, lambda_val in lambda_to_print.items():
                print("CONSTRAINT: %s lambda = %.2g" % (
                    constraint_names[constraint], lambda_val), flush=True)
                if constraint not in self.params:
                    continue
                if constraint in ("hsc", "mhs"):
                    if self.params[constraint] is None:
                        print("            param = inferred", flush=True)
                    elif isinstance(self.params[constraint], np.ndarray):
                        label = "            param = "
                        print(label + np.array2string(
                            self.params[constraint],
                            formatter={'float_kind': lambda x: "%.3g" % x},
                            prefix=" " * len(label), separator=", "))
                    elif isinstance(self.params[constraint], float):
                        print(f"            param = "
                              f"{self.params[constraint]:.3g}", flush=True)
                    else:
                        print(f"            {self.params[constraint]}",
                              flush=True)
                if constraint == 'bcc':
                    print(f"            param = {self.params[constraint]:.3g}",
                          flush=True)

    def apply(self, structures, alpha=None, inferring_alpha=False,
              mixture_coefs=None):
        """Apply constraints using given structure(s).

        Compute negative log likelhood for each constraint using the given
        structure.

        Parameters
        ----------
        structures : array or autograd SequenceBox or list of structures
            3D chromatin structure(s) for which to compute the constraint.
        alpha : float, optional
            Biophysical parameter of the transfer function used in converting
            counts to wish distances. If alpha is not specified, it will be
            inferred.
        inferring_alpha : bool, optional
            A value of "True" indicates that the current optimization aims to
            infer alpha, rather than the structure.

        Returns
        -------
        dict
            Dictionary of constraint names and negative log likelihoods.

        """

        if len(self.lambdas) == 0 or sum(self.lambdas.values()) == 0:
            return {}

        if mixture_coefs is None:
            mixture_coefs = [1.]
        if not (isinstance(structures, list) or isinstance(structures, SequenceBox)):
            structures = [structures]
        if len(structures) != len(mixture_coefs):
            raise ValueError(
                "The number of structures (%d) and of mixture coefficents (%d)"
                " should be identical." % (len(structures), len(mixture_coefs)))

        obj = {k: 0. for k, v in self.lambdas.items() if v != 0}

        if self.lambdas["bcc"] and not inferring_alpha:
            for struct, gamma in zip(structures, mixture_coefs):
                nghbr_dis = ag_np.sqrt((ag_np.square(
                    struct[self.row_nghbr] - struct[self.col_nghbr])).sum(axis=1))
                n_edges = nghbr_dis.shape[0]
                nghbr_dis_var = n_edges * ag_np.square(
                    nghbr_dis).sum() / ag_np.square(nghbr_dis.sum())
                obj_bcc = nghbr_dis_var - 1.
                obj_bcc = ag_np.power(obj_bcc, self.params["bcc"])
                obj['bcc'] = obj['bcc'] + gamma * self.lambdas['bcc'] * obj_bcc
        if self.lambdas["hsc"] and not inferring_alpha:
            for struct, gamma in zip(structures, mixture_coefs):
                homo_sep = self._homolog_separation(struct)
                homo_sep_diff = self.params["hsc"] - homo_sep
                # homo_sep_diff = 1 - homo_sep / self.params["hsc"]  # with scaleR

                if 'hsc10' in self.mods or 'hsc20' in self.mods:
                    if 'hsc20' in self.mods:
                        hsc_cutoff = 0.2
                    if 'hsc10' in self.mods:
                        hsc_cutoff = 0.1
                    hsc_r_hsc_cutoff = hsc_cutoff * self.params["hsc"]

                    gt0 = homo_sep_diff > 0
                    homo_sep_diff = homo_sep_diff.at[gt0].set(relu(
                        homo_sep_diff[gt0] - hsc_r_hsc_cutoff))
                    homo_sep_diff = homo_sep_diff.at[~gt0].set(-relu(
                        -(homo_sep_diff[~gt0] + hsc_r_hsc_cutoff)))
                elif 'HSCnoRELU'.lower() not in self.mods:
                    homo_sep_diff = relu(homo_sep_diff)
                    print("I thought we weren't doing RELU for HSC anymore"); exit(0)


                homo_sep_diff_sq = ag_np.square(homo_sep_diff)
                if 'sum_not_mean' in self.mods:
                    hsc = ag_np.sum(homo_sep_diff_sq)  # TODO fix on main branch: mean not sum!
                else:
                    hsc = ag_np.mean(homo_sep_diff_sq)  # TODO fix on main branch: mean not sum!
                obj["hsc"] = obj["hsc"] + gamma * self.lambdas["hsc"] * hsc
        if self.lambdas["shn"] and not inferring_alpha:
            for struct, gamma in zip(structures, mixture_coefs):
                tmp1 = ag_np.dot(struct.T, self.laplacian)
                tmp2 = ag_np.dot(tmp1, struct)
                shn = self.rho * 2 * ag_np.trace(tmp2)
                obj["shn"] = obj["shn"] + gamma * self.lambdas["shn"] * shn

        # Check constraints objective
        for k, v in obj.items():
            if ag_np.isnan(v) or ag_np.isinf(v):
                raise ValueError(f"Constraint {k.upper()} is {v}")

        return {'obj_' + k: v for k, v in obj.items()}

    def _homolog_separation(self, struct):
        """Compute distance between homolog centers of mass per chromosome.
        """
        struct_bw = struct * np.repeat(self.bead_weights, 3, axis=1)
        n = self.lengths_lowres.sum()

        homo_sep = ag_np.zeros(self.lengths_lowres.shape)
        begin = end = 0
        for i in range(self.lengths_lowres.shape[0]):
            end = end + ag_np.int32(self.lengths_lowres[i])
            chrom1_mean = ag_np.sum(struct_bw[begin:end], axis=0)
            chrom2_mean = ag_np.sum(struct_bw[(n + begin):(n + end)], axis=0)
            homo_sep_i = ag_np.sqrt(ag_np.sum(ag_np.square(
                chrom1_mean - chrom2_mean)))
            homo_sep = homo_sep.at[i].set(homo_sep_i)
            begin = end

        return homo_sep


def _neighboring_bead_indices(counts, lengths, ploidy, multiscale_factor):
    """Return row & col of neighboring beads, along a homolog of a chromosome.
    """

    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
    nbeads = lengths_lowres.sum() * ploidy

    row_nghbr = np.arange(nbeads - 1, dtype=int)
    col_nghbr = np.arange(1, nbeads, dtype=int)

    # Remove beads for which there is no counts data
    torm = find_beads_to_remove(
        counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor)
    where_torm = np.where(torm)[0]
    nghbr_dis_mask = (~np.isin(row_nghbr, where_torm)) & (
        ~np.isin(col_nghbr, where_torm))

    row_nghbr = row_nghbr[nghbr_dis_mask]
    col_nghbr = col_nghbr[nghbr_dis_mask]

    # Remove if "neighbor" beads are actually on different chromosomes
    # or homologs
    bins = np.tile(lengths_lowres, ploidy).cumsum()
    same_bin = np.digitize(row_nghbr, bins) == np.digitize(col_nghbr, bins)

    row_nghbr = row_nghbr[same_bin]
    col_nghbr = col_nghbr[same_bin]

    return row_nghbr, col_nghbr


def _get_laplacian_matrix(counts, lengths, ploidy, multiscale_factor,
                          sigma=None):
    """TODO"""

    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
    nbeads = lengths_lowres.sum() * ploidy

    if sigma is None:
        sigma = 0.023 * nbeads
        sigma /= 10

    torm = find_beads_to_remove(
        counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor)

    included_idx = np.where(~torm)[0]
    affinity_matrix = np.zeros((nbeads, nbeads))
    affinity_matrix[included_idx, :] += included_idx
    affinity_matrix[:, included_idx] -= included_idx.reshape(-1, 1)
    # print(np.abs(affinity_matrix[:10, :10]).astype(int))
    affinity_matrix = -np.square(affinity_matrix) / (2 * np.square(sigma))
    affinity_matrix = np.exp(affinity_matrix)

    # np.set_printoptions(threshold=np.inf); start = 30; end = 50
    # print((affinity_matrix != 0).astype(int)[start:end, start:end])
    affinity_matrix = sparse.coo_matrix(affinity_matrix)
    mask = _intra_counts_mask(affinity_matrix, lengths_counts=lengths_lowres)

    # nbins = 10
    # mask_near_diag = np.abs(affinity_matrix.row - affinity_matrix.col) <= nbins
    # mask = mask & mask_near_diag

    affinity_matrix = sparse.coo_matrix(
        (affinity_matrix.data[mask], (affinity_matrix.row[mask],
            affinity_matrix.col[mask])), shape=affinity_matrix.shape).toarray()
    # print((affinity_matrix != 0).astype(int)[start:end, start:end]); exit(0)

    diagonal_matrix = np.zeros((nbeads, nbeads))
    np.fill_diagonal(diagonal_matrix, affinity_matrix.sum(axis=1))

    laplacian_matrix = diagonal_matrix - affinity_matrix

    return laplacian_matrix, affinity_matrix


def _inter_homolog_dis(struct, lengths):
    """Computes distance between homologs for a normal diploid structure.
    """

    struct = struct.copy().reshape(-1, 3)

    n = int(struct.shape[0] / 2)
    homo1 = struct[:n, :]
    homo2 = struct[n:, :]

    homo_dis = []
    begin = end = 0
    for l in lengths:
        end += l
        if np.isnan(homo1[begin:end, 0]).sum() == l or np.isnan(
                homo2[begin:end, 0]).sum() == l:
            homo_dis.append(np.nan)
        else:
            homo_dis.append(((np.nanmean(homo1[
                begin:end, :], axis=0) - np.nanmean(
                homo2[begin:end, :], axis=0)) ** 2).sum() ** 0.5)
        begin = end

    homo_dis = np.array(homo_dis)
    homo_dis[np.isnan(homo_dis)] = np.nanmean(homo_dis)

    return homo_dis


def _inter_homolog_dis_via_simple_diploid(struct, lengths):
    """Computes distance between chromosomes for a faux-haploid structure.
    """

    from sklearn.metrics import euclidean_distances

    struct = struct.copy().reshape(-1, 3)

    chrom_barycenters = []
    begin = end = 0
    for l in lengths:
        end += l
        if np.isnan(struct[begin:end, 0]).sum() < l:
            chrom_barycenters.append(
                np.nanmean(struct[begin:end, :], axis=0).reshape(1, 3))
        begin = end

    chrom_barycenters = np.concatenate(chrom_barycenters)

    homo_dis = euclidean_distances(chrom_barycenters)
    homo_dis[np.tril_indices(homo_dis.shape[0])] = np.nan

    return np.full(lengths.shape, np.nanmean(homo_dis))


def distance_between_homologs(structures, lengths, mixture_coefs=None,
                              simple_diploid=False):
    """Computes distances between homologs for a given structure.

    For diploid organisms, this computes the distance between homolog centers
    of mass for each chromosome.

    Parameters
    ----------
    structures : array of float or list of array of float
        3D chromatin structure(s) for which to assess inter-homolog distances.
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    simple_diploid: bool, optional
        For diploid organisms: whether the structure is an inferred "simple
        diploid" structure in which homologs are assumed to be identical and
        completely overlapping with one another.

    Returns
    -------
    array of float
        Distance between homologs per chromosome.

    """

    from .utils_poisson import _format_structures

    structures = _format_structures(
        structures=structures, lengths=lengths,
        ploidy=(1 if simple_diploid else 2),
        mixture_coefs=mixture_coefs)

    homo_dis = []
    for struct in structures:
        if simple_diploid:
            homo_dis.append(_inter_homolog_dis_via_simple_diploid(
                struct=struct, lengths=lengths))
        else:
            homo_dis.append(_inter_homolog_dis(struct=struct, lengths=lengths))

    return np.mean(homo_dis, axis=0)
