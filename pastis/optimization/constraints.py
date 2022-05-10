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

from .multiscale_optimization import decrease_lengths_res, decrease_counts_res
from .multiscale_optimization import _count_fullres_per_lowres_bead
from .utils_poisson import find_beads_to_remove, _intra_counts_mask
from .counts import ambiguate_counts, _ambiguate_beta
from .likelihoods import skellam_nll, poisson_nll


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
            if "hsc" in constraint_params and constraint_params["hsc"] is not None:
                constraint_params["hsc"] = np.array(
                    constraint_params["hsc"]).flatten().astype(float)
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
        if self.lambdas["hsc"]:
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

        self.row_nghbr = None
        if self.lambdas["bcc"] or self.lambdas["ndc"]:
            self.row_nghbr = _neighboring_bead_indices(
                lengths=lengths, ploidy=ploidy,
                multiscale_factor=multiscale_factor)

        self.ndc_mu = self.ndc_beta = None  # FIXME ndc_beta needs to be reset with alpha infer
        if self.lambdas["ndc"]:
            self.ndc_mu, self.ndc_sigmamax, self.ndc_beta = _prep_nghbr_dis_constraint(
                counts=counts, lengths=lengths, ploidy=ploidy,
                multiscale_factor=multiscale_factor, bias=None,  # FIXME add bias
                fullres_torm=fullres_torm)
            print(f"\n\n*** NDC:  μ={self.ndc_mu} σ={self.ndc_sigmamax} beta={self.ndc_beta} ***\n\n")

        # ============================ FIXME hsc_new temp
        self.counts = None
        if 'hsc_new' in self.mods:
            self.counts = counts
        # ============================ FIXME hsc_new temp

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
        lambda_defaults = {"bcc": 0., "hsc": 0., "ndc": 0.}
        lambda_all = lambda_defaults
        if self.lambdas is not None:
            for k, v in self.lambdas.items():
                if k not in lambda_all:
                    raise ValueError(
                        "constraint_lambdas key not recognized - %s" % k)
                elif v is not None:
                    lambda_all[k] = float(v) # np.float32(v) # FIXME revert
        self.lambdas = lambda_all

        params_defaults = {"hsc": None, "ndc": None, 'bcc': 1, }
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

        if self.lambdas["hsc"] and self.ploidy == 1:
            raise ValueError("Homolog-separating constraint can not be"
                             " applied to haploid genome.")

        # Print constraints
        constraint_names = {"bcc": "bead chain connectivity",
                            "hsc": "homolog-separating",
                            "ndc": "neighbor distance"}
        lambda_to_print = {k: v for k, v in self.lambdas.items() if v != 0}
        if verbose and len(lambda_to_print) > 0:
            for constraint, lambda_val in lambda_to_print.items():
                print("CONSTRAINT: %s lambda = %.2g" % (
                    constraint_names[constraint], lambda_val), flush=True)
                if constraint not in self.params:
                    continue
                if constraint == "hsc":
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

    def apply(self, structures, alpha=None, epsilon=None, bias=None,
              inferring_alpha=False, mixture_coefs=None):
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
            col_nghbr = self.row_nghbr + 1
            for struct, gamma in zip(structures, mixture_coefs):
                nghbr_dis = ag_np.sqrt((ag_np.square(
                    struct[self.row_nghbr] - struct[col_nghbr])).sum(axis=1))
                n_edges = nghbr_dis.shape[0]
                nghbr_dis_var = n_edges * ag_np.square(
                    nghbr_dis).sum() / ag_np.square(nghbr_dis.sum())
                obj_bcc = nghbr_dis_var - 1.
                obj_bcc = ag_np.power(obj_bcc, self.params["bcc"])
                obj['bcc'] = obj['bcc'] + gamma * self.lambdas['bcc'] * obj_bcc
        if self.lambdas["ndc"]:
            col_nghbr = self.row_nghbr + 1
            for struct, gamma in zip(structures, mixture_coefs):
                if 'redirect_ndc' in self.mods:
                    continue
                nghbr_dis = ag_np.sqrt((ag_np.square(
                    struct[self.row_nghbr] - struct[col_nghbr])).sum(axis=1))
                obj_ndc = _get_nghbr_dis_constraint(
                    nghbr_dis, mu=self.ndc_mu, sigma_max=self.ndc_sigmamax,
                    beta=self.ndc_beta, alpha=alpha, mods=self.mods)
                obj['ndc'] = obj['ndc'] + gamma * self.lambdas['ndc'] * obj_ndc
        if self.lambdas["hsc"] and not inferring_alpha:
            for struct, gamma in zip(structures, mixture_coefs):
                # ============================ FIXME hsc_new temp
                if 'hsc_new' in self.mods and 'hsc_mean_mse' in self.mods:
                    hsc = _get_nghbr_ratio_constraint(
                        struct, counts=self.counts,
                        counts_interchrom_mean=self.params["hsc"][0],
                        lengths=self.lengths, bias=bias,
                        multiscale_factor=self.multiscale_factor, alpha=alpha,
                        mods=self.mods)
                    obj["hsc"] = obj["hsc"] + gamma * self.lambdas["hsc"] * hsc
                    continue
                elif 'hsc_new' in self.mods:
                    hsc = _get_nghbr_diff_constraint(
                        struct=struct, counts=self.counts,
                        counts_interchrom_mean=self.params["hsc"][0],
                        lengths=self.lengths, bias=bias,
                        multiscale_factor=self.multiscale_factor, alpha=alpha,
                        epsilon=epsilon, ndc_lambda=self.lambdas["ndc"],
                        mods=self.mods)
                    obj["hsc"] = obj["hsc"] + gamma * self.lambdas["hsc"] * hsc
                    continue
                # ============================ FIXME hsc_new temp

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
                    raise ValueError("I thought we weren't doing RELU for HSC anymore")

                homo_sep_diff_sq = ag_np.square(homo_sep_diff)
                if 'sum_not_mean' in self.mods:
                    hsc = ag_np.sum(homo_sep_diff_sq)  # TODO fix on main branch: mean not sum!
                else:
                    hsc = ag_np.mean(homo_sep_diff_sq)  # TODO fix on main branch: mean not sum!
                obj["hsc"] = obj["hsc"] + gamma * self.lambdas["hsc"] * hsc

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


def _euclidean_distance(struct, row, col):  # FIXME move to utils
    dis_sq = (ag_np.square(struct[row] - struct[col])).sum(axis=1)
    return ag_np.sqrt(dis_sq)


def _get_nghbr_diff_constraint(struct, counts, counts_interchrom_mean, lengths,
                               bias, multiscale_factor, alpha, epsilon=None,
                               ndc_lambda=0, mods=[]):

    row_nghbr = _neighboring_bead_indices(
        lengths=lengths, ploidy=2, multiscale_factor=multiscale_factor)
    num_nghbr_hmlg = int(row_nghbr.size / 2)
    row_nghbr_h1 = row_nghbr[:num_nghbr_hmlg]
    row_nghbr_h2 = row_nghbr[num_nghbr_hmlg:]
    dis_nghbr_h1h1 = _euclidean_distance(
        struct, row=row_nghbr_h1, col=row_nghbr_h1 + 1)
    dis_nghbr_h2h2 = _euclidean_distance(
        struct, row=row_nghbr_h2, col=row_nghbr_h2 + 1)
    dis_nghbr_h1h2 = _euclidean_distance(
        struct, row=row_nghbr_h1, col=row_nghbr_h2 + 1)
    dis_nghbr_h2h1 = _euclidean_distance(
        struct, row=row_nghbr_h2, col=row_nghbr_h1 + 1)

    # Get inter-homolog distances
    n = int(struct.shape[0] / 2)
    row, col = (x.flatten() for x in np.indices((n, n)))  # TODO what about excluded rows/cols?
    if ('sep2' in mods) and ('trim_inter' in mods):
        mask_inter_nghbr = ((col == row + 1) & np.isin(row, row_nghbr_h1)) | (
            (col + 1 == row) & np.isin(col, row_nghbr_h1))
        row = row[~mask_inter_nghbr]
        col = col[~mask_inter_nghbr]
    dis_interhmlg = _euclidean_distance(struct, row=row, col=col + n)
    if bias is None or np.all(bias == 1):
        bias_interhmlg = 1
    else:
        bias_interhmlg = bias.ravel()[row] * bias.ravel()[col]

    # Get counts corresponding to distances between neighoring beads
    counts_nghbr, beta, bias_nghbr = _get_nghbr_counts(
        counts, lengths=lengths, multiscale_factor=multiscale_factor, bias=bias)

    # # FIXME!! using fake counts
    # from sklearn.metrics import euclidean_distances
    # c = np.power(euclidean_distances(struct), alpha) * beta
    # c = c[:n, :n] + c[n:, n:] + c[:n, n:] + c[n:, :n]
    # counts_nghbr = c[row_nghbr_h1, row_nghbr_h1 + 1]
    # #####

    if multiscale_factor > 1:
        raise NotImplementedError

    lambda_nghbr_inter = beta * bias_nghbr * (ag_np.power(
        dis_nghbr_h1h2, alpha) + ag_np.power(dis_nghbr_h2h1, alpha))
    lambda_interhmlg = 2 * ag_np.mean(beta * bias_interhmlg * ag_np.power(
        dis_interhmlg, alpha))

    if ('stricter' in mods):
        lambda_sum1 = beta * bias_nghbr * 2 * ag_np.power(
            dis_nghbr_h1h1, alpha)
        lambda_sum2 = beta * bias_nghbr * 2 * ag_np.power(
            dis_nghbr_h2h2, alpha)
        if ('sep' in mods) or ('sep2' in mods):
            mu2 = np.array(0.)
        else:
            mu2 = lambda_interhmlg
    else:
        lambda_sum1 = beta * bias_nghbr * 2 * ag_np.power(
            dis_nghbr_h1h1, alpha) + lambda_nghbr_inter
        lambda_sum2 = beta * bias_nghbr * 2 * ag_np.power(
            dis_nghbr_h2h2, alpha) + lambda_nghbr_inter
        if ('sep2' in mods):
            mu2 = np.array(0.)
        elif ('sep' in mods):
            mu2 = lambda_nghbr_inter
        else:
            mu2 = lambda_nghbr_inter + lambda_interhmlg
        if ('mean_internghbr' in mods):  # not relevant for sep2
            mu2 = ag_np.mean(mu2)

    nll_interhmlg = nll_nghbr_inter = 0
    if ('sep' in mods) or ('sep2' in mods):
        nll_interhmlg = poisson_nll(
            2 * lambda_interhmlg, data=counts_interchrom_mean)
        if ('sep2' in mods):
            nll_nghbr_inter = poisson_nll(
                2 * ag_np.mean(lambda_nghbr_inter), data=counts_interchrom_mean)

        if ('stricter' in mods):  # (if sep/sep2 & stricter, mu2 = 0)
            data = np.maximum(0, counts_nghbr - counts_interchrom_mean / 2)
            nll1 = poisson_nll(lambda_sum1, data=data)
            nll2 = poisson_nll(lambda_sum2, data=data)
        elif ('sep2' in mods):
            data = counts_nghbr
            nll1 = poisson_nll(lambda_sum1, data=data)
            nll2 = poisson_nll(lambda_sum2, data=data)
        else:
            data = np.maximum(0, counts_nghbr - counts_interchrom_mean / 2)
            nll1 = skellam_nll(mu1=lambda_sum1, mu2=mu2, data=data, mods=mods)
            nll2 = skellam_nll(mu1=lambda_sum2, mu2=mu2, data=data, mods=mods)
    else:
        data = np.maximum(0, counts_nghbr - counts_interchrom_mean)
        nll1 = skellam_nll(mu1=lambda_sum1, mu2=mu2, data=data, mods=mods)
        nll2 = skellam_nll(mu1=lambda_sum2, mu2=mu2, data=data, mods=mods)

    nll_nghbr = (nll1 + nll2) / 2
    if nll_interhmlg != 0 and nll_nghbr_inter != 0:
        nll_inter = (nll_interhmlg + nll_nghbr_inter) / 2
    else:
        nll_inter = nll_interhmlg + nll_nghbr_inter
    if 'redirect_ndc' in mods and ndc_lambda != 0:
        nll_inter = nll_inter * ndc_lambda
    obj = nll_inter + nll_nghbr

    if type(nll1).__name__ == 'DeviceArray':
        nghbr = beta * bias_nghbr * ag_np.power(np.concatenate([dis_nghbr_h1h1, dis_nghbr_h2h2]), alpha)
        mu1 = ag_np.concatenate([lambda_sum1, lambda_sum2])
        mu_diff = ag_np.concatenate([lambda_sum1 - mu2, lambda_sum2 - mu2])
        print(f"nghbr={nghbr.mean():.1f}\tinter={lambda_interhmlg / 2:.3f}\tnghbr_i={(lambda_nghbr_inter / 2).mean():.3f}\t  μ1={mu1.mean():.1f}\tμ2={mu2.mean():.2f}\tμ1-μ2={mu_diff.mean():.1f}\tdata={data.mean():.1f}\t  obj_i={nll_interhmlg:.2f}\tobj_ni={nll_nghbr_inter:.2f}\tobj_n={nll_nghbr:.3f}\tobj={obj:.3g}", flush=True)
    return obj


def _get_nghbr_ratio_constraint(struct, counts, counts_interchrom_mean, lengths,
                                bias, multiscale_factor, alpha, mods=[]):
    counts_nghbr, _, bias_nghbr = _get_nghbr_counts(
        counts, lengths=lengths, multiscale_factor=multiscale_factor, bias=bias)
    nrc_k = (counts_nghbr.mean() / 2) / counts_interchrom_mean

    row_nghbr = _neighboring_bead_indices(
        lengths=lengths, ploidy=2, multiscale_factor=multiscale_factor)
    row_nghbr_h1 = row_nghbr[int(row_nghbr.size / 2):]
    row_nghbr_h2 = row_nghbr[:int(row_nghbr.size / 2)]
    row_nghbr_swap_hmlg = np.concatenate([row_nghbr_h1, row_nghbr_h2])

    nghbr_dis = _euclidean_distance(
        struct, row=row_nghbr, col=row_nghbr + 1)

    if "all_interh" in mods:
        n = int(struct.shape[0] / 2)
        row, col = np.indices((n, n))
        nghbr_dis_interhmlg = _euclidean_distance(
            struct, row=row.flatten(), col=col.flatten() + n)
    else:
        nghbr_dis_interhmlg = _euclidean_distance(
            struct, row=row_nghbr, col=row_nghbr_swap_hmlg + 1)

    ratio_dis_alpha = ag_np.power(nghbr_dis, alpha) / ag_np.mean(
        ag_np.power(nghbr_dis_interhmlg, alpha))
    mse = ag_np.mean(ag_np.square(ratio_dis_alpha / nrc_k - 1))

    if type(mse).__name__ == 'DeviceArray':
        nghbr = ag_np.power(nghbr_dis, alpha)
        n_hmlg = ag_np.power(nghbr_dis_interhmlg, alpha)
        print(f"mean(nghbr)={nghbr.mean():.3f}  var(nghbr)={nghbr.var():.3f}    mean(n_hmlg)={n_hmlg.mean():.3f}  var(n_hmlg)={n_hmlg.var():.3f}    mean(ratio)={ratio_dis_alpha.mean():.3f}  var(ratio)={ratio_dis_alpha.var():.3f}    mse={mse:.3f}", flush=True)
    return mse


def _get_nghbr_counts(counts, lengths, multiscale_factor, bias):
    """TODO
    """

    if multiscale_factor != 1:
        raise NotImplementedError("or maybe it is implemented. who am i to say??")

    # counts = [c for c in counts if c.sum() != 0]
    # beta = _ambiguate_beta(
    #     [c.beta for c in counts], counts=counts, lengths=lengths, ploidy=2)
    # counts_ambig = ambiguate_counts(
    #     counts=counts, lengths=lengths, ploidy=2, exclude_zeros=True)
    counts_ambig = [c for c in counts if c.name == 'ambig']
    if len(counts_ambig) == 1:
        counts_ambig = counts_ambig[0]
    elif len(counts_ambig) == 0:
        raise ValueError("0 ambiguous counts matrices inputted.")
    else:
        raise ValueError(">1 ambiguous counts matrix inputted.")

    row_nghbr = _neighboring_bead_indices(
        lengths=lengths, ploidy=1, multiscale_factor=multiscale_factor)

    # FIXME check the below
    idx = np.where(np.isin(counts_ambig.row, row_nghbr) & (
        counts_ambig.col == counts_ambig.row + 1))[0]
    counts_nghbr = np.zeros_like(row_nghbr)
    counts_nghbr[np.isin(row_nghbr, counts_ambig.row)] = counts_ambig.data[idx]

    # row_nghbr_non0 = counts_ambig.row[counts_ambig.col == counts_ambig.row + 1]
    # # Remove if "neighbor" beads are actually on different chromosomes
    # # or homologs
    # lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
    # bins = np.tile(lengths_lowres, ploidy).cumsum()
    # same_bin = np.digitize(row_nghbr_non0, bins) == np.digitize(row_nghbr_non0 + 1, bins)
    # row_nghbr_non0 = row_nghbr_non0[same_bin]
    # # blah
    # counts_nghbr_non0 = 


    if bias is None or np.all(bias == 1):
        bias_nghbr = 1
    else:
        bias_nghbr = bias.ravel()[row_nghbr] * bias.ravel()[row_nghbr + 1]

    return counts_nghbr, counts_ambig.beta, bias_nghbr


from .poisson import relu_min  # FIXME temporary


def _get_nghbr_dis_constraint(nghbr_dis, mu, sigma_max, beta, alpha, mods=[]):
    if 'ndc_ratio' in mods:
        n_per_hmlg = int(nghbr_dis.size / 2)
        ratio = nghbr_dis[:n_per_hmlg] / nghbr_dis[n_per_hmlg:]
        mse = ag_np.mean(ag_np.square(ratio - 1))
        # if type(mse).__name__ == 'DeviceArray':
        #     print(f"mean(ratio)={ratio.mean():.3f}  var(ratio)={ratio.var():.3f}  mse={mse:.3f}")
        return mse

    mad = ag_np.median(ag_np.absolute(
        nghbr_dis - ag_np.median(nghbr_dis)))
    sigma_tmp = 1.4826 * mad

    if 'norm_dis_nomin' not in mods:
        sigma = relu_min(sigma_tmp, sigma_max)  # FIXME temporary
    else:
        sigma = sigma_tmp

    obj_ndc = ag_np.log(sigma) + ag_np.mean(
        ag_np.square(nghbr_dis - mu)) / (2 * ag_np.square(sigma))

    # if type(obj_ndc).__name__ == 'DeviceArray':
    #     data = nghbr_dis
    #     # TRUE:    μ=1.010   σMax=0.119      mean=1.004      sigma=0.098     sigma_tmp=0.098     std=0.098   obj=-1.8202
    #     # oldTRUE: μ=0.985     mean=1.004      sigma=0.098     std=0.098   obj=-1.8019
    #     # BCC1e1:         rmsd_intra=3.72   disterr_intra=39.9   disterr_interhmlg=70.8   ndv_nrmse=3.53  ()
    #     # norm_dis:       rmsd_intra=   disterr_intra=   disterr_interhmlg=   ndv_nrmse= ()
    #     # norm_dis_nomin: rmsd_intra=   disterr_intra=   disterr_interhmlg=   ndv_nrmse= ()
    #     print(f'μ={mu:.3f} \t σMax={sigma_max:.3f} \t mean={data.mean():.3f} \t sigma={sigma:.3f} \t sigma_tmp={sigma_tmp:.3f} \t std={data.std():.3f} \t obj={obj_ndc:.5g}')

    return obj_ndc


def _get_nghbr_dis_constraint_old(nghbr_dis, mu, sigma_max, beta, alpha, mods=[]):
    data = beta * ag_np.power(nghbr_dis, alpha)

    # mad = ag_np.median(ag_np.absolute(
    #     data - ag_np.median(data)))

    sigma_tmp = ag_np.sqrt(ag_np.mean(ag_np.square(data - mu)))
    sigma = relu_min(sigma_tmp, sigma_max)  # FIXME temporary

    obj_ndc = ag_np.log(sigma) + ag_np.mean(
        ag_np.square(data - mu)) / (2 * ag_np.square(sigma))

    if type(obj_ndc).__name__ == 'DeviceArray':
        # TRUE:  [[ μ=7.939      σMax=2.648 ]]   data=7.937      sigma=2.455     sigma_tmp=2.455     std=2.455   nghbr_dis.mean=1.004    nghbr_dis.std=0.098     obj=1.3981
        print(f'[[ μ={mu:.3f} \t σMax={sigma_max:.3f} ]] \t data={data.mean():.3f} \t sigma={sigma:.3f} \t sigma_tmp={sigma_tmp:.3f} \t std={ag_np.std(data):.3f} \t nghbr_dis.mean={nghbr_dis.mean():.3f} \t nghbr_dis.std={nghbr_dis.std():.3f} \t obj={obj_ndc:.5g}')

    return obj_ndc


def _get_nghbr_dis_constraint_oldest(nghbr_dis, mods=['kurt']):
    # nghbr_dis_scaled = nghbr_dis / ag_np.median(nghbr_dis)
    nghbr_dis_scaled = nghbr_dis / ag_np.mean(nghbr_dis)

    if 'kurt' in mods or 'skew' in mods:
        if 'kurt' in mods and 'skew' in mods:
            raise ValueError("Can't do kurt + skew")
        z = (nghbr_dis_scaled - ag_np.mean(nghbr_dis_scaled)) / ag_np.std(
            nghbr_dis_scaled)
        if 'kurt' in mods:
            obj_ndc = ag_np.mean(ag_np.power(z, 4)) - 3
            # if type(obj_ndc).__name__ == 'DeviceArray':
            #     print(f'kurt\t{(obj_ndc + 3):.3f}\t{ag_np.square(obj_ndc):.3f}')
            # obj_ndc = relu(obj_ndc)
            obj_ndc = ag_np.square(obj_ndc)
        elif 'skew' in mods:
            obj_ndc = ag_np.mean(ag_np.power(z, 3))
    elif 'old' in mods:
        nghbr_dis_var = ag_np.var(nghbr_dis_scaled)
        nghbr_dis_mad_sq = ag_np.square(ag_np.median(ag_np.absolute(
            nghbr_dis_scaled - ag_np.median(nghbr_dis_scaled))))
        obj_ndc = nghbr_dis_var - nghbr_dis_mad_sq
        obj_ndc = relu(obj_ndc)
        # if type(obj_ndc).__name__ == 'DeviceArray':
        #     print(f"std: {nghbr_dis_std:.6f},\tvar:{ag_np.square(nghbr_dis_std):.6f},\tmad:{nghbr_dis_mad:6f}", flush=True)
    else:
        nghbr_dis_std = ag_np.std(nghbr_dis_scaled)
        nghbr_dis_mad = ag_np.median(ag_np.absolute(
            nghbr_dis_scaled - ag_np.median(nghbr_dis_scaled)))
        # std = 1.4826 * mad
        obj_ndc = nghbr_dis_std / nghbr_dis_mad - 1.4826

    return obj_ndc


def _prep_nghbr_dis_constraint(counts, lengths, ploidy, multiscale_factor, bias,
                               fullres_torm, alpha=-3):
    """TODO
    """

    counts = [c for c in counts if c.sum() != 0]

    beta = _ambiguate_beta(
        [c.beta for c in counts], counts=counts, lengths=lengths, ploidy=ploidy)
    counts_ambig = ambiguate_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, exclude_zeros=True)

    row_nghbr = _neighboring_bead_indices(
        lengths=lengths, ploidy=1, multiscale_factor=multiscale_factor,
        counts=counts_ambig, include_torm_beads=False)

    if bias is not None:
        raise NotImplementedError("Implement for bias")  # TODO

    counts_ambig = decrease_counts_res(
        counts_ambig, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)

    nghbr_counts = counts_ambig.diagonal(k=1)[row_nghbr] / ploidy

    if multiscale_factor > 1:
        fullres_per_lowres_bead = _count_fullres_per_lowres_bead(
            multiscale_factor=multiscale_factor, lengths=lengths,
            ploidy=1, fullres_torm=fullres_torm)
        raise NotImplementedError("Implement for multiscale")  # TODO

    if (nghbr_counts == 0).sum() == 0:
        nghbr_dis = np.power(nghbr_counts / beta, 1 / alpha)
        return nghbr_dis.mean(), nghbr_dis.std(), beta
    else:
        mean, std = taylor_approx_ndc(
            nghbr_counts, beta=beta, alpha=alpha, order=1)
        return mean, std, beta


def taylor_approx_ndc(x, beta=1, alpha=-3, order=1):
    x = x / beta
    x_mean = np.mean(x)
    x_var = np.var(x)

    fx_mean = np.power(x_mean, 1 / alpha)
    fx_var = 1 / np.power(alpha, 2) * np.power(
        x_mean, (2 - 2 * alpha) / alpha) * x_var

    if order == 2:
        tmp = (1 - alpha) / (2 * np.square(alpha)) * np.power(
            x_mean, (1 - 2 * alpha) / alpha)
        fx_mean = fx_mean + tmp * x_var
        fx_var = fx_var + np.square(tmp) * (np.var(
            np.square(x)) - 4 * np.square(x_mean) * x_var)

    fx_std = np.sqrt(fx_var)
    return fx_mean, fx_std


# def _remove_intermolecule_idx(row, col, lengths, ploidy, multiscale_factor):
#     """TODO"""
#     lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)

#     bins = np.tile(lengths_lowres, ploidy).cumsum()
#     same_bin = np.digitize(row, bins) == np.digitize(col, bins)

#     row = row[same_bin]
#     col = col[same_bin]

#     return row, col


def _neighboring_bead_indices(lengths, ploidy, multiscale_factor, counts=None,
                              include_torm_beads=True):
    """Return row & col of neighboring beads, along a homolog of a chromosome.
    """

    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
    nbeads = lengths_lowres.sum() * ploidy

    row_nghbr = np.arange(nbeads - 1, dtype=int)
    col_nghbr = np.arange(1, nbeads, dtype=int)

    # Optionally remove beads for which there is no counts data
    if not include_torm_beads:
        if counts is None:
            raise ValueError("Counts must be inputted if including torm beads.")
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

    return row_nghbr


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
