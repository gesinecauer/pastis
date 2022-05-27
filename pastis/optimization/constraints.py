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
from .utils_poisson import _euclidean_distance
from .counts import ambiguate_counts, _ambiguate_beta
from .likelihoods import skellam_nll, poisson_nll
from .poisson import relu_min  # FIXME temporary (for NDC)


class Constraint(object):
    """Compute loss for the given constraint.

    Prepares cand computes the loss function for the given constraint.

    # lambda_obj, lengths, multiscale_factor, params

    Parameters
    ----------
    lambda_obj : float
        Lambda that specifies how strongly constraint is applied during
        calculation of the entire objective.
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
    params : dict, optional
        Any parameters used for the calculation of constraint.

    Attributes
    ----------
    abbrev : str
        Three-letter abbreviation for constraint name.
    name : str
        Full name of constraint
    during_alpha_infer : bool
        Whether or not this constraint should be computed during inference of
        alpha.
    lambda_obj : float
        Lambda that specifies how strongly constraint is applied during
        calculation of the entire objective.
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
    params : dict, optional
        Any parameters used for the calculation of constraint.
    lowmem : bool, optional
        TODO
    """

    def __init__(self, lambda_obj, lengths, ploidy, multiscale_factor=1,
                 params=None, lowmem=False, mods=[]):
        self.abbrev = None
        self.name = None
        self.during_alpha_infer = None
        self.lambda_obj = lambda_obj
        self.lengths = lengths
        self.ploidy = ploidy
        self.multiscale_factor = multiscale_factor
        self.params = params
        self._var = None
        self._lowmem = lowmem
        self.mods = mods

    def __str__(self):
        out = [f"CONSTRAINT: {self.name},  LAMBDA={lambda_val:g}"]
        if self.params is None:
            return "\n".join(out)
        for name, val in self.params.items()
            label = f"\t\t\t{name} = "
            if isinstance(val, (np.ndarray, list)):
                out.append(label + np.array2string(
                    val, formatter={'float_kind': lambda x: "%.3g" % x},
                    prefix=" " * len(label), separator=", "))
            elif isinstance(val, float):
                out.append(f"{label}{val:g}")
            else:
                out.append(f"{label}{val}")
        return "\n".join(out)

    def check(self):
        """Check constraints object.

        Check that lambdas are greater than zero, and that necessary parameters
        are supplied."""
        pass

    def _setup(counts=None, bias=None, fullres_torm=None):
        """TODO"""
        pass

    def apply(self, struct, alpha, epsilon=None, counts=None, bias=None,
              inferring_alpha=False):

        """Apply constraint using given structure(s).

        Compute loss for the given constraint.

        Parameters
        ----------
        struct : ndarray
            3D chromatin structure(s) for which to compute the constraint.
        alpha : float
            Biophysical parameter of the transfer function used in converting
            counts to wish distances. If alpha is not specified, it will be
            inferred.
        epsilon : float, optional
            TODO
        counts : list of CountsMatrix subclass instances
            Preprocessed counts data.
        bias : array of float, optional
            Biases computed by ICE normalization.
        inferring_alpha : bool, optional
            A value of "True" indicates that the current optimization aims to
            infer alpha, rather than the structure.

        Returns
        -------
        constraint_obj
            Loss for constraint.
        """
        pass


class BeadChainConnectivity2019(Constraint):
    """TODO"""

    def __init__(self, lambda_obj, lengths, ploidy, multiscale_factor=1,
                 params=None, lowmem=False, mods=[]):
        self.abbrev = "bcc"
        self.name = "Bead-chain connectivity (2019)"
        self.during_alpha_infer = False
        self.lambda_obj = lambda_obj
        self.lengths = np.asarray(lengths)
        self.lengths_lowres = decrease_lengths_res(
            self.lengths, multiscale_factor=multiscale_factor)
        self.ploidy = ploidy
        self.multiscale_factor = multiscale_factor
        self.params = params
        self._var = None
        self._lowmem = lowmem
        self.mods = mods # TODO remove

        self.check()

    def check(self):
        if self.lambda_obj < 0:
            raise ValueError("Constraint lambda may not be < 0.")

        # Check params
        if self.params is not None and len(self.params) > 0:
            raise ValueError(f"{self.name} constraint may not have params")

    def _setup(counts=None, bias=None, fullres_torm=None):
        if self.lambda_obj <= 0:
            return
        if self._var is not None:
            return self._var
        row_nghbr = _neighboring_bead_indices(
            lengths=self.lengths, ploidy=self.ploidy,
            multiscale_factor=self.multiscale_factor)
        var = {'row_nghbr': row_nghbr}
        if not self._lowmem:
            self._var = var
        return var


    def apply(self, struct, alpha, epsilon=None, counts=None, bias=None,
              inferring_alpha=False):
        if self.lambda_obj == 0 or (
                inferring_alpha and not self.during_alpha_infer):
            return 0.
        var = self._setup(counts=counts, bias=bias)

        

        nghbr_dis = ag_np.sqrt((ag_np.square(
            struct[row_nghbr] - struct[row_nghbr + 1])).sum(axis=1))
        n_edges = nghbr_dis.shape[0]
        nghbr_dis_var = n_edges * ag_np.square(
            nghbr_dis).sum() / ag_np.square(nghbr_dis.sum())
        obj = nghbr_dis_var - 1.

        if ag_np.isnan(obj) or ag_np.isinf(obj):
            raise ValueError(f"{self.name} constraint objective is {obj}.")
        return obj


class HomologSeparating2019(Constraint):
    """TODO"""

    def __init__(self, lambda_obj, lengths, ploidy, multiscale_factor=1,
                 params=None, lowmem=False, mods=[]):
        self.abbrev = "hsc"
        self.name = "Homolog separating (2019)"
        self.during_alpha_infer = False
        self.lambda_obj = lambda_obj
        self.lengths = np.asarray(lengths)
        self.lengths_lowres = decrease_lengths_res(
            self.lengths, multiscale_factor=multiscale_factor)
        self.ploidy = ploidy
        self.multiscale_factor = multiscale_factor
        self.params = params
        self._var = None
        self._lowmem = lowmem
        self.mods = mods # TODO remove

        self.check()

    def check(self):
        if self.ploidy == 1 and self.lambda_obj > 0:
            raise ValueError(f"{self.name} constraint can not be applied to"
                             " haploid genomes.")
        if self.lambda_obj < 0:
            raise ValueError("Constraint lambda may not be < 0.")

        # Check params
        if self.params is None or "est_hmlg_sep" not in self.params:
            raise ValueError(f"{self.name} constraint is missing neccessary"
                             " params: est_hmlg_sep")
        if isinstance(self.params['est_hmlg_sep'], list):
            self.params['est_hmlg_sep'] = np.array(self.params['est_hmlg_sep'])
        if not isinstance(self.params['est_hmlg_sep'], (np.ndarray, float, int)):
            raise ValueError(f"{self.name} constraint param 'est_hmlg_sep' not"
                             " understood.")
        if isinstance(self.params['est_hmlg_sep'], np.ndarray):
            if self.params['est_hmlg_sep'].size not in (1, self.lengths.size):
                raise ValueError(f"{self.name} constraint param 'est_hmlg_sep'"
                                 " is not the correct size.")

    def _setup(counts=None, bias=None, fullres_torm=None):
        if self.lambda_obj <= 0:
            return
        if self._var is not None:
            return self._var

        if self.multiscale_factor != 1:
            fullres_per_lowres_bead = _count_fullres_per_lowres_bead(
                multiscale_factor=self.multiscale_factor, lengths=self.lengths,
                ploidy=self.ploidy, fullres_torm=fullres_torm)
            bead_weights = fullres_per_lowres_bead / self.multiscale_factor
        else:
            bead_weights = np.ones((self.lengths_lowres.sum() * self.ploidy,))
        torm = find_beads_to_remove(
            counts, lengths=self.lengths, ploidy=self.ploidy,
            multiscale_factor=self.multiscale_factor)
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
        bead_weights = bead_weights.reshape(-1, 1)

        var = {'bead_weights': bead_weights}
        if not self._lowmem:
            self._var = var
        return var

    def apply(self, struct, alpha, epsilon=None, counts=None, bias=None,
              inferring_alpha=False):
        if self.lambda_obj == 0 or (
                inferring_alpha and not self.during_alpha_infer):
            return 0.
        var = self._setup(counts=counts, bias=bias)

        # homo_sep = self._homolog_separation(struct)
        struct_bw = struct * np.repeat(var['bead_weights'], 3, axis=1)
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


        homo_sep_diff = self.params["est_hmlg_sep"] - homo_sep
        # homo_sep_diff = 1 - homo_sep / self.params["est_hmlg_sep"]  # with scaleR

        if 'hsc10' in self.mods or 'hsc20' in self.mods:
            if 'hsc20' in self.mods:
                hsc_cutoff = 0.2
            if 'hsc10' in self.mods:
                hsc_cutoff = 0.1
            hsc_r_hsc_cutoff = hsc_cutoff * self.params["est_hmlg_sep"]

            gt0 = homo_sep_diff > 0
            homo_sep_diff = homo_sep_diff.at[gt0].set(relu(
                homo_sep_diff[gt0] - hsc_r_hsc_cutoff))
            homo_sep_diff = homo_sep_diff.at[~gt0].set(-relu(
                -(homo_sep_diff[~gt0] + hsc_r_hsc_cutoff)))
        elif 'HSCnoRELU'.lower() not in self.mods:
            homo_sep_diff = relu(homo_sep_diff)
            raise ValueError("I thought we weren't doing RELU for HSC anymore")

        homo_sep_diff_sq = ag_np.square(homo_sep_diff)
        obj = ag_np.mean(homo_sep_diff_sq)

        if ag_np.isnan(obj) or ag_np.isinf(obj):
            raise ValueError(f"{self.name} constraint objective is {obj}.")
        return obj


class BeadChainConnectivity2021(Constraint):
    """TODO"""

    def __init__(self, lambda_obj, lengths, ploidy, multiscale_factor=1,
                 params=None, lowmem=False, mods=[]):
        self.abbrev = "bcc"
        self.name = "Bead-chain connectivity (2021)"
        self.during_alpha_infer = True
        self.lambda_obj = lambda_obj
        self.lengths = np.asarray(lengths)
        self.lengths_lowres = decrease_lengths_res(
            self.lengths, multiscale_factor=multiscale_factor)
        self.ploidy = ploidy
        self.multiscale_factor = multiscale_factor
        self.params = params
        self._var = None
        self._lowmem = lowmem
        self.mods = mods # TODO remove

        self.check()

    def check(self):
        if self.lambda_obj < 0:
            raise ValueError("Constraint lambda may not be < 0.")

    def _setup(counts=None, bias=None, fullres_torm=None):
        # FIXME adjust for different alpha; implement for multiscale; add bias
        if self.lambda_obj <= 0:
            return
        if self._var is not None:
            return self._var
        row_nghbr = _neighboring_bead_indices(
            lengths=self.lengths, ploidy=self.ploidy,
            multiscale_factor=self.multiscale_factor)

        counts = [c for c in counts if c.sum() != 0]
        beta = _ambiguate_beta(
            [c.beta for c in counts], counts=counts, lengths=self.lengths,
            ploidy=self.ploidy)
        counts_ambig = ambiguate_counts(
            counts=counts, lengths=self.lengths, ploidy=self.ploidy,
            exclude_zeros=True)
        row_nghbr_ambig = _neighboring_bead_indices(
            lengths=self.lengths, ploidy=1,
            multiscale_factor=self.multiscale_factor, counts=counts_ambig,
            include_torm_beads=False)

        if bias is not None:
            raise NotImplementedError("Implement for bias")  # TODO

        counts_ambig = decrease_counts_res(
            counts_ambig, multiscale_factor=self.multiscale_factor,
            lengths=self.lengths, ploidy=self.ploidy)
        nghbr_counts = counts_ambig.diagonal(k=1)[row_nghbr_ambig] / self.ploidy
        if self.multiscale_factor > 1:
            fullres_per_lowres_bead = _count_fullres_per_lowres_bead(
                multiscale_factor=self.multiscale_factor, lengths=self.lengths,
                ploidy=1, fullres_torm=self.fullres_torm)
            raise NotImplementedError("Implement for multiscale")  # TODO

        if (nghbr_counts == 0).sum() == 0:
            nghbr_dis = np.power(nghbr_counts / beta, 1 / alpha) # FIXME alpha
            mean = nghbr_dis.mean()
            std = nghbr_dis.std()
        else:
            mean, std = taylor_approx_ndc(
                nghbr_counts, beta=beta, alpha=alpha, order=1)
            # return mean, std, beta


        print(f"\n\n*** NDC:  μ={ndc_mu} σ={ndc_sigmamax} beta={ndc_beta} ***\n\n", flush=True)
        var = {'row_nghbr': row_nghbr}
        if not self._lowmem:
            self._var = var
        return var

    def apply(self, struct, alpha, epsilon=None, counts=None, bias=None,
              inferring_alpha=False):
        if self.lambda_obj == 0 or (
                inferring_alpha and not self.during_alpha_infer):
            return 0.
        var = self._setup(counts=counts, bias=bias)


        nghbr_dis = ag_np.sqrt((ag_np.square(
            struct[row_nghbr] - struct[row_nghbr + 1])).sum(axis=1))

        mad = ag_np.median(ag_np.absolute(
            nghbr_dis - ag_np.median(nghbr_dis)))
        sigma_tmp = 1.4826 * mad
        # sigma_tmp = ag_np.sqrt(ag_np.mean(ag_np.square(data - mu))) # old - i guess this didn't work

        if 'norm_dis_nomin' not in mods:
            sigma = relu_min(sigma_tmp, sigma_max)  # FIXME temporary
        else:
            sigma = sigma_tmp

        obj = ag_np.log(sigma) + ag_np.mean(
            ag_np.square(nghbr_dis - mu)) / (2 * ag_np.square(sigma))
        # if type(obj).__name__ == 'DeviceArray':
        #     data = nghbr_dis
        #     # TRUE:    μ=1.010   σMax=0.119      mean=1.004      sigma=0.098     sigma_tmp=0.098     std=0.098   obj=-1.8202
        #     # oldTRUE: μ=0.985     mean=1.004      sigma=0.098     std=0.098   obj=-1.8019
        #     # BCC1e1:         rmsd_intra=3.72   disterr_intra=39.9   disterr_interhmlg=70.8   ndv_nrmse=3.53  ()
        #     # norm_dis:       rmsd_intra=   disterr_intra=   disterr_interhmlg=   ndv_nrmse= ()
        #     # norm_dis_nomin: rmsd_intra=   disterr_intra=   disterr_interhmlg=   ndv_nrmse= ()
        #     print(f'μ={mu:.3f} \t σMax={sigma_max:.3f} \t mean={data.mean():.3f} \t sigma={sigma:.3f} \t sigma_tmp={sigma_tmp:.3f} \t std={data.std():.3f} \t obj={obj:.5g}')
        return obj


class BeadChainConnectivity2022(Constraint):
    """TODO"""

    def __init__(self, lambda_obj, lengths, ploidy, multiscale_factor=1,
                 params=None, lowmem=False, mods=[]):
        self.abbrev = "bcc"
        self.name = "Bead-chain connectivity (2022)"
        self.during_alpha_infer = True
        self.lambda_obj = lambda_obj
        self.lengths = np.asarray(lengths)
        self.lengths_lowres = decrease_lengths_res(
            self.lengths, multiscale_factor=multiscale_factor)
        self.ploidy = ploidy
        self.multiscale_factor = multiscale_factor
        self.params = params
        self._var = None
        self._lowmem = lowmem
        self.mods = mods # TODO remove

        self.check()

    def check(self):
        if self.ploidy == 1 and self.lambda_obj > 0:
            raise ValueError(f"{self.name} constraint can not be applied to"
                             " haploid genomes.")
        if self.lambda_obj < 0:
            raise ValueError("Constraint lambda may not be < 0.")

    def _setup(counts=None, bias=None, fullres_torm=None):
        if self.lambda_obj <= 0:
            return
        if self._var is not None:
            return self._var
        row_nghbr = _neighboring_bead_indices(
            lengths=self.lengths, ploidy=self.ploidy,
            multiscale_factor=self.multiscale_factor)
        var = {'row_nghbr': row_nghbr}
        if not self._lowmem:
            self._var = var
        return var

    def apply(self, struct, alpha, epsilon=None, counts=None, bias=None,
              inferring_alpha=False):
        if self.lambda_obj == 0 or (
                inferring_alpha and not self.during_alpha_infer):
            return 0.
        var = self._setup(counts=counts, bias=bias)


class HomologSeparating2022(Constraint):
    """TODO"""

    def __init__(self, lambda_obj, lengths, ploidy, multiscale_factor=1,
                 params=None, lowmem=False, mods=[]):
        self.abbrev = "hsc"
        self.name = "Homolog separating (2022)"
        self.during_alpha_infer = True
        self.lambda_obj = lambda_obj
        self.lengths = np.asarray(lengths)
        self.lengths_lowres = decrease_lengths_res(
            self.lengths, multiscale_factor=multiscale_factor)
        self.ploidy = ploidy
        self.multiscale_factor = multiscale_factor
        self.params = params
        self._var = None
        self._lowmem = lowmem
        self.mods = mods # TODO remove

        self.check()

    def check(self):
        if self.ploidy == 1 and self.lambda_obj > 0:
            raise ValueError(f"{self.name} constraint can not be applied to"
                             " haploid genomes.")
        if self.lambda_obj < 0:
            raise ValueError("Constraint lambda may not be < 0.")

    def _setup(counts=None, bias=None, fullres_torm=None):
        if self.lambda_obj <= 0:
            return
        if self._var is not None:
            return self._var
        
        var = {'': 1}
        if not self._lowmem:
            self._var = var
        return var

    def apply(self, struct, alpha, epsilon=None, counts=None, bias=None,
              inferring_alpha=False):
        if self.lambda_obj == 0 or (
                inferring_alpha and not self.during_alpha_infer):
            return 0.
        var = self._setup(counts=counts, bias=bias)







def prepare_constraints(counts, lengths, ploidy, multiscale_factor=1,
                        bcc_lambda=0., hsc_lambda=0., bcc_version='2019',
                        hsc_version='2019', counts_interchrom=None, hsc_r=None,
                        fullres_torm=None, verbose=True, mods=[]):
    # TODO remove
    if mods is None:
        mods = []
    elif isinstance(mods, str):
        mods = mods.lower().split('.')
    else:
        mods = [x.lower() for x in mods]




def _get_nghbr_diff_constraint(struct, counts, counts_interchrom, lengths,
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
            2 * lambda_interhmlg, data=counts_interchrom)
        if ('sep2' in mods):
            nll_nghbr_inter = poisson_nll(
                2 * ag_np.mean(lambda_nghbr_inter), data=counts_interchrom)

        if ('stricter' in mods):  # (if sep/sep2 & stricter, mu2 = 0)
            data = np.maximum(0, counts_nghbr - counts_interchrom / 2)
            nll1 = poisson_nll(lambda_sum1, data=data)
            nll2 = poisson_nll(lambda_sum2, data=data)
        elif ('sep2' in mods):
            data = counts_nghbr
            nll1 = poisson_nll(lambda_sum1, data=data)
            nll2 = poisson_nll(lambda_sum2, data=data)
        else:
            data = np.maximum(0, counts_nghbr - counts_interchrom / 2)
            nll1 = skellam_nll(mu1=lambda_sum1, mu2=mu2, data=data, mods=mods)
            nll2 = skellam_nll(mu1=lambda_sum2, mu2=mu2, data=data, mods=mods)
    else:
        data = np.maximum(0, counts_nghbr - counts_interchrom)
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

    # TODO check the below
    # FIXME NaN beads should be MEAN of nghbr_counts, not zero!!
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



def taylor_approx_ndc(x, beta=1, alpha=-3, order=1):
    x = x / beta
    x_mean = np.mean(x)
    x_var = np.var(x)

    fx_mean = np.power(x_mean, 1 / alpha)
    fx_var = 1 / np.power(alpha, 2) * np.power(
        x_mean, (2 - 2 * alpha) / alpha) * x_var

    if order == 2:  # FIXME
        tmp = (1 - alpha) / (2 * np.square(alpha)) * np.power(
            x_mean, (1 - 2 * alpha) / alpha)
        fx_mean = fx_mean + tmp * x_var
        fx_var = fx_var + np.square(tmp) * (np.var(
            np.square(x)) - 4 * np.square(x_mean) * x_var)

    fx_std = np.sqrt(fx_var)
    return fx_mean, fx_std


def _neighboring_bead_indices(lengths, ploidy, multiscale_factor=1,
                              counts=None, include_torm_beads=True):
    """Return row & col of neighboring beads, along a homolog of a chromosome.
    """

    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
    nbeads = lengths_lowres.sum() * ploidy

    row_nghbr = np.arange(nbeads - 1, dtype=int)

    # Optionally remove beads for which there is no counts data
    if not include_torm_beads:
        if counts is None:
            raise ValueError("Counts must be inputted if including torm beads.")
        torm = find_beads_to_remove(
            counts, lengths=lengths, ploidy=ploidy,
            multiscale_factor=multiscale_factor)
        where_torm = np.where(torm)[0]
        nghbr_dis_mask = (~np.isin(row_nghbr, where_torm)) & (
            ~np.isin(row_nghbr + 1, where_torm))
        row_nghbr = row_nghbr[nghbr_dis_mask]

    # Remove if "neighbor" beads are actually on different chromosomes
    # or homologs
    bins = np.tile(lengths_lowres, ploidy).cumsum()
    same_bin = np.digitize(row_nghbr, bins) == np.digitize(row_nghbr + 1, bins)

    row_nghbr = row_nghbr[same_bin]

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
    for i in range(lengths.size):
        end += lengths[i]
        if np.isnan(homo1[begin:end, 0]).sum() == lengths[i] or np.isnan(
                homo2[begin:end, 0]).sum() == lengths[i]:
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
    for i in range(lengths.size):
        end += lengths[i]
        if np.isnan(struct[begin:end, 0]).sum() < lengths[i]:
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
