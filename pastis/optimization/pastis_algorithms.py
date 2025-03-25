from __future__ import print_function
import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import os
import numpy as np
import pandas as pd

from .utils_poisson import _setup_jax
_setup_jax()
import jax.numpy as jnp

from .utils_poisson import _print_code_header, _get_output_files
from .utils_poisson import _output_subdir, _load_infer_param, _format_structures
from .utils_poisson import distance_between_homologs, distance_between_molecules
from .counts import preprocess_counts, _ambiguate_beta, ambiguate_counts
from .counts import _set_initial_beta, _get_counts_ambiguity, _disambiguate_beta
from .counts import _debias_counts
from .initialization import initialize
from .callbacks import Callback
from .constraints import prep_constraints, _neighboring_bead_indices
from .constraints import get_counts_interchrom, HomologSeparating2019
from .poisson import PastisPM, _convergence_criteria
from .multiscale_optimization import _choose_max_multiscale_factor
from .multiscale_optimization import decrease_lengths_res, decrease_bias_res
from .multiscale_optimization import get_epsilon_from_struct, decrease_counts_res
from .multiscale_optimization import toy_struct_max_epilon
from ..io.read import load_data


def _infer_draft(counts, lengths, ploidy, outdir=None, alpha=None, seed=0,
                 filter_threshold=0.04, normalize=True, bias=None, beta=None,
                 multiscale_rounds=1, beta_init=None,
                 init='mds', max_iter=30000, factr=1e7, pgtol=1e-05,
                 hsc_lambda=0, hsc_version='2019', est_hmlg_sep=None,
                 hsc_min_beads=5, callback_freq=None,
                 callback_fxns=None, reorienter=None,
                 multiscale_reform=False, alpha_true=None,
                 struct_true=None, input_weight=None, exclude_zeros=False,
                 chrom_full=None, chrom_subset=None, bias_per_hmlg=False,
                 mixture_coefs=None, verbose=True, mods=[]):
    """Infer draft 3D structures with PASTIS via Poisson model."""

    infer_draft_lowres = est_hmlg_sep is None and hsc_lambda > 0 and str(
        hsc_version) == '2019'

    if not infer_draft_lowres:
        return est_hmlg_sep, True

    if ploidy == 1:
        raise ValueError("Can not apply homolog-separating constraint"
                         " to haploid data.")

    counts, bias, lengths, _, _, _, struct_true = load_data(
        counts=counts, lengths_full=lengths, ploidy=ploidy,
        chrom_full=chrom_full, chrom_subset=chrom_subset,
        filter_threshold=filter_threshold, normalize=normalize, bias=bias,
        struct_true=struct_true, bias_per_hmlg=bias_per_hmlg, verbose=False)
    bias_per_hmlg = bias_per_hmlg and ploidy == 2 and len(counts) == 1 and set(
        counts[0].shape) == {lengths.sum() * ploidy}

    multires_factor_draft = _choose_max_multiscale_factor(
        lengths=lengths, min_beads=hsc_min_beads)
    _print_code_header(
        ['INFERRING DRAFT STRUCTURE',
            f'Low resolution ({multires_factor_draft}x)'],
        max_length=60, blank_lines=2, verbose=verbose and infer_draft_lowres)

    # Setup
    if mixture_coefs is None:
        mixture_coefs = [1]
    if outdir is None:
        lowres_outdir = None
    else:
        lowres_outdir = os.path.join(outdir, 'struct_draft_lowres')

    if beta is None:
        _, beta = _set_initial_beta(
            counts, lengths=lengths, ploidy=ploidy, bias=bias,
            exclude_zeros=exclude_zeros, bias_per_hmlg=bias_per_hmlg)
    elif isinstance(beta, (float, int)):
        beta = [beta]
    ua_index = [i for i in range(len(
        counts)) if counts[i].shape == (
        lengths.sum() * ploidy, lengths.sum() * ploidy)]
    if len(ua_index) > 1:
        raise ValueError("Only input one matrix of unambiguous counts."
                         " Please pool unambiguous counts before"
                         " inputting.")

    struct_true_draft = struct_true
    init_draft = init
    beta_init_draft = beta_init
    if len(ua_index) == 1:
        ploidy_draft = 2
        counts_draft = [counts[ua_index[0]]]
        beta_draft = [beta[ua_index[0]]]
    else:
        # Inferring "simplified" diploid structures, where we assume homologs of
        # a given chromosome to have identical conformations and to completely
        # overlap one another in 3D. We essentially treat the ambiguated
        # diploid data as if it were haploid.
        if lengths.size == 1:
            raise ValueError("Please input more than one chromosome to"
                             " estimate est_hmlg_sep from ambiguous data.")
        ploidy_draft = 1


        # Convert counts, betas, bias, & struct_true to "simplified" diploid
        beta_ambig = _ambiguate_beta(
            beta, counts=counts, lengths=lengths, ploidy=2)
        beta_draft = 2 * beta_ambig
        if beta_init_draft is not None:
            beta_init_draft = beta_init_draft * 2
        counts_draft = [ambiguate_counts(
            counts=counts, lengths=lengths, ploidy=2)]
        if struct_true_draft is not None:
            struct_true_draft = [x.reshape(-1, 3) for x in struct_true_draft]
            struct_true_draft = [np.nanmean(  # FIXME is this even correct?
                [x[:lengths.sum()], x[lengths.sum():]],
                axis=0) for x in struct_true_draft]
            # struct_true_draft = np.nanmean(  # FIXME is this even correct?
            #     [struct_true_draft[:int(struct_true_draft.shape[0] / 2)],
            #      struct_true_draft[int(struct_true_draft.shape[0] / 2):]],
            #     axis=0)
        if np.all(bias == 1):
            bias is None
        if bias_per_hmlg and bias is not None:
            bias = np.mean([bias[:lengths.sum()], bias[lengths.sum():]])


        # Convert initialization to "simplified" diploid (if necessary)
        if isinstance(init_draft, list):
            init_draft = jnp.concatenate(init_draft)._value
        if isinstance(init_draft, jnp.ndarray):
            init_draft = init_draft._value
        if isinstance(init_draft, np.ndarray) and (
                init_draft.size == lengths.sum() * 2 * 3 * len(mixture_coefs)):
            # The intialization is a full-res diploid 3D structure. However,
            # we are currently inferring "simplified" diploid structures.
            init_draft = _format_structures(
                init_draft, lengths=lengths, ploidy=2,
                mixture_coefs=mixture_coefs)
            init_draft = [np.nanmean(  # FIXME is this even correct?
                [x[:int(x.shape[0] / 2)], x[int(x.shape[0] / 2):]],
                axis=0) for x in init_draft]
            init_draft = np.concatenate(init_draft)

    struct_draft_lowres, infer_param_lowres = infer_at_alpha(
        counts=counts_draft, outdir=lowres_outdir,
        lengths=lengths, ploidy=ploidy_draft, alpha=alpha, seed=seed,
        filter_threshold=filter_threshold, normalize=normalize, bias=bias,
        beta=beta_draft, beta_init=beta_init_draft,
        multiscale_factor=multires_factor_draft,
        init=init_draft, max_iter=max_iter, factr=factr, pgtol=pgtol,
        callback_fxns=callback_fxns, callback_freq=callback_freq,
        reorienter=reorienter, multiscale_reform=multiscale_reform,
        alpha_true=alpha_true, struct_true=struct_true_draft,
        input_weight=input_weight, exclude_zeros=exclude_zeros, null=False,
        bias_per_hmlg=bias_per_hmlg, mixture_coefs=mixture_coefs,
        verbose=verbose, mods=mods)
    if not infer_param_lowres['converged']:
        return None, False

    if ploidy_draft == 2:
        est_hmlg_sep = distance_between_homologs(
            struct_draft_lowres, lengths=lengths,
            multiscale_factor=multires_factor_draft,
            mixture_coefs=mixture_coefs)
    else:
        est_hmlg_sep = [np.mean(distance_between_molecules(
            struct_draft_lowres, lengths=lengths, ploidy=ploidy_draft,
            multiscale_factor=multires_factor_draft,
            mixture_coefs=mixture_coefs))]
    if verbose:
        print("Estimated distances between homolog centers of mass:"
              " " + " ".join([f'{x:.2g}' for x in est_hmlg_sep]), flush=True)
        if struct_true is not None:
            true_hmlg_sep = np.mean([distance_between_homologs(
                structures=x, lengths=lengths) for x in struct_true], axis=0)
            print("   > TRUE distances between homolog centers of mass:"
                  " " + " ".join([f'{x:.2g}' for x in true_hmlg_sep]),
                  flush=True)

    _print_code_header(
        ['Draft inference complete', 'INFERRING STRUCTURE'],
        max_length=60, blank_lines=2, verbose=verbose)

    return est_hmlg_sep, True


def _prep_inference(counts, lengths, ploidy, outdir='', alpha=None, seed=0,
                    filter_threshold=0.04, normalize=True, bias=None, alpha_init=None,
                    max_alpha_loop=20, beta=None, multiscale_factor=1,
                    epsilon_min=1e-2, epsilon_max=1e6,
                    beta_init=None, init='mds', max_iter=30000,
                    factr=1e7, pgtol=1e-05, alpha_factr=1e12,
                    bcc_lambda=0, hsc_lambda=0, bcc_version='2019',
                    hsc_version='2019', data_interchrom=None, est_hmlg_sep=None,
                    hsc_min_beads=5, hsc_perc_diff=None, excluded_counts=None,
                    callback_freq=None, callback_fxns=None, reorienter=None,
                    multiscale_reform=False,
                    alpha_true=None, struct_true=None, input_weight=None,
                    exclude_zeros=False, null=False, chrom_full=None, chrom_subset=None,
                    bias_per_hmlg=False, mixture_coefs=None,
                    outfiles=None, verbose=True, mods=[]):
    """TODO"""

    # PREPARE COUNTS OBJECTS
    counts, struct_nan, fullres_struct_nan = preprocess_counts(
        counts=counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, exclude_zeros=exclude_zeros,
        beta=beta, bias=bias, input_weight=input_weight, verbose=verbose,
        excluded_counts=excluded_counts, multiscale_reform=multiscale_reform,
        bias_per_hmlg=bias_per_hmlg, mods=mods)
    bias_per_hmlg = bias_per_hmlg and ploidy == 2 and len(counts) == 1 and set(
        counts[0].shape) == {lengths.sum() * ploidy}

    # PRINT INFERENCE INFORMATION
    if verbose:
        if outfiles is not None:
            print(f"OUTPUT: {outfiles['struct_infer']}", flush=True)
        print('BETA: ' + ', '.join(
            [f'{c.ambiguity}={c.beta:.3g}' for c in counts]), flush=True)
        if alpha is None:
            print(f'ALPHA: to be inferred, init={alpha_init:.3g}', flush=True)
        else:
            print(f'ALPHA: {alpha:.3g}', flush=True)

    # REDUCE RESOLUTION OF BIAS, IF NEEDED
    if multiscale_factor > 1 and not multiscale_reform:
        bias = decrease_bias_res(
            bias, multiscale_factor=multiscale_factor, lengths=lengths,
            bias_per_hmlg=bias_per_hmlg)

    # INITIALIZATION
    random_state = np.random.RandomState(seed)
    struct_init = initialize(
        counts=counts, lengths=lengths, init=init, ploidy=ploidy,
        random_state=random_state,
        alpha=(alpha_init if alpha is None else alpha),
        bias=bias, multiscale_factor=multiscale_factor,
        reorienter=reorienter, mixture_coefs=mixture_coefs,
        struct_true=struct_true, bias_per_hmlg=bias_per_hmlg, verbose=verbose,
        mods=mods)
    if multiscale_reform and multiscale_factor != 1:
        if epsilon_min == epsilon_max:
            epsilon = epsilon_max
        elif epsilon_min * 1.1 >= min(1, epsilon_max):
            epsilon = random_state.uniform(
                low=epsilon_min, high=min(1, epsilon_max))
        else:
            epsilon = random_state.uniform(
                low=epsilon_min * 1.1, high=min(1, epsilon_max))
        if verbose:
            print(f"INITIALIZATION: {epsilon=:.3g}", flush=True)
    else:
        epsilon = None

    # HOMOLOG-SEPARATING CONSTRAINT
    if hsc_lambda > 0 and reorienter is not None and reorienter.reorient:
        est_hmlg_sep = distance_between_homologs(
            structures=reorienter.struct_init, lengths=lengths,
            mixture_coefs=mixture_coefs)

    # SETUP CONSTRAINTS
    constraints = prep_constraints(
        lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, multiscale_reform=multiscale_reform,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda, bcc_version=bcc_version,
        hsc_version=hsc_version, data_interchrom=data_interchrom,
        est_hmlg_sep=est_hmlg_sep, hsc_perc_diff=hsc_perc_diff,
        fullres_struct_nan=fullres_struct_nan, bias_per_hmlg=bias_per_hmlg,
        verbose=verbose, mods=mods)
    [x.setup(counts=counts, bias=bias) for x in constraints]  # For jax jit
    constraints = tuple(constraints)

    # SETUP CALLBACKS
    if callback_fxns is None:
        callback_fxns = {}
    if callback_freq is None:
        callback_freq = {}
    callback = Callback(
        lengths=lengths, ploidy=ploidy, counts=counts, bias=bias,
        multiscale_factor=multiscale_factor,
        multiscale_reform=multiscale_reform,
        directory=outdir, seed=seed, struct_true=struct_true,
        alpha_true=alpha_true, constraints=constraints, beta_init=beta_init,
        fullres_struct_nan=fullres_struct_nan, mixture_coefs=mixture_coefs,
        reorienter=reorienter, **callback_freq, **callback_fxns,
        verbose=verbose, mods=mods)

    return (counts, bias, struct_nan, struct_init, constraints, callback,
            epsilon, ploidy)


def set_epsilon_bounds(counts, lengths, ploidy, alpha=None, seed=0, bias=None,
                       beta=None, multiscale_factor=1, beta_init=None,
                       multiscale_reform=False, epsilon_min=1e-2,
                       epsilon_max=1e6, epsilon_prev=None, alpha_true=None,
                       struct_true=None, exclude_zeros=False,
                       bias_per_hmlg=False, max_attempt=5, verbose=True, mods=[]):
    """TODO"""

    est_epsilon_max = est_epsilon_x2 = None
    if not (multiscale_reform and multiscale_factor > 1):
        return (epsilon_min, epsilon_max), (est_epsilon_max, est_epsilon_x2)

    if 'epsilon_true' in mods:
        if struct_true is None:
            raise ValueError("Need true struct for true epsilon.")
        epsilon_true, _ = get_epsilon_from_struct(
            struct_true, lengths=lengths, ploidy=ploidy,
            multiscale_factor=multiscale_factor,
            mixture_coefs=[1] * len(struct_true), verbose=False)
        print(f"\tUsing true epsilon = {epsilon_true:.3g}", flush=True)
        return (epsilon_true, epsilon_true), (est_epsilon_max, est_epsilon_x2)

    if verbose:
        print(f"Setting bounds for epsilon at {multiscale_factor}x resolution"
              f"\n\tCurrent bounds = [{epsilon_min:.3g}, {epsilon_max:.3g}]",
              flush=True)

    # if beta is None:
    #     mean_nghbr_dis = 1
    # else:
    #     if beta_init is None:
    #         beta_init = _ambiguate_beta(
    #             beta, counts=counts, lengths=lengths, ploidy=ploidy)
    #     beta_nghbr_1, _ = _set_initial_beta(
    #         counts, lengths=lengths, ploidy=ploidy, bias=bias,
    #         exclude_zeros=exclude_zeros, bias_per_hmlg=bias_per_hmlg)
    #     mean_nghbr_dis = beta_nghbr_1 / beta_init
    if beta is None:
        beta_ambig, _ = _set_initial_beta(
            counts, lengths=lengths, ploidy=ploidy, bias=bias,
            exclude_zeros=exclude_zeros, bias_per_hmlg=bias_per_hmlg)
    else:
        beta_ambig = _ambiguate_beta(
            beta, counts=counts, lengths=lengths, ploidy=ploidy)
    if beta_init is None:
        beta_init = beta_ambig
        mean_nghbr_dis = 1
    else:
        mean_nghbr_dis = beta_ambig / beta_init
        print(f"\t>>> {mean_nghbr_dis=:.3g}", flush=True)  # TODO remove

    bias_per_hmlg = bias_per_hmlg and ploidy == 2 and len(counts) == 1 and set(
        counts[0].shape) == {sum(lengths) * ploidy}

    est_epsilon_x2 = mean_nghbr_dis / np.sqrt(6)

    if multiscale_factor > 2 and 'emin' in mods:
        if verbose and est_epsilon_x2 > epsilon_min:
            print(f"\tEpsilon lower bound updated: ↑ from {epsilon_min:.3g}"
                  f" to {est_epsilon_x2:.3g}", flush=True)
        epsilon_min = max(epsilon_min, est_epsilon_x2)

    if multiscale_factor == 2 and ('eps2' in mods or 'eps2mm' in mods):
        # print(f"\t{beta_ambig=:.3g}\t{beta_init=:.3g}\t{mean_nghbr_dis=:.3g}")  # TODO remove
        if est_epsilon_x2 < epsilon_max:
            if verbose:
                print(f"\tEpsilon upper bound updated: ↓ from {epsilon_max:.3g} to"
                      f" {est_epsilon_x2:.3g}", flush=True)
            epsilon_max = min(epsilon_max, est_epsilon_x2)
            if 'eps2mm' in mods:
                if verbose and est_epsilon_x2 > epsilon_min:
                    print(f"\tEpsilon lower bound updated: ↑ from {epsilon_min:.3g}"
                          f" to {est_epsilon_x2:.3g}", flush=True)
                epsilon_min = max(epsilon_min, est_epsilon_x2)

    elif 'espiral' in mods:
        if alpha is None:
            raise ValueError("Must provide alpha for epsmaxd")

        counts_norm = [_debias_counts(
            c, bias=bias, ploidy=ploidy, lengths=lengths,
            bias_per_hmlg=bias_per_hmlg) for c in counts]

        counts_ambig = ambiguate_counts(
            counts_norm, lengths=lengths, ploidy=ploidy)
        counts_ambig_lowres = decrease_counts_res(
            counts_ambig, multiscale_factor=multiscale_factor,
            lengths=lengths, ploidy=1, mean=True)

        row_nghbr_ambig = _neighboring_bead_indices(
            lengths, ploidy=1, counts=counts_ambig,
            include_struct_nan_beads=False)
        row_nghbr_ambig_lowres = _neighboring_bead_indices(
            lengths, ploidy=1, multiscale_factor=multiscale_factor,
            counts=counts_ambig_lowres, include_struct_nan_beads=False)

        nghbr_dis_alpha_fullres = counts_ambig.tocsr(
            ).diagonal(k=1)[row_nghbr_ambig] / ploidy / beta_ambig
        nghbr_dis_alpha_lowres = counts_ambig_lowres.tocsr(
            ).diagonal(k=1)[row_nghbr_ambig_lowres] / ploidy / beta_ambig

        if (nghbr_dis_alpha_fullres == 0).sum() > 0:
            nghbr_dis_fullres = np.power(
                nghbr_dis_alpha_fullres.mean(), 1 / alpha)
            # print(f"\t>>> {multiscale_factor}x: {nghbr_dis_fullres=:.3g}", flush=True)  # TODO remove
        else:
            nghbr_dis_fullres = np.power(
                nghbr_dis_alpha_fullres, 1 / alpha).mean()

        if 'no_ndlow' in mods:
            nghbr_dis_lowres = None
        elif (nghbr_dis_alpha_lowres == 0).sum() > 0:
            nghbr_dis_lowres = np.power(
                nghbr_dis_alpha_lowres.mean(), 1 / alpha)
            print(f"\t>>> {multiscale_factor}x: {nghbr_dis_lowres=:.3g}", flush=True)  # TODO remove
        else:
            nghbr_dis_lowres = np.power(
                nghbr_dis_alpha_lowres, 1 / alpha).mean()

        random_state = np.random.RandomState(seed)
        est_epsilon_max, attempt = None, 0
        while est_epsilon_max is None and attempt < max_attempt:
            est_epsilon_max, _ = toy_struct_max_epilon(
                multiscale_factor, nghbr_dis_fullres=nghbr_dis_fullres,
                nghbr_dis_lowres=nghbr_dis_lowres,
                epsilon_prev=epsilon_prev if ('eprev' in mods or 'eprev2' in mods) else None,
                random_state=random_state, verbose=verbose)
            max_attempt += 1
        if verbose and est_epsilon_max < epsilon_max:
            print(f"\tEpsilon upper bound updated: ↓ from {epsilon_max:.3g} to"
                  f" {est_epsilon_max:.3g}", flush=True)
        epsilon_max = min(epsilon_max, est_epsilon_max)

    elif 'epsmax' in mods:
        struct_epsilon_max = mean_nghbr_dis * np.concatenate([
            np.arange(multiscale_factor).reshape(-1, 1),
            np.zeros((multiscale_factor, 2))], axis=1)
        est_epsilon_max, _ = get_epsilon_from_struct(
            struct_epsilon_max, lengths=multiscale_factor, ploidy=1,
            multiscale_factor=multiscale_factor, verbose=False)
        if verbose and est_epsilon_max < epsilon_max:
            print(f"\tEpsilon upper bound updated: ↓ from {epsilon_max:.3g} to"
                  f" {est_epsilon_max:.3g}", flush=True)
        epsilon_max = min(epsilon_max, est_epsilon_max)

    if verbose and struct_true is not None:
        epsilon_true, _ = get_epsilon_from_struct(
            struct_true, lengths=lengths, ploidy=ploidy,
            multiscale_factor=multiscale_factor,
            mixture_coefs=[1] * len(struct_true), verbose=False)
        print(f"\tTrue epsilon = {epsilon_true:.3g}", flush=True)
        if epsilon_true > epsilon_max:
            print(f"\t  ↳ WARNING: true epsilon ({epsilon_true:.3g}) is above"
                  f" upper bound ({epsilon_max:.3g})", flush=True)
        if epsilon_true < epsilon_min:
            print(f"\t  ↳ WARNING: true epsilon ({epsilon_true:.3g}) is below"
                  f" lower bound ({epsilon_min:.3g})", flush=True)

    return (epsilon_min, epsilon_max), (est_epsilon_max, est_epsilon_x2)


def infer_at_alpha(counts, lengths, ploidy, outdir='', alpha=None, seed=0,
                   filter_threshold=0.04, normalize=True, bias=None, alpha_init=None,
                   max_alpha_loop=20, beta=None, multiscale_factor=1,
                   alpha_loop=None, update_alpha=False,
                   beta_init=None, init='mds', max_iter=30000,
                   factr=1e7, pgtol=1e-05, alpha_factr=1e12,
                   bcc_lambda=0, hsc_lambda=0, bcc_version='2019', hsc_version='2019',
                   data_interchrom=None, est_hmlg_sep=None, hsc_min_beads=5,
                   hsc_perc_diff=None, excluded_counts=None,
                   callback_freq=None, callback_fxns=None, reorienter=None,
                   multiscale_reform=False, epsilon_min=1e-2, epsilon_max=1e6,
                   epsilon_prev=None, alpha_true=None, struct_true=None,
                   input_weight=None, exclude_zeros=False, null=False,
                   chrom_full=None, chrom_subset=None, bias_per_hmlg=False,
                   mixture_coefs=None, verbose=True,
                   mods=[]):
    """Infer 3D structures with PASTIS via Poisson model.

    Optimize 3D structure from Hi-C contact counts data for diploid
    organisms. Optionally perform multiscale optimization during inference.

    Parameters
    ----------
    counts : list of array or coo_matrix
        Counts data without normalization or filtering.
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    outdir : str, optional
        Directory in which to save results.
    alpha : float, optional
        Biophysical parameter of the transfer function used in converting
        counts to wish distances. If alpha is not specified, it will be
        inferred.
    seed : int, optional
        Random seed used when generating the starting point in the
        optimization.
    normalize : bool, optional
        Perform ICE normalization on the counts prior to optimization.
        Normalization is reccomended.
    filter_threshold : float, optional
        Ratio of non-zero beads to be filtered out. Filtering is
        reccomended.
    alpha_init : float, optional
        For PM2, the initial value of alpha to use.
    max_alpha_loop : int, optional
        For PM2, Number of times alpha and structure are inferred.
    beta : array_like of float, optional
        Scaling parameter that determines the size of the structure, relative to
        each counts matrix. There should be one beta per counts matrix. If None,
        the optimal beta will be estimated.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
    multiscale_rounds : int, optional
        The number of resolutions at which a structure should be inferred
        during multiscale optimization. Values of 1 or 0 disable multiscale
        optimization. # TODO move to infer fxn
    beta_init : float, optional
        TODO update
    init : optional
        Method by which to initialize the structure.
    max_iter : int, optional
        Maximum number of iterations per optimization.
    factr : float, optional
        factr for scipy's L-BFGS-B, alters convergence criteria.
    pgtol : float, optional
        pgtol for scipy's L-BFGS-B, alters convergence criteria.
    alpha_factr : float, optional
        factr for convergence criteria of joint alpha/structure inference.
    bcc_lambda : float, optional
        Lambda of the bead chain connectivity constraint.
    hsc_lambda : float, optional
        For diploid organisms: lambda of the homolog-separating
        constraint.
    data_interchrom : int or float, optional
        TODO add bcc_version, hsc_version, & data_interchrom
    est_hmlg_sep : list of float, optional
        For diploid organisms: hyperparameter of the homolog-separating
        constraint specificying the expected distance between homolog
        centers of mass for each chromosome.
    hsc_min_beads : int, optional
        For diploid organisms: number of beads in the low-resolution
        structure from which `est_hmlg_sep` is estimated.
    hsc_perc_diff : float, optional
        For diploid organisms: TODO
    excluded_counts : str, optional
        Whether to exclude inter- or intra-chromosomal counts from optimization.

    Returns
    -------
    struct_ : array_like of float of shape (lengths.sum() * ploidy, 3)
        3D structure resulting from the optimization.
    infer_param : dict
        A few of the parameters used in inference or generated by inference.
        Keys: 'alpha', 'beta', 'est_hmlg_sep', 'obj', and 'seed'. TODO
    """

    if outdir is not None:
        outfiles = _get_output_files(outdir, seed=seed)
        if os.path.isfile(outfiles['struct_infer']) or os.path.isfile(
                outfiles['struct_nonconv']):
            infer_param = _load_infer_param(outfiles['infer_param'])
            if ('alpha_converged' in infer_param) and (
                    infer_param['alpha_converged'] is not None) and (
                    not infer_param['alpha_converged']):
                if verbose:
                    print("OPTIMIZATION DID NOT CONVERGE\n"
                          f"{infer_param['alpha_conv_desc']}\n", flush=True)
                struct_ = None
            elif os.path.isfile(outfiles['struct_infer']):
                if verbose:
                    if ('alpha_converged' in infer_param) and (
                            infer_param['alpha_converged'] is not None):
                        alpha_desc = f" (alpha={infer_param['alpha']:.3g})"
                    else:
                        alpha_desc = ""
                    print(f'CONVERGED{alpha_desc}\n', flush=True)
                struct_ = np.loadtxt(outfiles['struct_infer'])
            elif os.path.isfile(outfiles['struct_nonconv']):
                if verbose:
                    print("OPTIMIZATION DID NOT CONVERGE\n"
                          f"{infer_param['conv_desc']}\n", flush=True)
                struct_ = None
            return struct_, infer_param
    else:
        outfiles = None

    # LOAD DATA
    counts, bias, lengths, chromosomes, _, _, struct_true = load_data(
        counts=counts, lengths_full=lengths, ploidy=ploidy,
        chrom_full=chrom_full, chrom_subset=chrom_subset,
        filter_threshold=filter_threshold, normalize=normalize, bias=bias,
        struct_true=struct_true, bias_per_hmlg=bias_per_hmlg, verbose=verbose)
    bias_per_hmlg = bias_per_hmlg and ploidy == 2 and len(counts) == 1 and set(
        counts[0].shape) == {lengths.sum() * ploidy}

    # SET BOUNDS ON EPSILON
    (epsilon_min, epsilon_max), _ = set_epsilon_bounds(
        counts, lengths=lengths, ploidy=ploidy, alpha=alpha, bias=bias,
        beta=beta, multiscale_factor=multiscale_factor, beta_init=beta_init,
        multiscale_reform=multiscale_reform, epsilon_min=epsilon_min,
        epsilon_max=epsilon_max, epsilon_prev=epsilon_prev,
        alpha_true=alpha_true, struct_true=struct_true,
        exclude_zeros=exclude_zeros, bias_per_hmlg=bias_per_hmlg,
        verbose=verbose, mods=mods)
    if 'spiral4all' in mods:
        tmpname = os.path.basename(outfiles['dir'].split('/infer_')[0])
        print("\n\nname\tres\tepsilon_true\test_epsilon_max\test_epsilon_x2\tmax_ok", flush=True)
        epsilon_prev_tmp = {64: None, 32: None, 16: None, 8: None, 4: None}
        if ('eprev' in mods or 'eprev2' in mods):
            epsilon_prev_tmp = {64: None, 32: 2.09, 16: 1.64, 8: 1.35, 4: 0.739}
        for x in [32, 16, 8, 4, 2]:
            epsilon_true, _ = get_epsilon_from_struct(
                struct_true, lengths=lengths, ploidy=ploidy, verbose=False,
                multiscale_factor=x, mixture_coefs=[1] * len(struct_true))
            _, (est_epsilon_max, est_epsilon_x2) = set_epsilon_bounds(
                counts, lengths=lengths, ploidy=ploidy, alpha=alpha, bias=bias,
                beta=beta, multiscale_factor=x, beta_init=beta_init,
                multiscale_reform=True, epsilon_min=epsilon_min,
                epsilon_max=epsilon_max, epsilon_prev=epsilon_prev_tmp[x * 2],
                alpha_true=alpha_true, struct_true=struct_true,
                exclude_zeros=exclude_zeros, bias_per_hmlg=bias_per_hmlg,
                verbose=True, mods=mods)
            est_epsilon_max = np.nan if est_epsilon_max is None else est_epsilon_max
            est_epsilon_x2 = np.nan if est_epsilon_x2 is None else est_epsilon_x2
            max_ok = ' ' if (np.isnan(est_epsilon_max) or est_epsilon_max >= epsilon_true) else '!!!'
            print(f"{tmpname}\t{x}\t{epsilon_true:g}\t{est_epsilon_max:g}\t{est_epsilon_x2:g}\t{max_ok}", flush=True)
        print('\n', flush=True)
        exit(0)

    # PREP FOR INFERENCE
    prepped = _prep_inference(
        counts, lengths=lengths, ploidy=ploidy, outdir=outdir, alpha=alpha,
        seed=seed, filter_threshold=filter_threshold, normalize=normalize,
        bias=bias, alpha_init=alpha_init, max_alpha_loop=max_alpha_loop,
        beta=beta, multiscale_factor=multiscale_factor, epsilon_min=epsilon_min,
        epsilon_max=epsilon_max, beta_init=beta_init, init=init,
        max_iter=max_iter, factr=factr, pgtol=pgtol, alpha_factr=alpha_factr,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda, bcc_version=bcc_version,
        hsc_version=hsc_version, data_interchrom=data_interchrom,
        est_hmlg_sep=est_hmlg_sep, hsc_min_beads=hsc_min_beads,
        hsc_perc_diff=hsc_perc_diff, excluded_counts=excluded_counts,
        callback_freq=callback_freq, callback_fxns=callback_fxns, reorienter=reorienter,
        multiscale_reform=multiscale_reform,
        alpha_true=alpha_true, struct_true=struct_true, input_weight=input_weight,
        exclude_zeros=exclude_zeros, null=null,
        chrom_full=chrom_full, chrom_subset=chrom_subset,
        bias_per_hmlg=bias_per_hmlg, mixture_coefs=mixture_coefs,
        outfiles=outfiles, verbose=verbose, mods=mods)
    (counts, bias, struct_nan, struct_init, constraints, callback, epsilon,
        ploidy) = prepped

    # INFER STRUCTURE
    pm = PastisPM(
        counts=counts, lengths=lengths, ploidy=ploidy, alpha=alpha,
        init=struct_init, bias=bias, constraints=constraints,
        callback=callback, multiscale_factor=multiscale_factor, epsilon=epsilon,
        epsilon_bounds=[epsilon_min, epsilon_max], alpha_init=alpha_init,
        max_alpha_loop=max_alpha_loop, max_iter=max_iter, factr=factr,
        pgtol=pgtol, alpha_factr=alpha_factr, reorienter=reorienter,
        null=null, bias_per_hmlg=bias_per_hmlg,
        mixture_coefs=mixture_coefs, verbose=verbose, mods=mods)
    if 'skip_struct' in mods and ploidy == 2 and alpha_loop == 1:
        pm.struct_ = struct_init
        pm.converged_ = True
    else:
        pm.fit_structure(alpha_loop=alpha_loop)

    # OPTIONALLY RE-INFER ALPHA
    alpha_converged_ = alpha_conv_desc_ = None
    if update_alpha and multiscale_factor == 1:
        _print_code_header([
            "Jointly inferring structure & alpha",
            f"Inferring ALPHA #{alpha_loop}"], max_length=80,
            verbose=verbose and alpha_loop is not None)
        pm.fit_alpha(alpha_loop=alpha_loop)
        alpha_converged_ = pm.alpha_converged_
        alpha_conv_desc_ = pm.alpha_conv_desc_

    # GET RESCALING FACTOR: RE-SCALES STRUCTURE TO MATCH ORIGINAL BETA
    beta_current = _ambiguate_beta(
        pm.beta_, counts=counts, lengths=lengths, ploidy=ploidy)
    if beta_init is None:
        rescale_by = None
    else:
        # rescale_by = beta_current / beta_init
        rescale_by = np.power(beta_current / beta_init, 1 / pm.alpha_)
        # print(f">>>>>> alpha={pm.alpha_:.3g}\t{rescale_by=:.3g}\t{beta_init=:.3g}\t{beta_current=:.3g}")

    # SAVE RESULTS
    struct_ = pm.struct_.reshape(-1, 3)
    infer_param = {
        'alpha': pm.alpha_, 'beta': pm.beta_, 'obj': pm.obj_, 'seed': seed,
        'converged': pm.converged_, 'conv_desc': pm.conv_desc_,
        'time': pm.time_elapsed_, 'epsilon': pm.epsilon_,
        'alpha_obj': pm.alpha_obj_, 'alpha_converged': alpha_converged_,
        'alpha_conv_desc': alpha_conv_desc_, 'alpha_loop': alpha_loop,
        'rescale_by': rescale_by, 'est_hmlg_sep': est_hmlg_sep}
    if reorienter is not None and reorienter.reorient:
        infer_param['orient'] = pm.orientation_.flatten()

    if outfiles is not None:
        os.makedirs(outfiles['dir'], exist_ok=True)
        with open(outfiles['infer_param'], 'w') as f:
            for k, v in infer_param.items():
                if isinstance(v, (np.ndarray, list, tuple)):
                    f.write(f"{k}\t{' '.join(map(str, v))}\n")
                elif v is not None:
                    f.write(f'{k}\t{v}\n')
        if reorienter is not None and reorienter.reorient:
            np.savetxt(outfiles['reorient'], pm.orientation_)
        if pm.converged_:
            np.savetxt(outfiles['struct_infer'], struct_)
            if pm.log_ is not None and len(pm.log_) > 0:
                pd.DataFrame(pm.log_).to_csv(
                    outfiles['log'], sep='\t', index=False)
        else:
            np.savetxt(outfiles['struct_nonconv'], struct_)

        bead_desc = pd.DataFrame()
        bead_desc['homolog'] = np.repeat(np.arange(ploidy), lengths.sum())
        bead_desc['chromosome'] = np.tile(
            np.repeat(chromosomes, lengths), ploidy)
        bead_desc['bin'] = np.concatenate(
            [np.arange(x) for x in np.tile(lengths, ploidy)])
        mask = np.full(lengths.sum() * ploidy, True)
        mask[struct_nan] = False
        bead_desc['mask'] = mask
        pd.DataFrame(bead_desc).to_csv(
            os.path.join(outdir, 'bead_desc.txt'), sep='\t', index=False)

    if pm.converged_:
        return struct_, infer_param
    else:
        return None, infer_param


def infer(counts, lengths, ploidy, outdir='', alpha=None, seed=0,
          filter_threshold=0.04, normalize=True, bias=None, alpha_init=None,
          max_alpha_loop=20, beta=None, multiscale_rounds=1,
          alpha_loop=None, update_alpha=False, prev_alpha_obj=None,
          beta_init=None, init='mds', max_iter=30000,
          factr=1e7, pgtol=1e-05, alpha_factr=1e12, infer_alpha_intra=True,
          struct_at_final_alpha=True,
          bcc_lambda=0, hsc_lambda=0, bcc_version='2019', hsc_version='2019',
          data_interchrom=None, est_hmlg_sep=None, hsc_min_beads=5,
          hsc_perc_diff=None, excluded_counts=None,
          callback_freq=None, callback_fxns=None, reorienter=None,
          multiscale_reform=False, epsilon_min=1e-2, epsilon_max=1e6,
          alpha_true=None, struct_true=None,
          input_weight=None, exclude_zeros=False, null=False,
          chrom_full=None, chrom_subset=None, bias_per_hmlg=False,
          mixture_coefs=None, verbose=True, mods=[]):
    """TODO"""

    # if 'exclude_inter' in mods:
    #     excluded_counts = 'inter-chromosomal'

    # SETUP
    if alpha is None:
        update_alpha = True
        if alpha_init is None:
            random_state = np.random.RandomState(seed)
            if 'alpha_init' in mods:
                alpha_init = random_state.uniform(low=-3, high=-2)
                if verbose:
                    print(f"INITIALIZATION: alpha={alpha_init=:.3g}",
                          flush=True)
            else:
                alpha_init = random_state.uniform(low=-4, high=-1)
    if update_alpha and alpha_loop is None:
        alpha_loop = 1
    first_alpha_loop = update_alpha and alpha_loop == 1
    if multiscale_rounds <= 0:
        multiscale_rounds = 1
    if alpha is None:
        alpha = alpha_init
    # Get inter-chromosomal counts for homolog separating constraint (2022)
    if hsc_lambda > 0 and hsc_version == '2022' and (
            data_interchrom is None or isinstance(data_interchrom, str)):
        data_interchrom = get_counts_interchrom(
            counts, lengths=lengths, ploidy=ploidy,
            filter_threshold=filter_threshold, normalize=normalize,
            bias=bias, multiscale_reform=multiscale_reform,
            multiscale_rounds=multiscale_rounds,
            data_interchrom=data_interchrom, bias_per_hmlg=bias_per_hmlg,
            verbose=verbose, mods=mods)

    # OPTIONALLY INFER ALPHA
    init_ = init
    est_hmlg_sep_ = est_hmlg_sep
    if first_alpha_loop:
        # Load data
        loaded = load_data(
            counts=counts, lengths_full=lengths, ploidy=ploidy,
            chrom_full=chrom_full, chrom_subset=chrom_subset,
            filter_threshold=filter_threshold, normalize=normalize, bias=bias,
            struct_true=struct_true, bias_per_hmlg=bias_per_hmlg, verbose=verbose)
        (_counts, _bias, lengths_subset, chrom_subset_, lengths, chrom_full,
            _struct_true) = loaded

        # Get initial beta
        if beta_init is None:
            if beta is not None:
                beta_init = _ambiguate_beta(
                    beta, counts=_counts, lengths=lengths_subset, ploidy=ploidy)
            else:
                beta_init, beta = _set_initial_beta(
                    _counts, lengths=lengths_subset, ploidy=ploidy, bias=_bias,
                    exclude_zeros=exclude_zeros, bias_per_hmlg=bias_per_hmlg)
        elif beta is None:
            beta = _disambiguate_beta(
                beta_init, counts=_counts, lengths=lengths_subset,
                ploidy=ploidy, bias=_bias, bias_per_hmlg=bias_per_hmlg)

        init_alpha_1chrom = 'init_alpha_1chrom' in mods and (
            lengths_subset.size > 1)
        infer_alpha_intra = infer_alpha_intra and lengths_subset.size > 1 and (
            excluded_counts != 'inter-chromosomal')

        ambiguities = [_get_counts_ambiguity(
            c.shape, nbeads=lengths_subset.sum() * ploidy) for c in _counts]
        infer_alpha_intra_mol = 'infer_alpha_intra_mol' in mods and (
            lengths_subset.size * ploidy > 1) and (
            excluded_counts != 'inter-molecular') and set(ambiguities) == {"ua"}

    else:
        init_alpha_1chrom = False
        infer_alpha_intra = False
        infer_alpha_intra_mol = False

    if first_alpha_loop and (multiscale_rounds > 1 or infer_alpha_intra or init_alpha_1chrom or infer_alpha_intra_mol):
        if init_alpha_1chrom:
            chrom_subset_init_alpha = chrom_subset_[np.argmax(lengths_subset)]
        else:
            chrom_subset_init_alpha = chrom_subset_
        if infer_alpha_intra:
            excluded_counts_infer_alpha = 'inter-chromosomal'
        elif infer_alpha_intra_mol:
            excluded_counts_infer_alpha = 'inter-molecular'
        else:
            excluded_counts_infer_alpha = excluded_counts

        if outdir is None:
            outdir_init_alpha = outdir
        else:
            outdir_tmp = 'initial_alpha_inference'
            if multiscale_rounds > 1:
                outdir_tmp += '.singleres'
            if init_alpha_1chrom:
                outdir_tmp += '.' + chrom_subset_init_alpha.replace(' ', '_')
            elif infer_alpha_intra_mol:
                outdir_tmp += '.intra-molecular'
            elif infer_alpha_intra:
                outdir_tmp += '.intra-chromosomal'
            outdir_init_alpha = os.path.join(outdir, outdir_tmp)

        struct_, infer_param = infer(  # TODO should I be using pre-loaded _counts etc here?
            counts=counts, lengths=lengths, ploidy=ploidy,
            outdir=outdir_init_alpha, alpha_init=alpha_init, seed=seed,
            filter_threshold=filter_threshold, normalize=normalize, bias=bias,
            update_alpha=update_alpha,
            beta_init=beta_init, max_alpha_loop=max_alpha_loop,
            beta=beta, multiscale_rounds=1, init=init,
            max_iter=max_iter, factr=factr, pgtol=pgtol,
            alpha_factr=alpha_factr, infer_alpha_intra=False,
            struct_at_final_alpha=not (infer_alpha_intra or init_alpha_1chrom or infer_alpha_intra_mol),
            bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda,
            bcc_version=bcc_version, hsc_version=hsc_version,
            data_interchrom=data_interchrom, est_hmlg_sep=est_hmlg_sep,
            hsc_min_beads=hsc_min_beads, hsc_perc_diff=hsc_perc_diff,
            excluded_counts=excluded_counts_infer_alpha, callback_freq=callback_freq,
            callback_fxns=callback_fxns, reorienter=reorienter,
            multiscale_reform=multiscale_reform, epsilon_min=epsilon_min,
            epsilon_max=epsilon_max, alpha_true=alpha_true,
            struct_true=struct_true, input_weight=input_weight,
            exclude_zeros=exclude_zeros, null=null, chrom_full=chrom_full,
            chrom_subset=chrom_subset_init_alpha, bias_per_hmlg=bias_per_hmlg,
            mixture_coefs=mixture_coefs, verbose=verbose, mods=mods)

        # Do not continue unless inference converged
        if not infer_param['converged']:
            return struct_, infer_param
        if ('alpha_converged' in infer_param) and (
                infer_param['alpha_converged'] is not None) and (
                not infer_param['alpha_converged']):
            return struct_, infer_param

        if not (infer_alpha_intra or init_alpha_1chrom or infer_alpha_intra_mol):
            init_ = struct_
            beta = infer_param['beta']  # FIXME not sure about the beta situation
        if (not init_alpha_1chrom) and 'est_hmlg_sep' in infer_param:
            est_hmlg_sep_ = infer_param['est_hmlg_sep']

        alpha = infer_param['alpha']
        alpha_loop = infer_param['alpha_loop']
        prev_alpha_obj = None
        first_alpha_loop = False
        if infer_alpha_intra or init_alpha_1chrom or infer_alpha_intra_mol:
            alpha_loop += 1
        if infer_alpha_intra or infer_alpha_intra_mol:
            update_alpha = False
        infer_alpha_intra = False
        infer_alpha_intra_mol = False
    elif first_alpha_loop and multiscale_rounds == 1:
        # No need to repeatedly re-load if inferring with single-res
        bias = _bias
        struct_true = _struct_true
        filter_threshold = 0  # Counts have already been filtered (_counts)
        chrom_subset = None  # Chromosomes have already been selected
        lengths = lengths_subset  # Chromosomes have already been selected
        chrom_full = chrom_subset_  # Chromosomes have already been selected
        counts, _, _ = preprocess_counts(
            counts=_counts, lengths=lengths, ploidy=ploidy,
            multiscale_factor=1, exclude_zeros=exclude_zeros,
            beta=beta, bias=bias, input_weight=input_weight, verbose=False,
            excluded_counts=excluded_counts, multiscale_reform=multiscale_reform,
            bias_per_hmlg=bias_per_hmlg, mods=mods)

    # MORE SETUP
    if verbose:
        if alpha_loop is None:
            _print_code_header([
                "INFERRING STRUCTURE WITH FIXED ALPHA",
                f"Alpha = {alpha:.3g}"], max_length=100)
        else:
            if first_alpha_loop:
                _print_code_header([
                    "JOINTLY INFERRING STRUCTURE + ALPHA",
                    f"Initial alpha = {alpha_init:.3g}"], max_length=100)
            _print_code_header([
                "Jointly inferring structure & alpha",
                f"Inferring STRUCTURE #{alpha_loop}"], max_length=80)
    if update_alpha and outdir is not None:
        outdir_ = os.path.join(
            outdir, f"alpha_coord_descent.try{alpha_loop:03d}")
    else:
        outdir_ = outdir

    # INFER DRAFT STRUCTURES (to obtain est_hmlg_sep for hsc2019, if applicable)
    est_hmlg_sep_, draft_converged = _infer_draft(
        counts, lengths=lengths, ploidy=ploidy, outdir=outdir_,
        alpha=alpha, seed=seed, filter_threshold=filter_threshold,
        normalize=normalize, bias=bias, beta=beta,
        multiscale_rounds=multiscale_rounds, beta_init=beta_init, init=init_,
        max_iter=max_iter, factr=factr, pgtol=pgtol, hsc_lambda=hsc_lambda,
        hsc_version=hsc_version, est_hmlg_sep=est_hmlg_sep_,
        hsc_min_beads=hsc_min_beads,
        callback_freq=callback_freq, callback_fxns=callback_fxns,
        reorienter=reorienter, multiscale_reform=multiscale_reform,
        alpha_true=alpha_true, struct_true=struct_true,
        input_weight=input_weight, exclude_zeros=exclude_zeros,
        chrom_full=chrom_full, chrom_subset=chrom_subset,
        bias_per_hmlg=bias_per_hmlg, mixture_coefs=mixture_coefs,
        verbose=verbose, mods=mods)
    if not draft_converged:  # Do not continue unless inference converged
        return None, {'seed': seed, 'converged': draft_converged}

    # INFER FULL-RES GENOME STRUCTURE (with or without multires optimization)
    all_multiscale_factors = 2 ** np.flip(np.arange(int(multiscale_rounds)))
    epsilon_max_ = epsilon_max
    epsilon_prev_ = None
    for multiscale_factor in all_multiscale_factors:
        _print_code_header(
            f'MULTIRES FACTOR {multiscale_factor}', max_length=60,
            blank_lines=1, verbose=verbose and multiscale_rounds > 1)
        if outdir_ is None or multiscale_factor == 1:
            outdir_multires = outdir_
        else:
            outdir_multires = os.path.join(
                outdir_, f'multiscale_x{multiscale_factor}')
        struct_, infer_param = infer_at_alpha(
            counts=counts, outdir=outdir_multires,
            lengths=lengths, ploidy=ploidy, alpha=alpha, seed=seed,
            filter_threshold=filter_threshold, normalize=normalize, bias=bias,
            alpha_init=alpha_init, max_alpha_loop=max_alpha_loop,
            beta=beta, multiscale_factor=multiscale_factor,
            alpha_loop=alpha_loop, update_alpha=update_alpha,
            beta_init=beta_init,
            init=init_, max_iter=max_iter, factr=factr, pgtol=pgtol,
            alpha_factr=alpha_factr, bcc_lambda=bcc_lambda,
            hsc_lambda=hsc_lambda, bcc_version=bcc_version,
            hsc_version=hsc_version, data_interchrom=data_interchrom,
            est_hmlg_sep=est_hmlg_sep_, hsc_min_beads=hsc_min_beads,
            hsc_perc_diff=hsc_perc_diff,
            callback_fxns=callback_fxns, excluded_counts=excluded_counts,
            callback_freq=callback_freq, reorienter=reorienter,
            multiscale_reform=multiscale_reform, epsilon_min=epsilon_min,
            epsilon_max=epsilon_max_, epsilon_prev=epsilon_prev_,
            alpha_true=alpha_true,
            struct_true=struct_true, input_weight=input_weight,
            exclude_zeros=exclude_zeros, null=null,
            chrom_full=chrom_full, chrom_subset=chrom_subset,
            bias_per_hmlg=bias_per_hmlg, mixture_coefs=mixture_coefs,
            verbose=verbose, mods=mods)

        # Do not continue unless inference converged
        if not infer_param['converged']:
            return struct_, infer_param
        if update_alpha and ('alpha_converged' in infer_param) and (
                infer_param['alpha_converged'] is not None) and (
                not infer_param['alpha_converged']):
            return struct_, infer_param

        if reorienter is not None and reorienter.reorient:
            init_ = infer_param['orient']
        else:
            init_ = struct_
        if 'epsilon' in infer_param and infer_param['epsilon'] is not None:
            epsilon_prev_ = infer_param['epsilon']
            if 'eprev' in mods:
                # Epsilon for the next-highest resolution should be smaller than
                # the previous resolution's epsilon
                epsilon_max_ = infer_param['epsilon']
            else:
                # Epsilon for the next-highest resolution should be smaller than
                # the previous resolution's epsilon... but allow some wiggle room
                epsilon_max_ = infer_param['epsilon'] * 1.1
        if 'lowres_exit' in mods:
            exit(0)

    # RE-INFER WITH NEW ALPHA
    if not update_alpha:
        return struct_, infer_param

    if prev_alpha_obj is None:
        prev_alpha_obj = [None, None, None]
    alpha_obj_conv = _convergence_criteria(
        f_k=prev_alpha_obj[0], f_kplus1=infer_param['alpha_obj'],
        factr=alpha_factr)
    stuct_obj_conv = _convergence_criteria(
        f_k=prev_alpha_obj[1], f_kplus1=infer_param['obj'],
        factr=alpha_factr)
    alpha_conv = _convergence_criteria(
        f_k=prev_alpha_obj[2], f_kplus1=infer_param['alpha'],
        factr=alpha_factr)
    print(f"alpha={infer_param['alpha']:.3g}    {alpha_obj_conv=}    {stuct_obj_conv=}    {alpha_conv=}")

    if (alpha_loop >= max_alpha_loop) or _convergence_criteria(
            f_k=prev_alpha_obj[0], f_kplus1=infer_param['alpha_obj'],
            factr=alpha_factr):
        update_alpha = False
        if not struct_at_final_alpha:
            return struct_, infer_param
    return infer(
        counts=counts, lengths=lengths, ploidy=ploidy,
        outdir=outdir, alpha=infer_param['alpha'], seed=seed,
        filter_threshold=filter_threshold, normalize=normalize, bias=bias,
        update_alpha=update_alpha,
        prev_alpha_obj=[infer_param['alpha_obj'], infer_param['obj'], infer_param['alpha']],
        beta_init=beta_init,
        alpha_loop=alpha_loop + 1, max_alpha_loop=max_alpha_loop,
        beta=infer_param['beta'], multiscale_rounds=multiscale_rounds,
        init=init_, max_iter=max_iter, factr=factr, pgtol=pgtol,
        alpha_factr=alpha_factr, infer_alpha_intra=infer_alpha_intra,
        struct_at_final_alpha=struct_at_final_alpha,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda, bcc_version=bcc_version,
        hsc_version=hsc_version,
        data_interchrom=data_interchrom, est_hmlg_sep=est_hmlg_sep,
        hsc_min_beads=hsc_min_beads, hsc_perc_diff=hsc_perc_diff,
        excluded_counts=excluded_counts, callback_freq=callback_freq,
        callback_fxns=callback_fxns, reorienter=reorienter,
        multiscale_reform=multiscale_reform, epsilon_min=epsilon_min,
        epsilon_max=epsilon_max, alpha_true=alpha_true, struct_true=struct_true,
        input_weight=input_weight, exclude_zeros=exclude_zeros, null=null,
        chrom_full=chrom_full, chrom_subset=chrom_subset,
        bias_per_hmlg=bias_per_hmlg, mixture_coefs=mixture_coefs,
        verbose=verbose, mods=mods)


def pastis_poisson(counts, lengths, ploidy, outdir='', chromosomes=None,
                   chrom_subset=None, alpha=None, seed=0,
                   filter_threshold=0.04, normalize=True, bias=None,
                   alpha_init=None, max_alpha_loop=20,
                   beta=None, multiscale_rounds=1,
                   max_iter=30000, factr=1e7, pgtol=1e-05,
                   alpha_factr=1e12, infer_alpha_intra=True,
                   struct_at_final_alpha=True, bcc_lambda=0,
                   hsc_lambda=0, bcc_version='2019', hsc_version='2019',
                   data_interchrom=None, est_hmlg_sep=None, hsc_min_beads=5,
                   hsc_perc_diff=None,
                   callback_fxns=None, print_freq=100, log_freq=1000,
                   save_freq=None, piecewise=False, piecewise_step=None,
                   piecewise_chrom=None, piecewise_min_beads=5,
                   piecewise_fix_homo=False, piecewise_opt_orient=True,
                   piecewise_step3_multiscale=False,
                   piecewise_step1_accuracy=1,
                   multiscale_reform=False, epsilon_min=1e-2, epsilon_max=1e6,
                   alpha_true=None, struct_true=None, init='mds', input_weight=None,
                   exclude_zeros=False, null=False, bias_per_hmlg=False,
                   mixture_coefs=None, verbose=True, mods=[]):
    """Infer 3D structures with PASTIS via Poisson model.

    Infer 3D structure from Hi-C contact counts data for haploid or diploid
    organisms.

    Parameters
    ----------
    counts : list of str
        Counts data files in the hiclib format or as numpy ndarrays.
    lengths : str or list
        Number of beads per homolog of each chromosome, or hiclib .bed file with
        lengths data.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    outdir : str, optional
        Directory in which to save results.
    chromosomes : list of str, optional
        Label for each chromosome in the data, or file with chromosome labels
        (one label per line).
    chrom_subset : list of str, optional
        Labels of chromosomes for which inference should be performed.
    alpha : float, optional
        Biophysical parameter of the transfer function used in converting
        counts to wish distances. If alpha is not specified, it will be
        inferred.
    seed : int, optional
        Random seed used when generating the starting point in the
        optimization.
    normalize : bool, optional
        Perfrom ICE normalization on the counts prior to optimization.
        Normalization is reccomended.
    filter_threshold : float, optional
        Ratio of non-zero beads to be filtered out. Filtering is
        reccomended.
    alpha_init : float, optional
        For PM2, the initial value of alpha to use.
    max_alpha_loop : int, optional
        For PM2, Number of times alpha and structure are inferred.
    beta : array_like of float, optional
        Scaling parameter that determines the size of the structure, relative to
        each counts matrix. There should be one beta per counts matrix. If None,
        the optimal beta will be estimated.
    multiscale_rounds : int, optional
        The number of resolutions at which a structure should be inferred
        during multiscale optimization. Values of 1 or 0 disable multiscale
        optimization.
    max_iter : int, optional
        Maximum number of iterations per optimization.
    factr : float, optional
        factr for scipy's L-BFGS-B, alters convergence criteria.
    pgtol : float, optional
        pgtol for scipy's L-BFGS-B, alters convergence criteria.
    alpha_factr : float, optional
        factr for convergence criteria of joint alpha/structure inference.
    bcc_lambda : float, optional
        Lambda of the bead chain connectivity constraint.
    hsc_lambda : float, optional
        For diploid organisms: lambda of the homolog-separating
        constraint.
    TODO add bcc_version, hsc_version, & data_interchrom
    est_hmlg_sep : list of float, optional
        For diploid organisms: hyperparameter of the homolog-separating
        constraint specificying the expected distance between homolog
        centers of mass for each chromosome. If not supplied, `est_hmlg_sep` will
        be inferred from the counts data.
    hsc_min_beads : int, optional
        For diploid organisms: number of beads in the low-resolution
        structure from which `est_hmlg_sep` is estimated.
    hsc_perc_diff : float, optional
        For diploid organisms: TODO

    Returns
    -------
    struct_ : array_like of float of shape (lengths.sum() * ploidy, 3)
        3D structure resulting from the optimization.
    infer_param : dict
        A few of the parameters used in inference or generated by inference.
        Keys: 'alpha', 'beta', 'est_hmlg_sep', 'obj', and 'seed'.
    """

    if mods is None:
        mods = []
    elif isinstance(mods, str):
        mods = mods.lower().split('.')
    else:
        mods = [x.lower() for x in mods]
    if 'debug_nan_inf' in mods:
        _setup_jax(debug_nan_inf=True)
    if 'bias_per_hmlg' in mods:
        bias_per_hmlg = True

    if not isinstance(counts, (list, tuple)):
        counts = [counts]
    if verbose:
        print(f"\nRANDOM SEED = {seed:03d}", flush=True)
        if all([isinstance(c, str) for c in counts]):
            print("COUNTS: " + '        \n'.join(counts) + "\n", flush=True)

    callback_freq = {'print_freq': print_freq, 'log_freq': log_freq,
                     'save_freq': save_freq}
    outdir = _output_subdir(
        outdir=outdir, chrom_full=chromosomes, chrom_subset=chrom_subset,
        null=null, lengths=lengths)

    if piecewise and (chrom_subset is None or len(chrom_subset) > 1):
        from .piecewise_whole_genome import infer_piecewise

        struct_, infer_param = infer_piecewise(
            counts=counts, outdir=outdir, lengths=lengths,
            ploidy=ploidy, chromosomes=chromosomes, chrom_subset=chrom_subset,
            alpha=alpha, seed=seed,
            filter_threshold=filter_threshold, normalize=normalize, bias=bias,
            alpha_init=alpha_init, max_alpha_loop=max_alpha_loop, beta=beta,
            multiscale_rounds=multiscale_rounds, max_iter=max_iter,
            factr=factr, pgtol=pgtol, alpha_factr=alpha_factr,
            bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda,
            bcc_version=bcc_version, hsc_version=hsc_version,
            data_interchrom=data_interchrom,
            est_hmlg_sep=est_hmlg_sep, hsc_perc_diff=hsc_perc_diff,
            hsc_min_beads=hsc_min_beads, callback_fxns=callback_fxns,
            callback_freq=callback_freq,
            piecewise_step=piecewise_step,
            piecewise_chrom=piecewise_chrom,
            piecewise_min_beads=piecewise_min_beads,
            piecewise_fix_homo=piecewise_fix_homo,
            piecewise_opt_orient=piecewise_opt_orient,
            piecewise_step3_multiscale=piecewise_step3_multiscale,
            piecewise_step1_accuracy=piecewise_step1_accuracy,
            multiscale_reform=multiscale_reform, alpha_true=alpha_true,
            struct_true=struct_true, init=init, input_weight=input_weight,
            exclude_zeros=exclude_zeros, null=null, mixture_coefs=mixture_coefs,
            verbose=verbose, mods=mods)
    else:
        struct_, infer_param = infer(
            counts=counts, lengths=lengths, ploidy=ploidy,
            outdir=outdir, alpha=alpha, seed=seed,
            filter_threshold=filter_threshold, normalize=normalize, bias=bias,
            alpha_init=alpha_init, max_alpha_loop=max_alpha_loop, beta=beta,
            multiscale_rounds=multiscale_rounds,
            init=init, max_iter=max_iter, factr=factr, pgtol=pgtol,
            alpha_factr=alpha_factr, infer_alpha_intra=infer_alpha_intra,
            struct_at_final_alpha=struct_at_final_alpha,
            bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda,
            bcc_version=bcc_version, hsc_version=hsc_version,
            data_interchrom=data_interchrom, est_hmlg_sep=est_hmlg_sep,
            hsc_min_beads=hsc_min_beads, hsc_perc_diff=hsc_perc_diff,
            callback_freq=callback_freq, callback_fxns=callback_fxns,
            multiscale_reform=multiscale_reform, epsilon_min=epsilon_min,
            epsilon_max=epsilon_max, alpha_true=alpha_true, struct_true=struct_true,
            input_weight=input_weight, exclude_zeros=exclude_zeros, null=null,
            chrom_full=chromosomes, chrom_subset=chrom_subset,
            bias_per_hmlg=bias_per_hmlg, mixture_coefs=mixture_coefs,
            verbose=verbose, mods=mods)

    if verbose:
        if ('alpha_converged' in infer_param) and (
                infer_param['alpha_converged'] is not None) and (
                not infer_param['alpha_converged']):
            print("INFERENCE COMPLETE: DID NOT CONVERGE", flush=True)
        elif infer_param['converged']:
            print("INFERENCE COMPLETE: CONVERGED", flush=True)
        else:
            print("INFERENCE COMPLETE: DID NOT CONVERGE", flush=True)

    return struct_, infer_param
