from __future__ import print_function

#import jax; jax.config.update('jax_platform_name', 'cpu')

import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.utils import check_random_state

from .utils_poisson import _print_code_header, _get_output_files
from .utils_poisson import _output_subdir, _load_infer_param
from .counts import preprocess_counts, ambiguate_counts, _ambiguate_beta
from .counts import check_counts, _set_initial_beta
from .initialization import initialize
from .callbacks import Callback
from .constraints import prep_constraints, distance_between_homologs
from .constraints import calc_counts_interchrom
from .poisson import PastisPM, _convergence_criteria, get_eps_types
from .multiscale_optimization import get_multiscale_variances_from_struct
from .multiscale_optimization import _get_stretch_of_fullres_beads
from .multiscale_optimization import _choose_max_multiscale_factor
from .multiscale_optimization import decrease_lengths_res
from ..io.read import load_data


def _infer_draft(counts, lengths, ploidy, outdir=None, alpha=None, seed=0,
                 filter_threshold=0.04, normalize=True, bias=None, beta=None,
                 multiscale_rounds=1, use_multiscale_variance=True, beta_init=None,
                 init='mds', max_iter=30000, factr=1e7, pgtol=1e-05,
                 hsc_lambda=0, hsc_version='2019', est_hmlg_sep=None,
                 hsc_min_beads=5, struct_draft_fullres=None, callback_freq=None,
                 callback_fxns=None, reorienter=None,
                 multiscale_reform=False, alpha_true=None,
                 struct_true=None, input_weight=None, exclude_zeros=False,
                 null=False, chrom_full=None, chrom_subset=None,
                 mixture_coefs=None, verbose=True, mods=[]):
    """Infer draft 3D structures with PASTIS via Poisson model.
    """

    infer_draft_lowres = est_hmlg_sep is None and hsc_lambda > 0 and str(
        hsc_version) == '2019'
    need_multiscale_var = use_multiscale_variance and (
        multiscale_rounds > 1 or infer_draft_lowres) and (not multiscale_reform)
    infer_draft_fullres = struct_draft_fullres is None and (
        need_multiscale_var)

    if not infer_draft_fullres or infer_draft_lowres:
        return struct_draft_fullres, est_hmlg_sep, True

    counts, bias, lengths, _, _, _, struct_true = load_data(
        counts=counts, lengths_full=lengths, ploidy=ploidy,
        chrom_full=chrom_full, chrom_subset=chrom_subset,
        filter_threshold=filter_threshold, normalize=normalize, bias=bias,
        exclude_zeros=exclude_zeros, struct_true=struct_true, verbose=False)

    multiscale_factor_for_lowres = _choose_max_multiscale_factor(
        lengths=lengths, min_beads=hsc_min_beads)
    if verbose:
        if (infer_draft_fullres and infer_draft_lowres):
            _print_code_header(
                'INFERRING DRAFT STRUCTURES', max_length=60, blank_lines=2)
        elif infer_draft_fullres:
            _print_code_header(
                ['INFERRING DRAFT STRUCTURE', 'Full resolution'],
                max_length=60, blank_lines=2)
        elif infer_draft_lowres:
            _print_code_header(
                ['INFERRING DRAFT STRUCTURE',
                    f'Low resolution ({multiscale_factor_for_lowres}x)'],
                max_length=60, blank_lines=2)

    counts_preprocess, _, _ = preprocess_counts(
        counts_raw=counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=1, exclude_zeros=exclude_zeros, beta=beta,
        input_weight=input_weight, verbose=False, mixture_coefs=mixture_coefs,
        mods=mods)
    beta = [c.beta for c in counts_preprocess if c.sum() != 0]

    if infer_draft_fullres:
        if verbose and infer_draft_lowres:
            _print_code_header(
                "Inferring full-res draft structure",
                max_length=50, blank_lines=1)
        if outdir is None:
            fullres_outdir = None
        else:
            fullres_outdir = os.path.join(outdir, 'struct_draft_fullres')
        struct_draft_fullres, infer_param_fullres = infer_at_alpha(
            counts=counts, outdir=fullres_outdir, lengths=lengths,
            ploidy=ploidy, alpha=alpha, seed=seed,
            filter_threshold=filter_threshold, normalize=normalize, bias=bias,
            beta=beta, beta_init=beta_init, init=init, max_iter=max_iter,
            factr=factr, pgtol=pgtol, draft=True, simple_diploid=(ploidy == 2),
            callback_fxns=callback_fxns, callback_freq=callback_freq,
            reorienter=reorienter, multiscale_reform=multiscale_reform,
            alpha_true=alpha_true, struct_true=struct_true,
            input_weight=input_weight, exclude_zeros=exclude_zeros, null=null,
            mixture_coefs=mixture_coefs, verbose=verbose, mods=mods)
        if not infer_param_fullres['converged']:
            return None, est_hmlg_sep, False

    if infer_draft_lowres:
        if verbose and infer_draft_fullres:
            _print_code_header(
                "Inferring low-res draft structure (%dx)"
                % multiscale_factor_for_lowres,
                max_length=50, blank_lines=1)
        if ploidy == 1:
            raise ValueError("Can not apply homolog-separating constraint"
                             " to haploid data.")
        if outdir is None:
            lowres_outdir = None
        else:
            lowres_outdir = os.path.join(outdir, 'struct_draft_lowres')
        ua_index = [i for i in range(len(
            counts)) if counts[i].shape == (lengths.sum() * ploidy,
                                                    lengths.sum() * ploidy)]
        if len(ua_index) == 1:
            counts_for_lowres = [counts[ua_index[0]]]
            simple_diploid_for_lowres = False
            beta_for_lowres = [beta[ua_index[0]]]
        elif len(ua_index) > 1:
            raise ValueError("Only input one matrix of unambiguous counts."
                             " Please pool unambiguous counts before"
                             " inputting.")
        else:
            if lengths.shape[0] == 1:
                raise ValueError("Please input more than one chromosome to"
                                 " estimate est_hmlg_sep from ambiguous data.")
            counts_for_lowres = counts
            simple_diploid_for_lowres = True
            beta_for_lowres = beta
        struct_draft_lowres, infer_param_lowres = infer_at_alpha(
            counts=counts_for_lowres, outdir=lowres_outdir,
            lengths=lengths, ploidy=ploidy, alpha=alpha, seed=seed,
            filter_threshold=filter_threshold, normalize=normalize, bias=bias,
            beta=beta_for_lowres, beta_init=beta_init,
            multiscale_factor=multiscale_factor_for_lowres,
            use_multiscale_variance=use_multiscale_variance,
            init=init, max_iter=max_iter, factr=factr, pgtol=pgtol,
            struct_draft_fullres=struct_draft_fullres, draft=True,
            simple_diploid=simple_diploid_for_lowres,
            callback_fxns=callback_fxns,
            callback_freq=callback_freq,
            reorienter=reorienter, multiscale_reform=multiscale_reform,
            alpha_true=alpha_true, struct_true=struct_true,
            input_weight=input_weight, exclude_zeros=exclude_zeros, null=null,
            mixture_coefs=mixture_coefs, verbose=verbose, mods=mods)
        if not infer_param_lowres['converged']:
            return struct_draft_fullres, None, False
        est_hmlg_sep = distance_between_homologs(
            structures=struct_draft_lowres,
            lengths=decrease_lengths_res(
                lengths=lengths,
                multiscale_factor=multiscale_factor_for_lowres),
            mixture_coefs=mixture_coefs,
            simple_diploid=simple_diploid_for_lowres)
        if verbose:
            print("Estimated distance between homolog barycenters for each"
                  f" chromosome: {' '.join(map(str, est_hmlg_sep.round(2)))}",
                  flush=True)

    if verbose:
        _print_code_header(
            ['Draft inference complete', 'INFERRING STRUCTURE'],
            max_length=60, blank_lines=2)

    return struct_draft_fullres, est_hmlg_sep, True


def _prep_inference(counts_raw, lengths, ploidy, outdir='', alpha=None, seed=0,
                    filter_threshold=0.04, normalize=True, bias=None, alpha_init=-3,
                    max_alpha_loop=20, beta=None, multiscale_factor=1,
                    use_multiscale_variance=True, beta_init=None,
                    init='mds', max_iter=30000,
                    factr=1e7, pgtol=1e-05, alpha_factr=1e12,
                    bcc_lambda=0, hsc_lambda=0, bcc_version='2019',
                    hsc_version='2019', counts_inter_mv=None, est_hmlg_sep=None,
                    hsc_min_beads=5, hsc_perc_diff=None, excluded_counts=None,
                    struct_draft_fullres=None, draft=False, simple_diploid=False,
                    callback_freq=None, callback_fxns=None, reorienter=None,
                    multiscale_reform=False, init_std_dev=None,
                    alpha_true=None, struct_true=None, input_weight=None,
                    exclude_zeros=False, null=False, chrom_full=None, chrom_subset=None,
                    mixture_coefs=None,
                    outfiles=None, verbose=True, mods=[]):
    """TODO"""

    if verbose and outfiles is not None:
        print(f"OUTPUT: {outfiles['struct_infer']}", flush=True)

    # MULTISCALE VARIANCES
    if multiscale_factor != 1 and use_multiscale_variance and struct_draft_fullres is not None and not multiscale_reform:
        multiscale_variances = np.median(get_multiscale_variances_from_struct(
            struct_draft_fullres, lengths=lengths,
            multiscale_factor=multiscale_factor, mixture_coefs=mixture_coefs,
            verbose=verbose))
        if struct_true is not None and verbose:
            multiscale_variances_true = np.median(
                get_multiscale_variances_from_struct(
                    struct_true, lengths=lengths,
                    multiscale_factor=multiscale_factor,
                    mixture_coefs=mixture_coefs, verbose=False))
            print(f"True multiscale variance ({multiscale_factor}x):"
                  f" {multiscale_variances_true:.3g}", flush=True)
    else:
        multiscale_variances = None

    # PREPARE COUNTS OBJECTS
    counts, struct_nan, fullres_struct_nan = preprocess_counts(
        counts_raw=counts_raw, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, exclude_zeros=exclude_zeros,
        beta=beta, input_weight=input_weight, verbose=verbose,
        excluded_counts=excluded_counts, mixture_coefs=mixture_coefs,
        simple_diploid=simple_diploid, mods=mods)
    if simple_diploid:
        ploidy = 1
        if beta_init is not None:
            beta_init = beta_init * 2
    if verbose:
        print('BETA: ' + ', '.join(
            [f'{c.ambiguity}={c.beta:.3g}' for c in counts if c.sum() != 0]),
            flush=True)
        if alpha is None:
            print(f'ALPHA: to be inferred, init = {alpha_init:.3g}', flush=True)
        else:
            print(f'ALPHA: {alpha:.3g}', flush=True)

    # SETUP MULTI-RES
    stretch_fullres_beads = mean_fullres_nghbr_dis = None
    if multiscale_reform and ('adjust_eps' in mods):
        if beta is None:
            mean_fullres_nghbr_dis = 1
        else:
            from .constraints import _neighboring_bead_indices
            from .utils_poisson import _euclidean_distance
            row_nghbr = _neighboring_bead_indices(
                lengths=lengths, ploidy=1, multiscale_factor=1)
            dis_nghbr = _euclidean_distance(struct_true, row=row_nghbr, col=row_nghbr + 1)._value
            print(f"\n{dis_nghbr.mean()=:.4g}")
            print(f"{np.power(dis_nghbr, alpha).mean()=:.4g}\n")

            beta_tmp, _ = _set_initial_beta(
                counts_raw, lengths=lengths, ploidy=ploidy, bias=bias,
                exclude_zeros=exclude_zeros, neighboring_beads_only=True)
            beta_ambig = _ambiguate_beta(
                beta, counts_raw, lengths=lengths, ploidy=ploidy)
            mean_fullres_nghbr_dis_alpha = np.power(
                beta_tmp / beta_ambig, 1 / alpha)

            row_nghbr2 = _neighboring_bead_indices(
                lengths=lengths, ploidy=2, multiscale_factor=1)
            c = counts_raw[0]
            c_nghbr = c[row_nghbr2, row_nghbr2 + 1]
            print(f"{np.nanmean(c_nghbr / beta_ambig)=:.4g}")
            print(f"{np.nanmean(np.power(c_nghbr / beta_ambig, 1 / alpha))=:.4g}\n")

            mean_fullres_nghbr_dis = beta_tmp / beta_ambig
            print(f"{mean_fullres_nghbr_dis=:.3g}\n")
            print(f"{mean_fullres_nghbr_dis_alpha=:.3g}\n")
            exit(1)

        stretch_fullres_beads = _get_stretch_of_fullres_beads(
                multiscale_factor=multiscale_factor, lengths=lengths,
                ploidy=ploidy, fullres_struct_nan=fullres_struct_nan)
        if 'adjust_eps_all' in mods:
            eps_types = get_eps_types(stretch_fullres_beads)
            if eps_types.size > 1:
                epsilon = np.append(
                    epsilon, random_state.uniform(size=eps_types.size - 1))

    # HOMOLOG-SEPARATING CONSTRAINT
    if ploidy == 1 and hsc_lambda > 0:
        raise ValueError("Can not apply homolog-separating constraint to"
                         " haploid data.")
    if hsc_lambda > 0:
        if est_hmlg_sep is not None:
            est_hmlg_sep = np.array(est_hmlg_sep, dtype=float).reshape(-1, )
            if est_hmlg_sep.shape[0] == 1 and lengths.shape[0] != 1:
                est_hmlg_sep = np.tile(est_hmlg_sep, lengths.shape[0])
        if est_hmlg_sep is None and reorienter is not None and reorienter.reorient:
            est_hmlg_sep = distance_between_homologs(
                structures=reorienter.struct_init, lengths=lengths,
                mixture_coefs=mixture_coefs)

    # INITIALIZATION
    random_state = np.random.RandomState(seed)
    random_state = check_random_state(random_state)
    if isinstance(init, str) and init.lower() == 'true':
        if struct_true is None:
            raise ValueError("Attempting to initialize with struct_true but"
                             " struct_true is None")
        if verbose:
            print('INITIALIZATION: initializing with true structure',
                  flush=True)
        init = struct_true
    struct_init = initialize(
        counts=counts, lengths=lengths, init=init, ploidy=ploidy,
        random_state=random_state,
        alpha=alpha_init if alpha is None else alpha,
        bias=bias, multiscale_factor=multiscale_factor,
        reorienter=reorienter, std_dev=init_std_dev,
        mixture_coefs=mixture_coefs, verbose=verbose, mods=mods)
    if multiscale_reform and multiscale_factor != 1:
        epsilon = random_state.uniform()
    else:
        epsilon = None

    # SETUP CONSTRAINTS
    constraints = prep_constraints(
        counts=counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, bcc_lambda=bcc_lambda,
        hsc_lambda=hsc_lambda, bcc_version=bcc_version,
        hsc_version=hsc_version, counts_inter_mv=counts_inter_mv,
        est_hmlg_sep=est_hmlg_sep, hsc_perc_diff=hsc_perc_diff,
        fullres_struct_nan=fullres_struct_nan, verbose=verbose, mods=mods)

    # SETUP CALLBACKS
    if simple_diploid and struct_true is not None:
        struct_true_tmp = np.nanmean(
            [struct_true[:int(struct_true.shape[0] / 2)],
             struct_true[int(struct_true.shape[0] / 2):]], axis=0)
    else:
        struct_true_tmp = struct_true
    if callback_fxns is None:
        callback_fxns = {}
    callback = Callback(
        lengths=lengths, ploidy=ploidy, counts=counts, bias=bias,
        multiscale_factor=multiscale_factor,
        multiscale_reform=multiscale_reform, frequency=callback_freq,
        directory=outdir, seed=seed, struct_true=struct_true_tmp,
        alpha_true=alpha_true, constraints=constraints, beta_init=beta_init,
        mixture_coefs=mixture_coefs, **callback_fxns,
        multiscale_variances=multiscale_variances, verbose=verbose,
        mods=mods)

    return (counts, bias, struct_nan, struct_init, constraints, callback,
            multiscale_variances, epsilon, stretch_fullres_beads,
            mean_fullres_nghbr_dis)


def infer_at_alpha(counts, lengths, ploidy, outdir='', alpha=None, seed=0,
                   filter_threshold=0.04, normalize=True, bias=None, alpha_init=-3,
                   max_alpha_loop=20, beta=None, multiscale_factor=1,
                   use_multiscale_variance=True, alpha_loop=None, update_alpha=False,
                   beta_init=None, init='mds', max_iter=30000,
                   factr=1e7, pgtol=1e-05, alpha_factr=1e12,
                   bcc_lambda=0, hsc_lambda=0, bcc_version='2019', hsc_version='2019',
                   counts_inter_mv=None, est_hmlg_sep=None, hsc_min_beads=5,
                   hsc_perc_diff=None, excluded_counts=None,
                   struct_draft_fullres=None, draft=False, simple_diploid=False,
                   callback_freq=None, callback_fxns=None, reorienter=None,
                   multiscale_reform=False, epsilon_min=1e-6, epsilon_max=1e6,
                   epsilon_coord_descent=False, init_std_dev=None,
                   alpha_true=None, struct_true=None, input_weight=None,
                   exclude_zeros=False, null=False,
                   chrom_full=None, chrom_subset=None, mixture_coefs=None, verbose=True,
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
        optimization. # TODO move to infer()
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
    counts_inter_mv : int or float, optional
        TODO add bcc_version, hsc_version, & counts_inter_mv
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
    excluded_counts : {"inter", "intra"}, optional
        Whether to exclude inter- or intra-chromosomal counts from optimization.
    struct_draft_fullres : np.ndarray, optional
        The full-resolution draft structure from whihc to derive multiscale
        variances.
    draft: bool, optional
        Whether this optimization is inferring a draft structure.
    simple_diploid: bool, optional
        For diploid organisms: whether this optimization is inferring a "simple
        diploid" structure in which homologs are assumed to be identical and
        completely overlapping with one another.

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
        if os.path.exists(outfiles['struct_infer']) or os.path.exists(
                outfiles['struct_nonconv']):
            infer_param = _load_infer_param(outfiles['infer_param'])
            if verbose:
                if os.path.exists(outfiles['struct_infer']):
                    print('CONVERGED\n', flush=True)
                    struct_ = np.loadtxt(outfiles['struct_infer'])
                elif os.path.exists(outfiles['struct_nonconv']):
                    print('OPTIMIZATION DID NOT CONVERGE\n', flush=True)
                    struct_ = None
            return struct_, infer_param
    else:
        outfiles = None

    # LOAD DATA
    counts, bias, lengths, _, _, _, struct_true = load_data(
        counts=counts, lengths_full=lengths, ploidy=ploidy,
        chrom_full=chrom_full, chrom_subset=chrom_subset,
        filter_threshold=filter_threshold, normalize=normalize, bias=bias,
        exclude_zeros=exclude_zeros, struct_true=struct_true, verbose=verbose)

    # PREP FOR INFERENCE
    prepped = _prep_inference(
        counts, lengths=lengths, ploidy=ploidy, outdir=outdir, alpha=alpha,
        seed=seed, filter_threshold=filter_threshold, normalize=normalize,
        bias=bias, alpha_init=alpha_init, max_alpha_loop=max_alpha_loop,
        beta=beta, multiscale_factor=multiscale_factor,
        use_multiscale_variance=use_multiscale_variance, beta_init=beta_init, init=init,
        max_iter=max_iter, factr=factr, pgtol=pgtol, alpha_factr=alpha_factr,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda, bcc_version=bcc_version,
        hsc_version=hsc_version, counts_inter_mv=counts_inter_mv,
        est_hmlg_sep=est_hmlg_sep, hsc_min_beads=hsc_min_beads,
        hsc_perc_diff=hsc_perc_diff, excluded_counts=excluded_counts,
        struct_draft_fullres=struct_draft_fullres, draft=draft,
        simple_diploid=simple_diploid, callback_freq=callback_freq,
        callback_fxns=callback_fxns, reorienter=reorienter,
        multiscale_reform=multiscale_reform, init_std_dev=init_std_dev,
        alpha_true=alpha_true, struct_true=struct_true, input_weight=input_weight,
        exclude_zeros=exclude_zeros, null=null,
        chrom_full=chrom_full, chrom_subset=chrom_subset, mixture_coefs=mixture_coefs,
        outfiles=outfiles, verbose=verbose, mods=mods)
    (counts, bias, struct_nan, struct_init, constraints, callback,
        multiscale_variances, epsilon, stretch_fullres_beads,
        mean_fullres_nghbr_dis) = prepped

    # INFER STRUCTURE
    # original_counts_beta = [c.beta for c in counts if c.sum() != 0]
    pm = PastisPM(
        counts=counts, lengths=lengths, ploidy=ploidy, alpha=alpha,
        init=struct_init, bias=bias, constraints=constraints,
        callback=callback, multiscale_factor=multiscale_factor,
        multiscale_variances=multiscale_variances, epsilon=epsilon,
        epsilon_bounds=[epsilon_min, epsilon_max],
        stretch_fullres_beads=stretch_fullres_beads,
        mean_fullres_nghbr_dis=mean_fullres_nghbr_dis,
        epsilon_coord_descent=epsilon_coord_descent, alpha_init=alpha_init,
        max_alpha_loop=max_alpha_loop, max_iter=max_iter, factr=factr,
        pgtol=pgtol, alpha_factr=alpha_factr, reorienter=reorienter,
        null=null, mixture_coefs=mixture_coefs, verbose=verbose, mods=mods)
    pm._fit_structure(alpha_loop=alpha_loop)
    struct_ = pm.struct_.reshape(-1, 3)
    struct_[struct_nan] = np.nan

    # OPTIONALLY RE-INFER ALPHA
    alpha_converged_ = None
    if update_alpha and multiscale_factor == 1:
        if alpha_loop is not None:
            _print_code_header([
                "Jointly inferring structure & alpha",
                f"Inferring ALPHA #{alpha_loop}"], max_length=80)
        pm._fit_alpha(alpha_loop=alpha_loop)
        alpha_converged_ = pm.alpha_converged_

    # GET RESCALING FACTOR: RE-SCALES STRUCTURE TO MATCH ORIGINAL BETA
    beta_current = _ambiguate_beta(
        pm.beta_, counts=counts, lengths=lengths, ploidy=ploidy)
    if beta_init is None:
        rescale_by = None
    else:
        rescale_by = beta_init / beta_current

    # SAVE RESULTS
    infer_param = {
        'alpha': pm.alpha_, 'beta': pm.beta_, 'obj': pm.obj_, 'seed': seed,
        'converged': pm.converged_, 'conv_desc': pm.conv_desc_,
        'time': pm.time_elapsed_, 'epsilon': pm.epsilon_,
        'alpha_obj': pm.alpha_obj_, 'alpha_converged': alpha_converged_,
        'multiscale_variances': multiscale_variances, 'alpha_loop': alpha_loop,
        'rescale_by': rescale_by}
    if constraints is not None:
        hsc19 = [c for c in constraints if (
            c.name == "Homolog separating (2019)" and c.lambda_val > 0)]
        if len(hsc19) == 1:
            infer_param['est_hmlg_sep'] = hsc19[0].hparams['est_hmlg_sep']
    if reorienter is not None and reorienter.reorient:
        infer_param['orient'] = pm.orientation_.flatten()

    if outfiles is not None:
        os.makedirs(outfiles['dir'], exist_ok=True)
        with open(outfiles['infer_param'], 'w') as f:
            for k, v in infer_param.items():
                if isinstance(v, (np.ndarray, list, tuple)):
                    f.write(f"{k}\t{' '.join([f'{x:g}' for x in v])}\n")
                elif v is not None:
                    f.write(f'{k}\t{v}\n')
        if reorienter is not None and reorienter.reorient:
            np.savetxt(outfiles['reorient'], pm.orientation_)
        if pm.converged_:
            np.savetxt(outfiles['struct_infer'], struct_)
            if pm.history_ is not None and len(pm.history_) > 0:
                pd.DataFrame(pm.history_).to_csv(
                    outfiles['history'], sep='\t', index=False)
        else:
            np.savetxt(outfiles['struct_nonconv'], struct_)

    if pm.converged_:
        return struct_, infer_param
    else:
        return None, infer_param


def infer(counts, lengths, ploidy, outdir='', alpha=None, seed=0,
          filter_threshold=0.04, normalize=True, bias=None, alpha_init=-3,
          max_alpha_loop=20, beta=None,
          multiscale_rounds=1, use_multiscale_variance=True,
          struct_draft_fullres=None,
          alpha_loop=None, update_alpha=False, prev_alpha_obj=None,
          beta_init=None, init='mds', max_iter=30000,
          factr=1e7, pgtol=1e-05, alpha_factr=1e12,
          bcc_lambda=0, hsc_lambda=0, bcc_version='2019', hsc_version='2019',
          counts_inter_mv=None, est_hmlg_sep=None, hsc_min_beads=5,
          hsc_perc_diff=None, excluded_counts=None,
          callback_freq=None, callback_fxns=None, reorienter=None,
          multiscale_reform=False, epsilon_min=1e-6, epsilon_max=1e6,
          epsilon_coord_descent=False, alpha_true=None, struct_true=None,
          input_weight=None, exclude_zeros=False, null=False,
          chrom_full=None, chrom_subset=None,
          mixture_coefs=None, verbose=True, mods=[]):
    """TODO"""

    # SETUP
    if alpha is None:
        update_alpha = True
        alpha = alpha_init
    if update_alpha and alpha_loop is None:
        alpha_loop = 1
    if update_alpha:
        outdir_ = os.path.join(outdir, f"alpha_coord_desc.{alpha_loop:03d}")
    else:
        outdir_ = outdir
    first_alpha_loop = update_alpha and alpha_loop == 1
    if multiscale_rounds <= 0:
        multiscale_rounds = 1

    # LOAD DATA
    if first_alpha_loop or multiscale_rounds == 1:
        _counts, _bias, lengths_subset, chrom_full, _, _, _struct_true = load_data(
            counts=counts, lengths_full=lengths, ploidy=ploidy,
            chrom_full=chrom_full, chrom_subset=chrom_subset,
            filter_threshold=filter_threshold, normalize=normalize, bias=bias,
            exclude_zeros=exclude_zeros, struct_true=struct_true,
            verbose=verbose)
        # Get initial beta
        if first_alpha_loop and beta_init is None:
            if beta is not None:
                beta_init = _ambiguate_beta(
                    beta, counts=_counts, lengths=lengths_subset, ploidy=ploidy)
            else:
                beta_init, _ = _set_initial_beta(
                    _counts, lengths=lengths_subset, ploidy=ploidy, bias=_bias,
                    exclude_zeros=exclude_zeros)
    # Get inter-chromosomal counts (if needed)
    if counts_inter_mv is None and ((
            bcc_lambda > 0 and bcc_version == '2022') or (
            hsc_lambda > 0 and hsc_version == '2022')):
        counts_inter_mv = calc_counts_interchrom(
            counts, lengths=lengths, ploidy=ploidy,
            filter_threshold=filter_threshold, normalize=normalize, bias=bias,
            alpha=alpha, beta=beta,
            verbose=verbose, mods=mods)
    # No need to repeatedly re-load if inferring with single-res
    if multiscale_rounds == 1:
        counts = _counts
        bias = _bias
        struct_true = _struct_true
        filter_threshold = 0  # Counts have already been filtered
        chrom_subset = None # Chromosomes have already been selected
        lengths = lengths_subset
        chrom_full = chrom_subset
    # Get neighbor counts (if needed)


    # OPTIONALLY INFER ALPHA VIA SINGLERES
    if first_alpha_loop and multiscale_rounds > 1:
        # for this infer() call: _counts, _bias, _struct_true, filter_threshold=0, multiscale_rounds=1, outdir, alpha=None, alpha_init, init

        # foar later infer() call: alpha=infer_param['alpha'], prev_alpha_obj=infer_param['alpha_obj'], alpha_loop=alpha_loop + 1, beta=infer_param['beta'], init=X_
        #                         max_alpha_loop?
        init, infer_param = infer(
            counts=_counts, lengths=lengths, ploidy=ploidy,
            outdir=os.path.join(outdir, 'singleres_alpha_inference'),
            alpha_init=alpha_init, seed=seed,
            filter_threshold=0, normalize=normalize, bias=_bias,
            update_alpha=update_alpha,
            beta_init=beta_init, max_alpha_loop=max_alpha_loop,
            beta=beta, multiscale_rounds=1,
            use_multiscale_variance=use_multiscale_variance,
            struct_draft_fullres=struct_draft_fullres, init=init,
            max_iter=max_iter, factr=factr, pgtol=pgtol,
            alpha_factr=alpha_factr, bcc_lambda=bcc_lambda,
            hsc_lambda=hsc_lambda, bcc_version=bcc_version, hsc_version=hsc_version,
            counts_inter_mv=counts_inter_mv, est_hmlg_sep=est_hmlg_sep,
            hsc_min_beads=hsc_min_beads, hsc_perc_diff=hsc_perc_diff,
            excluded_counts=excluded_counts, callback_freq=callback_freq,
            callback_fxns=callback_fxns, reorienter=reorienter,
            multiscale_reform=multiscale_reform, epsilon_min=epsilon_min,
            epsilon_max=epsilon_max, epsilon_coord_descent=epsilon_coord_descent,
            alpha_true=alpha_true, struct_true=_struct_true,
            input_weight=input_weight, exclude_zeros=exclude_zeros, null=null,
            chrom_full=chrom_full, chrom_subset=chrom_subset,
            mixture_coefs=mixture_coefs, verbose=verbose, mods=mods)
        alpha = infer_param['alpha']
        beta = infer_param['beta']
        prev_alpha_obj = infer_param['alpha_obj']
        alpha_loop = infer_param['alpha_loop']
        first_alpha_loop = False

    if alpha_loop is not None:
        _print_code_header([
            "Jointly inferring structure & alpha",
            f"Inferring STRUCTURE #{alpha_loop}"], max_length=80)

    # INFER DRAFT STRUCTURES (to obtain multiscale_variance & est_hmlg_sep)
    struct_draft_fullres_, est_hmlg_sep_, draft_converged = _infer_draft(
        counts, lengths=lengths, ploidy=ploidy, outdir=outdir_,
        alpha=alpha, seed=seed, filter_threshold=filter_threshold,
        normalize=normalize, bias=bias, beta=beta,
        multiscale_rounds=multiscale_rounds, beta_init=beta_init,
        use_multiscale_variance=use_multiscale_variance, init=init,
        max_iter=max_iter, factr=factr, pgtol=pgtol, hsc_lambda=hsc_lambda,
        hsc_version=hsc_version, est_hmlg_sep=est_hmlg_sep,
        hsc_min_beads=hsc_min_beads,
        struct_draft_fullres=struct_draft_fullres,
        callback_freq=callback_freq, callback_fxns=callback_fxns,
        reorienter=reorienter, multiscale_reform=multiscale_reform,
        alpha_true=alpha_true, struct_true=struct_true,
        input_weight=input_weight, exclude_zeros=exclude_zeros, null=null,
        chrom_full=chrom_full, chrom_subset=chrom_subset,
        mixture_coefs=mixture_coefs, verbose=verbose, mods=mods)
    if not draft_converged:
        return None, {'seed': seed, 'converged': draft_converged}

    # INFER FULL-RES STRUCTURE (with or without multires optimization)
    if first_alpha_loop and False:  # FIXME
        all_multiscale_factors = [1]
    else:
        all_multiscale_factors = 2 ** np.flip(
            np.arange(multiscale_rounds), axis=0)
    X_ = init
    epsilon_max_ = epsilon_max
    init_std_dev = None  # TODO
    for multiscale_factor in all_multiscale_factors:
        if verbose and len(all_multiscale_factors) > 1:
            _print_code_header(
                f'MULTISCALE FACTOR {multiscale_factor}', max_length=60,
                blank_lines=1)
        if multiscale_factor == 1:
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
            use_multiscale_variance=use_multiscale_variance,
            alpha_loop=alpha_loop, update_alpha=update_alpha,
            beta_init=beta_init,
            init=X_, max_iter=max_iter, factr=factr, pgtol=pgtol,
            alpha_factr=alpha_factr, bcc_lambda=bcc_lambda,
            hsc_lambda=hsc_lambda, bcc_version=bcc_version,
            hsc_version=hsc_version, counts_inter_mv=counts_inter_mv,
            est_hmlg_sep=est_hmlg_sep_, hsc_min_beads=hsc_min_beads,
            hsc_perc_diff=hsc_perc_diff,
            struct_draft_fullres=struct_draft_fullres_,
            callback_fxns=callback_fxns,
            callback_freq=callback_freq, reorienter=reorienter,
            multiscale_reform=multiscale_reform, epsilon_min=epsilon_min,
            epsilon_max=epsilon_max_,
            epsilon_coord_descent=epsilon_coord_descent,
            init_std_dev=init_std_dev, alpha_true=alpha_true,
            struct_true=struct_true, input_weight=input_weight,
            exclude_zeros=exclude_zeros, null=null,
            chrom_full=chrom_full, chrom_subset=chrom_subset,
            mixture_coefs=mixture_coefs, verbose=verbose, mods=mods)

        if not infer_param['converged']:
            return struct_, infer_param
        if update_alpha and multiscale_factor == 1 and (
                not infer_param['alpha_converged']):
            return struct_, infer_param

        if reorienter is not None and reorienter.reorient:
            X_ = infer_param['orient']
        else:
            X_ = struct_
        if 'epsilon' in infer_param:
            epsilon_max_ = infer_param['epsilon']  # FIXME??
        # if use_prev_std_dev and multiscale_reform:
        #     init_std_dev = infer_param['epsilon'] / np.sqrt(2)
        if 'lowres_exit' in mods:
            exit(0)

    # RE-INFER WITH NEW ALPHA
    if not update_alpha:
        return struct_, infer_param
    if (alpha_loop >= max_alpha_loop) or _convergence_criteria(
            f_k=prev_alpha_obj, f_kplus1=infer_param['alpha_obj'],
            factr=alpha_factr):
        update_alpha = False
    return infer(
        counts=counts, lengths=lengths, ploidy=ploidy,
        outdir=outdir, alpha=infer_param['alpha'], seed=seed,
        filter_threshold=filter_threshold, normalize=normalize, bias=bias,
        update_alpha=update_alpha, prev_alpha_obj=infer_param['alpha_obj'],
        beta_init=beta_init,
        alpha_loop=alpha_loop + 1, max_alpha_loop=max_alpha_loop,
        beta=infer_param['beta'], multiscale_rounds=multiscale_rounds,
        use_multiscale_variance=use_multiscale_variance,
        struct_draft_fullres=struct_draft_fullres, init=X_,
        max_iter=max_iter, factr=factr, pgtol=pgtol,
        alpha_factr=alpha_factr, bcc_lambda=bcc_lambda,
        hsc_lambda=hsc_lambda, bcc_version=bcc_version, hsc_version=hsc_version,
        counts_inter_mv=counts_inter_mv, est_hmlg_sep=est_hmlg_sep,
        hsc_min_beads=hsc_min_beads, hsc_perc_diff=hsc_perc_diff,
        excluded_counts=excluded_counts, callback_freq=callback_freq,
        callback_fxns=callback_fxns, reorienter=reorienter,
        multiscale_reform=multiscale_reform, epsilon_min=epsilon_min,
        epsilon_max=epsilon_max, epsilon_coord_descent=epsilon_coord_descent,
        alpha_true=alpha_true, struct_true=struct_true,
        input_weight=input_weight, exclude_zeros=exclude_zeros, null=null,
        chrom_full=chrom_full, chrom_subset=chrom_subset,
        mixture_coefs=mixture_coefs, verbose=verbose, mods=mods)


def pastis_poisson(counts, lengths, ploidy, outdir='', chromosomes=None,
                   chrom_subset=None, alpha=None, seed=0,
                   filter_threshold=0.04, normalize=True, bias=None,
                   alpha_init=-3, max_alpha_loop=20,
                   beta=None, multiscale_rounds=1, use_multiscale_variance=True,
                   max_iter=30000, factr=1e7, pgtol=1e-05,
                   alpha_factr=1e12, bcc_lambda=0, hsc_lambda=0,
                   bcc_version='2019', hsc_version='2019',
                   counts_inter_mv=None, est_hmlg_sep=None, hsc_min_beads=5,
                   hsc_perc_diff=None, struct_draft_fullres=None,
                   callback_fxns=None, print_freq=100, history_freq=100,
                   save_freq=None, piecewise=False, piecewise_step=None,
                   piecewise_chrom=None, piecewise_min_beads=5,
                   piecewise_fix_homo=False, piecewise_opt_orient=True,
                   piecewise_step3_multiscale=False,
                   piecewise_step1_accuracy=1,
                   multiscale_reform=False, epsilon_min=1e-6, epsilon_max=1e6,
                   epsilon_coord_descent=False, alpha_true=None,
                   struct_true=None, init='mds', input_weight=None,
                   exclude_zeros=False, null=False, mixture_coefs=None,
                   verbose=True, mods=[]):
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
    TODO add bcc_version, hsc_version, & counts_inter_mv
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

    if not isinstance(counts, list):
        counts = [counts]
    if verbose:
        print(f"\nRANDOM SEED = {seed:03d}", flush=True)
        if all([isinstance(c, str) for c in counts]):
            print(f"COUNTS: " + '        \n'.join(counts) + "\n", flush=True)

    callback_freq = {'print': print_freq, 'history': history_freq,
                     'save': save_freq}
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
            multiscale_rounds=multiscale_rounds,
            use_multiscale_variance=use_multiscale_variance, max_iter=max_iter,
            factr=factr, pgtol=pgtol, alpha_factr=alpha_factr,
            bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda,
            bcc_version=bcc_version, hsc_version=hsc_version,
            counts_inter_mv=counts_inter_mv,
            est_hmlg_sep=est_hmlg_sep, hsc_perc_diff=hsc_perc_diff,
            struct_draft_fullres=struct_draft_fullres,
            hsc_min_beads=hsc_min_beads, callback_fxns=callback_fxns,
            callback_freq=callback_freq,
            piecewise_step=piecewise_step,
            piecewise_chrom=piecewise_chrom,
            piecewise_min_beads=piecewise_min_beads,
            piecewise_fix_homo=piecewise_fix_homo,
            piecewise_opt_orient=piecewise_opt_orient,
            piecewise_step3_multiscale=piecewise_step3_multiscale,
            piecewise_step1_accuracy=piecewise_step1_accuracy,
            multiscale_reform=multiscale_reform,
            epsilon_coord_descent=epsilon_coord_descent, alpha_true=alpha_true,
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
            use_multiscale_variance=use_multiscale_variance,
            struct_draft_fullres=struct_draft_fullres,
            init=init, max_iter=max_iter, factr=factr, pgtol=pgtol,
            alpha_factr=alpha_factr, bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda,
            bcc_version=bcc_version, hsc_version=hsc_version,
            counts_inter_mv=counts_inter_mv, est_hmlg_sep=est_hmlg_sep,
            hsc_min_beads=hsc_min_beads, hsc_perc_diff=hsc_perc_diff,
            callback_freq=callback_freq, callback_fxns=callback_fxns,
            multiscale_reform=multiscale_reform, epsilon_min=epsilon_min,
            epsilon_max=epsilon_max, epsilon_coord_descent=epsilon_coord_descent,
            alpha_true=alpha_true, struct_true=struct_true,
            input_weight=input_weight, exclude_zeros=exclude_zeros, null=null,
            chrom_full=chromosomes, chrom_subset=chrom_subset,
            mixture_coefs=mixture_coefs, verbose=verbose, mods=mods)

    if verbose:
        if infer_param['converged']:
            print("INFERENCE COMPLETE: CONVERGED", flush=True)
        else:
            print("INFERENCE COMPLETE: DID NOT CONVERGE", flush=True)

    return struct_, infer_param
