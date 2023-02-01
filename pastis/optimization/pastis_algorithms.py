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
from .counts import _set_initial_beta
from .initialization import initialize
from .callbacks import Callback
from .constraints import prep_constraints
from .constraints import get_counts_interchrom, HomologSeparating2019
from .poisson import PastisPM, _convergence_criteria
from .multiscale_optimization import _choose_max_multiscale_factor
from .multiscale_optimization import decrease_lengths_res
from .multiscale_optimization import decrease_bias_res
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
                 null=False, chrom_full=None, chrom_subset=None,
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
        struct_true=struct_true, verbose=False)

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
            exclude_zeros=exclude_zeros)
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

        # Convert counts & betas & struct_true to "simplified" diploid
        beta_ambig = _ambiguate_beta(
            beta, counts=counts, lengths=lengths, ploidy=2)
        beta_draft = 2 * beta_ambig
        if beta_init_draft is not None:
            beta_init_draft = beta_init_draft * 2
        counts_draft = [ambiguate_counts(
            counts=counts, lengths=lengths, ploidy=2)]
        if struct_true_draft is not None:
            struct_true_draft = struct_true_draft.reshape(-1, 3)
            struct_true_draft = np.nanmean(  # FIXME is this even correct?
                [struct_true_draft[:int(struct_true_draft.shape[0] / 2)],
                 struct_true_draft[int(struct_true_draft.shape[0] / 2):]],
                axis=0)

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
        input_weight=input_weight, exclude_zeros=exclude_zeros, null=null,
        mixture_coefs=mixture_coefs, verbose=verbose, mods=mods)
    if not infer_param_lowres['converged']:
        return None, False

    if ploidy_draft == 2:
        est_hmlg_sep = distance_between_homologs(
            struct_draft_lowres, lengths=lengths,
            multiscale_factor=multires_factor_draft,
            mixture_coefs=mixture_coefs)
    else:
        est_hmlg_sep = distance_between_molecules(
            struct_draft_lowres, lengths=lengths, ploidy=ploidy_draft,
            multiscale_factor=multires_factor_draft,
            mixture_coefs=mixture_coefs)
    if verbose:
        print("Estimated distances between homolog centers of mass:"
              " " + " ".join([f'{x:.2g}' for x in est_hmlg_sep]), flush=True)
        if struct_true is not None:
            true_hmlg_sep = distance_between_homologs(
                structures=struct_true, lengths=lengths)
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
                    beta_init=None, init='mds', max_iter=30000,
                    factr=1e7, pgtol=1e-05, alpha_factr=1e12,
                    bcc_lambda=0, hsc_lambda=0, bcc_version='2019',
                    hsc_version='2019', data_interchrom=None, est_hmlg_sep=None,
                    hsc_min_beads=5, hsc_perc_diff=None, excluded_counts=None,
                    callback_freq=None, callback_fxns=None, reorienter=None,
                    multiscale_reform=False,
                    alpha_true=None, struct_true=None, input_weight=None,
                    exclude_zeros=False, null=False, chrom_full=None, chrom_subset=None,
                    mixture_coefs=None,
                    outfiles=None, verbose=True, mods=[]):
    """TODO"""

    # PREPARE COUNTS OBJECTS
    counts, struct_nan, fullres_struct_nan = preprocess_counts(
        counts=counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, exclude_zeros=exclude_zeros,
        beta=beta, bias=bias, input_weight=input_weight, verbose=verbose,
        excluded_counts=excluded_counts, multiscale_reform=multiscale_reform,
        mods=mods)

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
            bias, multiscale_factor=multiscale_factor, lengths=lengths)

    # INITIALIZATION
    random_state = np.random.RandomState(seed)
    struct_init = initialize(
        counts=counts, lengths=lengths, init=init, ploidy=ploidy,
        random_state=random_state,
        alpha=(alpha_init if alpha is None else alpha),
        bias=bias, multiscale_factor=multiscale_factor,
        reorienter=reorienter, mixture_coefs=mixture_coefs,
        struct_true=struct_true, verbose=verbose, mods=mods)
    if multiscale_reform and multiscale_factor != 1:
        epsilon = random_state.uniform()
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
        fullres_struct_nan=fullres_struct_nan, verbose=verbose, mods=mods)
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
        mixture_coefs=mixture_coefs, reorienter=reorienter, **callback_freq,
        **callback_fxns, verbose=verbose, mods=mods)

    return (counts, bias, struct_nan, struct_init, constraints, callback,
            epsilon, ploidy)


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
    excluded_counts : {"inter", "intra"}, optional
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
            if verbose:
                if os.path.isfile(outfiles['struct_infer']):
                    print('CONVERGED\n', flush=True)
                    struct_ = np.loadtxt(outfiles['struct_infer'])
                elif os.path.isfile(outfiles['struct_nonconv']):
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
        struct_true=struct_true, verbose=verbose)

    # PREP FOR INFERENCE
    prepped = _prep_inference(
        counts, lengths=lengths, ploidy=ploidy, outdir=outdir, alpha=alpha,
        seed=seed, filter_threshold=filter_threshold, normalize=normalize,
        bias=bias, alpha_init=alpha_init, max_alpha_loop=max_alpha_loop,
        beta=beta, multiscale_factor=multiscale_factor, beta_init=beta_init, init=init,
        max_iter=max_iter, factr=factr, pgtol=pgtol, alpha_factr=alpha_factr,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda, bcc_version=bcc_version,
        hsc_version=hsc_version, data_interchrom=data_interchrom,
        est_hmlg_sep=est_hmlg_sep, hsc_min_beads=hsc_min_beads,
        hsc_perc_diff=hsc_perc_diff, excluded_counts=excluded_counts,
        callback_freq=callback_freq, callback_fxns=callback_fxns, reorienter=reorienter,
        multiscale_reform=multiscale_reform,
        alpha_true=alpha_true, struct_true=struct_true, input_weight=input_weight,
        exclude_zeros=exclude_zeros, null=null,
        chrom_full=chrom_full, chrom_subset=chrom_subset, mixture_coefs=mixture_coefs,
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
        null=null, mixture_coefs=mixture_coefs, verbose=verbose, mods=mods)
    pm.fit_structure(alpha_loop=alpha_loop)
    struct_ = pm.struct_.reshape(-1, 3)
    struct_[struct_nan] = np.nan

    # OPTIONALLY RE-INFER ALPHA
    alpha_converged_ = None
    if update_alpha and multiscale_factor == 1:
        _print_code_header([
            "Jointly inferring structure & alpha",
            f"Inferring ALPHA #{alpha_loop}"], max_length=80,
            verbose=verbose and alpha_loop is not None)
        pm.fit_alpha(alpha_loop=alpha_loop)
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
        'alpha_loop': alpha_loop, 'rescale_by': rescale_by,
        'est_hmlg_sep': est_hmlg_sep}
    # if constraints is not None:  # TODO remove
    #     hsc19 = [x for x in constraints if (
    #         isinstance(x, HomologSeparating2019) and x.lambda_val > 0)]
    #     if len(hsc19) == 1:
    #         infer_param['est_hmlg_sep'] = hsc19[0].hparams['est_hmlg_sep']
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
            if pm.log_ is not None and len(pm.log_) > 0:
                pd.DataFrame(pm.log_).to_csv(
                    outfiles['log'], sep='\t', index=False)
        else:
            np.savetxt(outfiles['struct_nonconv'], struct_)

    if pm.converged_:
        return struct_, infer_param
    else:
        return None, infer_param


def infer(counts, lengths, ploidy, outdir='', alpha=None, seed=0,
          filter_threshold=0.04, normalize=True, bias=None, alpha_init=None,
          max_alpha_loop=20, beta=None, multiscale_rounds=1,
          alpha_loop=None, update_alpha=False, prev_alpha_obj=None,
          beta_init=None, init='mds', max_iter=30000,
          factr=1e7, pgtol=1e-05, alpha_factr=1e12,
          bcc_lambda=0, hsc_lambda=0, bcc_version='2019', hsc_version='2019',
          data_interchrom=None, est_hmlg_sep=None, hsc_min_beads=5,
          hsc_perc_diff=None, excluded_counts=None,
          callback_freq=None, callback_fxns=None, reorienter=None,
          multiscale_reform=False, epsilon_min=1e-2, epsilon_max=1e6,
          alpha_true=None, struct_true=None,
          input_weight=None, exclude_zeros=False, null=False,
          chrom_full=None, chrom_subset=None,
          mixture_coefs=None, verbose=True, mods=[]):
    """TODO"""

    # SETUP
    if alpha is None:
        update_alpha = True
        if alpha_init is None:
            random_state = np.random.RandomState(seed)
            alpha_init = random_state.uniform(low=-4, high=-1)
    if update_alpha and alpha_loop is None:
        alpha_loop = 1
    if update_alpha and outdir is not None:
        outdir_ = os.path.join(
            outdir, f"alpha_coord_descent.try{alpha_loop:03d}")
    else:
        outdir_ = outdir
    first_alpha_loop = update_alpha and alpha_loop == 1
    if multiscale_rounds <= 0:
        multiscale_rounds = 1
    if verbose:
        if first_alpha_loop and alpha is None and multiscale_rounds == 1:
            _print_code_header([
                "JOINTLY INFERRING STRUCTURE + ALPHA",
                f"Initial alpha = {alpha_init:.3g}"], max_length=100)
        elif alpha_loop is None:
            _print_code_header([
                "INFERRING STRUCTURE WITH FIXED ALPHA",
                f"Fixed alpha = {alpha:.3g}"], max_length=100)
    if alpha is None:
        alpha = alpha_init

    # LOAD DATA
    if first_alpha_loop or multiscale_rounds == 1:
        _counts, _bias, lengths_subset, chrom_full, _, _, _struct_true = load_data(
            counts=counts, lengths_full=lengths, ploidy=ploidy,
            chrom_full=chrom_full, chrom_subset=chrom_subset,
            filter_threshold=filter_threshold, normalize=normalize, bias=bias,
            struct_true=struct_true, verbose=verbose)
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
    if data_interchrom is None and (hsc_lambda > 0 and hsc_version == '2022'):
        data_interchrom = get_counts_interchrom(
            counts, lengths=lengths, ploidy=ploidy,
            filter_threshold=filter_threshold, normalize=normalize,
            bias=bias, multiscale_reform=multiscale_reform,
            multiscale_rounds=multiscale_rounds, verbose=verbose, mods=mods)
    # No need to repeatedly re-load if inferring with single-res
    if multiscale_rounds == 1:
        counts = _counts
        bias = _bias
        struct_true = _struct_true
        filter_threshold = 0  # Counts have already been filtered
        chrom_subset = None  # Chromosomes have already been selected
        lengths = lengths_subset
        chrom_full = chrom_subset

    # OPTIONALLY INFER ALPHA VIA SINGLERES
    init_ = init
    est_hmlg_sep_ = est_hmlg_sep
    if first_alpha_loop and multiscale_rounds > 1:
        if outdir is None:
            outdir_1x_alpha = None
        else:
            outdir_1x_alpha = os.path.join(outdir, 'singleres_alpha_inference')

        init_, infer_param = infer(
            counts=_counts, lengths=lengths, ploidy=ploidy,
            outdir=outdir_1x_alpha, alpha_init=alpha_init, seed=seed,
            filter_threshold=0, normalize=normalize, bias=_bias,
            update_alpha=update_alpha,
            beta_init=beta_init, max_alpha_loop=max_alpha_loop,
            beta=beta, multiscale_rounds=1, init=init,
            max_iter=max_iter, factr=factr, pgtol=pgtol,
            alpha_factr=alpha_factr, bcc_lambda=bcc_lambda,
            hsc_lambda=hsc_lambda, bcc_version=bcc_version, hsc_version=hsc_version,
            data_interchrom=data_interchrom, est_hmlg_sep=est_hmlg_sep,
            hsc_min_beads=hsc_min_beads, hsc_perc_diff=hsc_perc_diff,
            excluded_counts=excluded_counts, callback_freq=callback_freq,
            callback_fxns=callback_fxns, reorienter=reorienter,
            multiscale_reform=multiscale_reform, epsilon_min=epsilon_min,
            epsilon_max=epsilon_max, alpha_true=alpha_true,
            struct_true=_struct_true, input_weight=input_weight,
            exclude_zeros=exclude_zeros, null=null, chrom_full=chrom_full,
            chrom_subset=chrom_subset, mixture_coefs=mixture_coefs,
            verbose=verbose, mods=mods)

        # Do not continue unless inference converged
        if not infer_param['converged']:
            return init_, infer_param
        if ('alpha_converged' in infer_param) and (
                infer_param['alpha_converged'] is not None) and (
                not infer_param['alpha_converged']):
            return init_, infer_param

        alpha = infer_param['alpha']
        beta = infer_param['beta']
        est_hmlg_sep_ = infer_param['est_hmlg_sep']
        alpha_loop = infer_param['alpha_loop']
        if outdir is not None:
            outdir_ = os.path.join(
                outdir, f"alpha_coord_descent.try{alpha_loop:03d}")
        prev_alpha_obj = None
        first_alpha_loop = False

    _print_code_header([
        "Jointly inferring structure & alpha",
        f"Inferring STRUCTURE #{alpha_loop}"], max_length=80,
        verbose=verbose and alpha_loop is not None)

    # INFER DRAFT STRUCTURES (to obtain est_hmlg_sep for hsc2019)
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
        input_weight=input_weight, exclude_zeros=exclude_zeros, null=null,
        chrom_full=chrom_full, chrom_subset=chrom_subset,
        mixture_coefs=mixture_coefs, verbose=verbose, mods=mods)
    if not draft_converged:  # Do not continue unless inference converged
        return None, {'seed': seed, 'converged': draft_converged}

    # INFER FULL-RES STRUCTURE (with or without multires optimization)
    all_multiscale_factors = 2 ** np.flip(np.arange(int(multiscale_rounds)))
    epsilon_max_ = epsilon_max
    for multiscale_factor in all_multiscale_factors:
        _print_code_header(
            f'MULTISCALE FACTOR {multiscale_factor}', max_length=60,
            blank_lines=1, verbose=verbose and len(all_multiscale_factors) > 1)
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
            callback_fxns=callback_fxns,
            callback_freq=callback_freq, reorienter=reorienter,
            multiscale_reform=multiscale_reform, epsilon_min=epsilon_min,
            epsilon_max=epsilon_max_, alpha_true=alpha_true,
            struct_true=struct_true, input_weight=input_weight,
            exclude_zeros=exclude_zeros, null=null,
            chrom_full=chrom_full, chrom_subset=chrom_subset,
            mixture_coefs=mixture_coefs, verbose=verbose, mods=mods)

        # Do not continue unless inference converged
        if not infer_param['converged']:
            return struct_, infer_param
        if update_alpha and multiscale_factor == 1 and (
                not infer_param['alpha_converged']):
            return struct_, infer_param

        if reorienter is not None and reorienter.reorient:
            init_ = infer_param['orient']
        else:
            init_ = struct_
        if 'epsilon' in infer_param and infer_param['epsilon'] is not None:
            # Epsilon for the next-highest resolution should be smaller than
            # the previous resolution's epsilon... but allow some wiggle room
            epsilon_max_ = infer_param['epsilon'] * 1.5
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
        init=init_, max_iter=max_iter, factr=factr, pgtol=pgtol,
        alpha_factr=alpha_factr, bcc_lambda=bcc_lambda,
        hsc_lambda=hsc_lambda, bcc_version=bcc_version, hsc_version=hsc_version,
        data_interchrom=data_interchrom, est_hmlg_sep=est_hmlg_sep,
        hsc_min_beads=hsc_min_beads, hsc_perc_diff=hsc_perc_diff,
        excluded_counts=excluded_counts, callback_freq=callback_freq,
        callback_fxns=callback_fxns, reorienter=reorienter,
        multiscale_reform=multiscale_reform, epsilon_min=epsilon_min,
        epsilon_max=epsilon_max, alpha_true=alpha_true, struct_true=struct_true,
        input_weight=input_weight, exclude_zeros=exclude_zeros, null=null,
        chrom_full=chrom_full, chrom_subset=chrom_subset,
        mixture_coefs=mixture_coefs, verbose=verbose, mods=mods)


def pastis_poisson(counts, lengths, ploidy, outdir='', chromosomes=None,
                   chrom_subset=None, alpha=None, seed=0,
                   filter_threshold=0.04, normalize=True, bias=None,
                   alpha_init=None, max_alpha_loop=20,
                   beta=None, multiscale_rounds=1,
                   max_iter=30000, factr=1e7, pgtol=1e-05,
                   alpha_factr=1e12, bcc_lambda=0, hsc_lambda=0,
                   bcc_version='2019', hsc_version='2019',
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
            alpha_factr=alpha_factr, bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda,
            bcc_version=bcc_version, hsc_version=hsc_version,
            data_interchrom=data_interchrom, est_hmlg_sep=est_hmlg_sep,
            hsc_min_beads=hsc_min_beads, hsc_perc_diff=hsc_perc_diff,
            callback_freq=callback_freq, callback_fxns=callback_fxns,
            multiscale_reform=multiscale_reform, epsilon_min=epsilon_min,
            epsilon_max=epsilon_max, alpha_true=alpha_true, struct_true=struct_true,
            input_weight=input_weight, exclude_zeros=exclude_zeros, null=null,
            chrom_full=chromosomes, chrom_subset=chrom_subset,
            mixture_coefs=mixture_coefs, verbose=verbose, mods=mods)

    if verbose:
        if infer_param['converged']:
            print("INFERENCE COMPLETE: CONVERGED", flush=True)
        else:
            print("INFERENCE COMPLETE: DID NOT CONVERGE", flush=True)

    return struct_, infer_param
