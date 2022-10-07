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
from .utils_poisson import _output_subdir
from .counts import preprocess_counts, ambiguate_counts, _ambiguate_beta
from .counts import check_counts
from .initialization import initialize
from .callbacks import Callback
from .constraints import prep_constraints, distance_between_homologs
from .poisson import PastisPM
from .multiscale_optimization import get_multiscale_variances_from_struct
from .multiscale_optimization import get_multiscale_epsilon_from_struct
from .multiscale_optimization import _choose_max_multiscale_factor
from .multiscale_optimization import decrease_lengths_res
from ..io.read import load_data


def _infer_draft(counts_raw, lengths, ploidy, outdir=None, alpha=None, seed=0,
                 normalize=True, filter_threshold=0.04, alpha_init=-3.,
                 max_alpha_loop=20, beta=None, multiscale_rounds=1,
                 use_multiscale_variance=True, init='mds', max_iter=30000,
                 factr=1e7, pgtol=1e-05, alpha_factr=1e12,
                 hsc_lambda=0., hsc_version='2019', est_hmlg_sep=None,
                 hsc_min_beads=5, struct_draft_fullres=None, callback_freq=None,
                 callback_fxns=None, reorienter=None,
                 multiscale_reform=False, alpha_true=None,
                 struct_true=None, input_weight=None, exclude_zeros=False,
                 null=False, mixture_coefs=None, verbose=True, mods=[]):
    """Infer draft 3D structures with PASTIS via Poisson model.
    """

    infer_draft_lowres = est_hmlg_sep is None and hsc_lambda > 0 and str(
        hsc_version) == '2019'
    need_multiscale_var = use_multiscale_variance and (
        multiscale_rounds > 1 or infer_draft_lowres) and (not multiscale_reform)
    infer_draft_fullres = struct_draft_fullres is None and (
        need_multiscale_var)

    multiscale_factor_for_lowres = _choose_max_multiscale_factor(
        lengths=lengths, min_beads=hsc_min_beads)

    if verbose:
        if (infer_draft_fullres and infer_draft_lowres):
            _print_code_header(
                'INFERRING DRAFT STRUCTURES', max_length=80, blank_lines=2)
        elif infer_draft_fullres:
            _print_code_header(
                ['INFERRING DRAFT STRUCTURE', 'Full resolution'],
                max_length=80, blank_lines=2)
        elif infer_draft_lowres:
            _print_code_header(
                ['INFERRING DRAFT STRUCTURE',
                    'Low resolution (%dx)' % multiscale_factor_for_lowres],
                max_length=80, blank_lines=2)

    if not infer_draft_fullres or infer_draft_lowres:
        None, alpha, beta, None, True

    counts, _, _, _ = preprocess_counts(
        counts_raw=counts_raw, lengths=lengths, ploidy=ploidy,
        normalize=normalize, filter_threshold=filter_threshold,
        multiscale_factor=1, exclude_zeros=exclude_zeros, beta=beta,
        input_weight=input_weight, verbose=False, mixture_coefs=mixture_coefs,
        mods=mods)
    beta = [c.beta for c in counts if c.sum() != 0]

    alpha_ = alpha
    beta_ = beta
    if infer_draft_fullres:
        if verbose and infer_draft_lowres:
            _print_code_header(
                "Inferring full-res draft structure",
                max_length=60, blank_lines=1)
        if outdir is None:
            fullres_outdir = None
        else:
            fullres_outdir = os.path.join(outdir, 'struct_draft_fullres')
        struct_draft_fullres, infer_param_fullres = infer(
            counts_raw=counts_raw, outdir=fullres_outdir, lengths=lengths,
            ploidy=ploidy, alpha=alpha, seed=seed, normalize=normalize,
            filter_threshold=filter_threshold, alpha_init=alpha_init,
            max_alpha_loop=max_alpha_loop, beta=beta, init=init,
            max_iter=max_iter, factr=factr, pgtol=pgtol,
            alpha_factr=alpha_factr, draft=True, simple_diploid=(ploidy == 2),
            callback_fxns=callback_fxns, callback_freq=callback_freq,
            reorienter=reorienter, multiscale_reform=multiscale_reform,
            alpha_true=alpha_true, struct_true=struct_true,
            input_weight=input_weight, exclude_zeros=exclude_zeros, null=null,
            mixture_coefs=mixture_coefs, verbose=verbose, mods=mods)
        if not infer_param_fullres['converged']:
            return struct_draft_fullres, alpha_, beta_, est_hmlg_sep, False
        if alpha is not None:
            alpha_ = infer_param_fullres['alpha']
            if ploidy == 2:  # TODO add this line on the main branch too
                beta_ = list(infer_param_fullres['beta'] * np.array(
                    beta) / (2 * _ambiguate_beta(
                        beta, counts=counts_raw, lengths=lengths, ploidy=2)))

    if infer_draft_lowres:
        if verbose and infer_draft_fullres:
            _print_code_header(
                "Inferring low-res draft structure (%dx)"
                % multiscale_factor_for_lowres,
                max_length=60, blank_lines=1)
        if ploidy == 1:
            raise ValueError("Can not apply homolog-separating constraint"
                             " to haploid data.")
        #if alpha_ is None:
        #    raise ValueError("Alpha must be set prior to inferring r from"
        #                     " counts data")
        if outdir is None:
            lowres_outdir = None
        else:
            lowres_outdir = os.path.join(outdir, 'struct_draft_lowres')
        ua_index = [i for i in range(len(
            counts_raw)) if counts_raw[i].shape == (lengths.sum() * ploidy,
                                                    lengths.sum() * ploidy)]
        if len(ua_index) == 1:
            counts_for_lowres = [counts_raw[ua_index[0]]]
            simple_diploid_for_lowres = False
            beta_for_lowres = [beta[ua_index[0]]]
        elif len(ua_index) > 1:
            raise ValueError("Only input one matrix of unambiguous counts."
                             " Please pool unambiguos counts before"
                             " inputting.")
        else:
            if lengths.shape[0] == 1:
                raise ValueError("Please input more than one chromosome to"
                                 " estimate est_hmlg_sep from ambiguous data.")
            counts_for_lowres = counts_raw
            simple_diploid_for_lowres = True
            beta_for_lowres = beta
        struct_draft_lowres, infer_param_lowres = infer(
            counts_raw=counts_for_lowres, outdir=lowres_outdir,
            lengths=lengths, ploidy=ploidy, alpha=alpha_,
            seed=seed, normalize=normalize,
            filter_threshold=filter_threshold, beta=beta_for_lowres,
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
            return struct_draft_fullres, alpha_, beta_, est_hmlg_sep, False
        est_hmlg_sep = distance_between_homologs(
            structures=struct_draft_lowres,
            lengths=decrease_lengths_res(
                lengths=lengths,
                multiscale_factor=multiscale_factor_for_lowres),
            mixture_coefs=mixture_coefs,
            simple_diploid=simple_diploid_for_lowres)
        if verbose:
            print("Estimated distance between homolog barycenters for each"
                  " chromosome: %s" % ' '.join(map(str, est_hmlg_sep.round(2))),
                  flush=True)

    return struct_draft_fullres, alpha_, beta_, est_hmlg_sep, True


def _get_tmp_fullres(counts_raw, lengths, ploidy, struct_infer, outdir='',
                     alpha=None, seed=0, normalize=True, filter_threshold=0.04,
                     alpha_init=-3., max_alpha_loop=20, beta=None,
                     multiscale_factor=1, max_iter=30000, factr=1e7,
                     pgtol=1e-05, alpha_factr=1e12, callback_freq=None,
                     callback_fxns=None, alpha_true=None, struct_true=None,
                     input_weight=None, exclude_zeros=False, null=False,
                     mixture_coefs=None, verbose=True, mods=[]):

    if outdir is not None:
        outfiles = _get_output_files(
            os.path.join(outdir, 'temp_idk'), seed=seed)
        if os.path.exists(outfiles['struct_infer']) or os.path.exists(
                outfiles['struct_nonconv']):
            if verbose:
                if os.path.exists(outfiles['struct_infer']):
                    print('CONVERGED\n', flush=True)
                elif os.path.exists(outfiles['struct_nonconv']):
                    print('OPTIMIZATION DID NOT CONVERGE\n', flush=True)
            infer_param = _load_infer_param(outfiles['infer_param'])
            struct_ = np.loadtxt(outfiles['struct_infer'])
            return struct_, infer_param
    else:
        outfiles = None

    # PREPARE COUNTS OBJECTS
    counts_mini, bias, struct_nan_mini, _ = preprocess_counts(
        counts_raw=counts_raw, lengths=lengths, ploidy=ploidy,
        normalize=normalize, filter_threshold=filter_threshold,
        multiscale_factor=1, exclude_zeros=exclude_zeros, beta=beta,
        input_weight=input_weight, excluded_counts=multiscale_factor,
        diploid_to_unambig=True, verbose=False, mixture_coefs=mixture_coefs,
        mods=mods)

    # INITIALIZATION
    random_state = np.random.RandomState(seed)
    struct_init = initialize(
        counts=counts_mini, lengths=lengths, init=struct_infer, ploidy=ploidy,
        random_state=random_state, alpha=alpha_init if alpha is None else alpha,
        bias=bias, multiscale_factor=1, mixture_coefs=mixture_coefs,
        verbose=False, mods=mods)

    # SETUP CALLBACKS
    if callback_fxns is None:
        callback_fxns = {}
    callback = Callback(
        lengths=lengths, ploidy=ploidy, counts=counts_mini, bias=bias,
        multiscale_factor=1, frequency=callback_freq, directory=outdir,
        seed=seed, struct_true=struct_true, alpha_true=alpha_true,
        mixture_coefs=mixture_coefs, **callback_fxns, verbose=False, mods=mods)

    # INFER TEMPORARY FULLRES STRUCT
    _infer_struct(
        counts=counts_mini, lengths=lengths, ploidy=ploidy, alpha=alpha,
        struct_init=struct_init, struct_nan=struct_nan_mini, bias=bias,
        callback=callback, multiscale_factor=1, max_iter=max_iter, factr=factr,
        pgtol=pgtol, alpha_init=alpha_init, max_alpha_loop=max_alpha_loop,
        alpha_factr=alpha_factr, null=null, seed=seed, outfiles=outfiles,
        mixture_coefs=mixture_coefs, verbose=verbose, mods=mods)


    # # INFER ALPHA  #....probably use pm.infer_alpha() for this or something
    # _infer_struct(
    #     counts=counts, lengths=lengths, ploidy=ploidy, alpha=alpha,
    #     struct_init=struct_init, struct_nan=struct_nan, bias=bias,
    #     constraints=constraints, callback=callback,
    #     multiscale_factor=1, alpha_init=alpha_init,
    #     max_alpha_loop=max_alpha_loop, max_iter=max_iter, factr=factr,
    #     pgtol=pgtol, alpha_factr=alpha_factr,
    #     null=null, seed=seed, outfiles=outfiles, mixture_coefs=mixture_coefs,
    #     verbose=verbose, mods=mods)



def _prep_inference(counts_raw, lengths, ploidy, outdir='', alpha=None, seed=0,
                    normalize=True, filter_threshold=0.04, alpha_init=-3.,
                    max_alpha_loop=20, beta=None, multiscale_factor=1,
                    multiscale_rounds=1, use_multiscale_variance=True,
                    final_multiscale_round=False, init='mds', max_iter=30000,
                    factr=1e7, pgtol=1e-05, alpha_factr=1e12,
                    bcc_lambda=0., hsc_lambda=0., bcc_version='2019',
                    hsc_version='2019', counts_interchrom=None, est_hmlg_sep=None,
                    hsc_min_beads=5, hsc_perc_diff=None, excluded_counts=None,
                    struct_draft_fullres=None, draft=False, simple_diploid=False,
                    callback_freq=None, callback_fxns=None, reorienter=None,
                    multiscale_reform=False, init_std_dev=None,
                    alpha_true=None, struct_true=None, input_weight=None,
                    exclude_zeros=False, null=False, mixture_coefs=None,
                    outfiles=None, verbose=True, mods=[]):
    """TODO
    """

    # INFER DRAFT STRUCTURES (for estimation of multiscale_variance & est_hmlg_sep)
    alpha_ = alpha
    beta_ = beta
    if draft and alpha_ is None:
        alpha_ = alpha_init
    if multiscale_factor == 1 and not (draft or simple_diploid):
        infer_draft_lowres = est_hmlg_sep is None and hsc_lambda > 0 and str(
            hsc_version) == '2019'
        need_multiscale_var = use_multiscale_variance and (
            multiscale_rounds > 1 or infer_draft_lowres) and (
            not multiscale_reform)
        infer_draft_fullres = struct_draft_fullres is None and (
            need_multiscale_var)
        struct_draft_fullres, alpha_, beta_, est_hmlg_sep, draft_converged = _infer_draft(
            counts_raw, lengths=lengths, ploidy=ploidy, outdir=outdir,
            alpha=alpha, seed=seed, normalize=normalize,
            filter_threshold=filter_threshold, alpha_init=alpha_init,
            max_alpha_loop=max_alpha_loop, beta=beta,
            multiscale_rounds=multiscale_rounds,
            use_multiscale_variance=use_multiscale_variance, init=init,
            max_iter=max_iter, factr=factr, pgtol=pgtol,
            alpha_factr=alpha_factr, hsc_lambda=hsc_lambda,
            hsc_version=hsc_version, est_hmlg_sep=est_hmlg_sep,
            hsc_min_beads=hsc_min_beads,
            struct_draft_fullres=struct_draft_fullres,
            callback_freq=callback_freq, callback_fxns=callback_fxns,
            reorienter=reorienter, multiscale_reform=multiscale_reform,
            alpha_true=alpha_true, struct_true=struct_true,
            input_weight=input_weight, exclude_zeros=exclude_zeros, null=null,
            mixture_coefs=mixture_coefs, verbose=verbose, mods=mods)
        if not draft_converged:
            return None, {'alpha': alpha_, 'beta': beta_, 'seed': seed,
                          'converged': draft_converged}
        elif verbose and (infer_draft_fullres or infer_draft_lowres):
            _print_code_header(
                ['Draft inference complete', 'INFERRING STRUCTURE'],
                max_length=80, blank_lines=2)

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
    if simple_diploid:
        if ploidy != 2:
            raise ValueError("Ploidy is not 2, but simple_diploid specified.")
        counts_raw = check_counts(
            counts_raw, lengths=lengths, ploidy=2, exclude_zeros=exclude_zeros)
        beta_ = 2 * _ambiguate_beta(
            beta_, counts=counts_raw, lengths=lengths, ploidy=2)
        counts_raw = [ambiguate_counts(
            counts=counts_raw, lengths=lengths, ploidy=2,
            exclude_zeros=exclude_zeros)]
        ploidy = 1
    if excluded_counts is None and 'diag8' in mods:
        excluded_counts = 8
    counts, bias, struct_nan, fullres_struct_nan = preprocess_counts(
        counts_raw=counts_raw, lengths=lengths, ploidy=ploidy,
        normalize=normalize, filter_threshold=filter_threshold,
        multiscale_factor=multiscale_factor, exclude_zeros=exclude_zeros,
        beta=beta_, input_weight=input_weight, verbose=verbose,
        excluded_counts=excluded_counts, mixture_coefs=mixture_coefs, mods=mods)
    if verbose:
        print('BETA: %s' % ', '.join(
            ['%s=%.3g' % (c.ambiguity, c.beta) for c in counts if c.sum() != 0]),
            flush=True)
        if alpha_ is None:
            print('ALPHA: to be inferred, init = %.3g' % alpha_init, flush=True)
        else:
            print('ALPHA: %.3g' % alpha_, flush=True)

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

    if multiscale_rounds <= 1 or multiscale_factor > 1 or final_multiscale_round:
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
            alpha=alpha_init if alpha_ is None else alpha_,
            bias=bias, multiscale_factor=multiscale_factor,
            reorienter=reorienter, std_dev=init_std_dev,
            mixture_coefs=mixture_coefs, verbose=verbose, mods=mods)
        if multiscale_reform and multiscale_factor != 1:
            epsilon = random_state.rand()
        else:
            epsilon = None

        # # TEST
        # if isinstance(init, np.ndarray) and np.array_equal(init, struct_true):
        #     from .multiscale_optimization import increase_struct_res_gaussian
        #     test_struct_fullres = increase_struct_res_gaussian(
        #         struct_init, current_multiscale_factor=multiscale_factor,
        #         final_multiscale_factor=1, lengths=lengths,
        #         std_dev=epsilon_true / np.sqrt(2))
        #     epsilon_true_test = np.mean(get_multiscale_epsilon_from_struct(
        #         test_struct_fullres, lengths=lengths,
        #         multiscale_factor=multiscale_factor, verbose=False))
        #     print(f"True epsilon TEST ({multiscale_factor}x):"
        #           f" {epsilon_true_test:.3g}", flush=True)
        #     exit(0)

        # SETUP CONSTRAINTS
        constraints = prep_constraints(
            counts=counts, lengths=lengths, ploidy=ploidy,
            multiscale_factor=multiscale_factor, bcc_lambda=bcc_lambda,
            hsc_lambda=hsc_lambda, bcc_version=bcc_version,
            hsc_version=hsc_version, counts_interchrom=counts_interchrom,
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
            alpha_true=alpha_true, constraints=constraints,
            mixture_coefs=mixture_coefs, **callback_fxns,
            multiscale_variances=multiscale_variances, verbose=verbose,
            mods=mods)
    else:
        struct_init = None
        epsilon = None
        constraints = None
        callback = None

    return (counts, alpha_, beta_, bias, struct_nan, struct_init, constraints,
            est_hmlg_sep, callback, struct_draft_fullres, multiscale_variances,
            epsilon)


def _infer_struct(counts, lengths, ploidy, alpha, struct_init, struct_nan,
                  bias=None, constraints=None, callback=None,
                  multiscale_factor=1, multiscale_variances=None, epsilon=None,
                  epsilon_min=1e-6, epsilon_max=1e6,
                  epsilon_coord_descent=False, multiscale_rounds=1,
                  final_multiscale_round=False, alpha_init=-3,
                  max_alpha_loop=20, max_iter=30000, factr=1e7, pgtol=1e-05,
                  alpha_factr=1e12, reorienter=None, null=False, seed=None,
                  outfiles=None, mixture_coefs=None, verbose=True, mods=[]):
    """TODO"""

    # INFER STRUCTURE
    original_counts_beta = [c.beta for c in counts if c.sum() != 0]
    pm = PastisPM(
        counts=counts, lengths=lengths, ploidy=ploidy, alpha=alpha,
        init=struct_init, bias=bias, constraints=constraints,
        callback=callback, multiscale_factor=multiscale_factor,
        multiscale_variances=multiscale_variances, epsilon=epsilon,
        epsilon_bounds=[epsilon_min, epsilon_max],
        epsilon_coord_descent=epsilon_coord_descent, alpha_init=alpha_init,
        max_alpha_loop=max_alpha_loop, max_iter=max_iter, factr=factr,
        pgtol=pgtol, alpha_factr=alpha_factr, reorienter=reorienter,
        null=null, mixture_coefs=mixture_coefs, verbose=verbose, mods=mods)
    pm.fit()
    struct_ = pm.struct_.reshape(-1, 3)
    struct_[struct_nan] = np.nan

    # # If inferred epsilon is near epsilon_max, rerun with higher epsilon_max
    # # FIXME not sure about whether to include this
    # if (pm.epsilon_ is not None) and multiscale_factor == 2 ** (multiscale_rounds - 1):
    #     i = 1
    #     while pm.epsilon_ >= epsilon_max * 0.9:
    #         i += 1
    #         epsilon_max *= 2
    #         if verbose:
    #             _print_code_header(
    #                 [f'RE-INFERRING: MULTISCALE FACTOR {multiscale_factor}',
    #                  f'Try {i}, new epsilon_max={epsilon_max:.3g}'],
    #                 max_length=60, blank_lines=1)
    #         pm.epsilon_bounds = [epsilon_min, epsilon_max]
    #         pm.fit()

    # RE-SCALE STRUCTURE TO MATCH ORIGINAL BETA
    if alpha is None and (
            multiscale_rounds <= 1 or final_multiscale_round):
        original_beta = _ambiguate_beta(
            original_counts_beta, counts=counts, lengths=lengths,
            ploidy=ploidy)
        final_beta = _ambiguate_beta(
            pm.beta_, counts=counts, lengths=lengths, ploidy=ploidy)
        struct_ *= (original_beta / final_beta)
        pm.beta_ = original_counts_beta

    # SAVE RESULTS
    infer_param = {'alpha': pm.alpha_, 'beta': pm.beta_, 'obj': pm.obj_,
                 'seed': seed, 'converged': pm.converged_,
                 'conv_desc': pm.conv_desc_, 'time': pm.time_elapsed_,
                 'epsilon': pm.epsilon_,
                 'multiscale_variances': multiscale_variances}
    if hsc_lambda > 0:
        infer_param['est_hmlg_sep'] = est_hmlg_sep
    if reorienter is not None and reorienter.reorient:
        infer_param['orient'] = pm.orientation_.flatten()

    if outfiles is not None:
        os.makedirs(outfiles['dir'], exist_ok=True)
        with open(outfiles['infer_param'], 'w') as f:
            for k, v in infer_param.items():
                if isinstance(v, np.ndarray) or isinstance(v, list):
                    f.write(
                        '%s\t%s\n' % (k, ' '.join(['%g' % x for x in v])))
                elif v is not None:
                    f.write(f'{k}\t{v}\n')
        if reorienter is not None and reorienter.reorient:
            np.savetxt(outfiles['reorient'], pm.orientation_)
        if pm.converged_:
            np.savetxt(outfiles['struct_infer'], struct_)
            if pm.history_ is not None:
                pd.DataFrame(pm.history_).to_csv(
                    outfiles['history'], sep='\t', index=False)
        else:
            np.savetxt(outfiles['struct_nonconv'], struct_)

    if pm.converged_:
        return struct_, infer_param
    else:
        return None, infer_param


def infer(counts_raw, lengths, ploidy, outdir='', alpha=None, seed=0,
          normalize=True, filter_threshold=0.04, alpha_init=-3.,
          max_alpha_loop=20, beta=None, multiscale_factor=1,
          multiscale_rounds=1, use_multiscale_variance=True,
          final_multiscale_round=False, init='mds', max_iter=30000,
          factr=1e7, pgtol=1e-05, alpha_factr=1e12,
          bcc_lambda=0., hsc_lambda=0., bcc_version='2019', hsc_version='2019',
          counts_interchrom=None, est_hmlg_sep=None, hsc_min_beads=5,
          hsc_perc_diff=None, excluded_counts=None,
          struct_draft_fullres=None, draft=False, simple_diploid=False,
          callback_freq=None, callback_fxns=None, reorienter=None,
          multiscale_reform=False, epsilon_min=1e-6, epsilon_max=1e6,
          epsilon_coord_descent=False, init_std_dev=None,
          alpha_true=None, struct_true=None, input_weight=None,
          exclude_zeros=False, null=False, mixture_coefs=None, verbose=True,
          mods=[]):
    """Infer 3D structures with PASTIS via Poisson model.

    Optimize 3D structure from Hi-C contact counts data for diploid
    organisms. Optionally perform multiscale optimization during inference.

    Parameters
    ----------
    counts_raw : list of array or coo_matrix
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
        optimization.
    final_multiscale_round : bool, optional
        Whether this is the final (full-resolution) round of multiscale
        optimization.
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
    counts_interchrom : int or float, optional
        TODO add bcc_version, hsc_version, & counts_interchrom
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
        Keys: 'alpha', 'beta', 'est_hmlg_sep', 'obj', and 'seed'.
    """

    if outdir is not None:
        outfiles = _get_output_files(outdir, seed=seed)
        if os.path.exists(outfiles['struct_infer']) or os.path.exists(
                outfiles['struct_nonconv']):
            if verbose:
                if os.path.exists(outfiles['struct_infer']):
                    print('CONVERGED\n', flush=True)
                elif os.path.exists(outfiles['struct_nonconv']):
                    print('OPTIMIZATION DID NOT CONVERGE\n', flush=True)
            infer_param = _load_infer_param(outfiles['infer_param'])
            struct_ = np.loadtxt(outfiles['struct_infer'])
            return struct_, infer_param
    else:
        outfiles = None

    # PREP FOR INFERENCE
    prepped = _prep_inference(
        counts_raw, lengths=lengths, ploidy=ploidy, outdir=outdir, alpha=alpha,
        seed=seed, normalize=normalize, filter_threshold=filter_threshold,
        alpha_init=alpha_init, max_alpha_loop=max_alpha_loop, beta=beta,
        multiscale_factor=multiscale_factor,
        multiscale_rounds=multiscale_rounds,
        use_multiscale_variance=use_multiscale_variance,
        final_multiscale_round=final_multiscale_round, init=init,
        max_iter=max_iter, factr=factr, pgtol=pgtol, alpha_factr=alpha_factr,
        bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda, bcc_version=bcc_version,
        hsc_version=hsc_version, counts_interchrom=counts_interchrom,
        est_hmlg_sep=est_hmlg_sep, hsc_min_beads=hsc_min_beads,
        hsc_perc_diff=hsc_perc_diff, excluded_counts=excluded_counts,
        struct_draft_fullres=struct_draft_fullres, draft=draft,
        simple_diploid=simple_diploid, callback_freq=callback_freq,
        callback_fxns=callback_fxns, reorienter=reorienter,
        multiscale_reform=multiscale_reform, init_std_dev=init_std_dev,
        alpha_true=alpha_true, struct_true=struct_true, input_weight=input_weight,
        exclude_zeros=exclude_zeros, null=null, mixture_coefs=mixture_coefs,
        outfiles=outfiles, verbose=verbose, mods=mods)
    (counts, alpha_, beta_, bias, struct_nan, struct_init, constraints, est_hmlg_sep,
        callback, struct_draft_fullres, multiscale_variances, epsilon) = prepped

    if multiscale_rounds <= 1 or multiscale_factor > 1 or final_multiscale_round:
        return _infer_struct(
            counts=counts, lengths=lengths, ploidy=ploidy, alpha=alpha,
            struct_init=struct_init, struct_nan=struct_nan, bias=bias,
            constraints=constraints, callback=callback,
            multiscale_factor=multiscale_factor,
            multiscale_variances=multiscale_variances, epsilon=epsilon,
            epsilon_min=epsilon_min, epsilon_max=epsilon_max,
            epsilon_coord_descent=epsilon_coord_descent,
            multiscale_rounds=multiscale_rounds,
            final_multiscale_round=final_multiscale_round, alpha_init=alpha_init,
            max_alpha_loop=max_alpha_loop, max_iter=max_iter, factr=factr,
            pgtol=pgtol, alpha_factr=alpha_factr, reorienter=reorienter,
            null=null, seed=seed, outfiles=outfiles,
            mixture_coefs=mixture_coefs, verbose=verbose, mods=mods)
    else:
        # BEGIN MULTISCALE OPTIMIZATION
        all_multiscale_factors = 2 ** np.flip(
            np.arange(multiscale_rounds), axis=0)
        X_ = init
        prev_std_dev = None

        for i in all_multiscale_factors:
            if verbose:
                _print_code_header(
                    f'MULTISCALE FACTOR {i}', max_length=60, blank_lines=1)
            if i == 1:
                multiscale_outdir = outdir
                final_multiscale_round = True
            else:
                multiscale_outdir = os.path.join(outdir, 'multiscale_x%d' % i)
                final_multiscale_round = False
            struct_, infer_param = infer(
                counts_raw=counts_raw, outdir=multiscale_outdir,
                lengths=lengths, ploidy=ploidy, alpha=alpha_, seed=seed,
                normalize=normalize, filter_threshold=filter_threshold,
                alpha_init=alpha_init, max_alpha_loop=max_alpha_loop,
                beta=beta_, multiscale_factor=i,
                multiscale_rounds=multiscale_rounds,
                use_multiscale_variance=use_multiscale_variance,
                final_multiscale_round=final_multiscale_round, init=X_,
                max_iter=max_iter, factr=factr, pgtol=pgtol,
                alpha_factr=alpha_factr, bcc_lambda=bcc_lambda,
                hsc_lambda=hsc_lambda, bcc_version=bcc_version,
                hsc_version=hsc_version, counts_interchrom=counts_interchrom,
                est_hmlg_sep=est_hmlg_sep, hsc_min_beads=hsc_min_beads,
                hsc_perc_diff=hsc_perc_diff,
                struct_draft_fullres=struct_draft_fullres,
                callback_fxns=callback_fxns,
                callback_freq=callback_freq, reorienter=reorienter,
                multiscale_reform=multiscale_reform, epsilon_min=epsilon_min,
                epsilon_max=epsilon_max,
                epsilon_coord_descent=epsilon_coord_descent,
                init_std_dev=prev_std_dev, alpha_true=alpha_true,
                struct_true=struct_true, input_weight=input_weight,
                exclude_zeros=exclude_zeros, null=null,
                mixture_coefs=mixture_coefs, verbose=verbose, mods=mods)
            if not infer_param['converged']:
                return struct_, infer_param
            if reorienter is not None and reorienter.reorient:
                X_ = infer_param['orient']
            else:
                X_ = struct_
            alpha_ = infer_param['alpha']

            use_prev_std_dev = False
            if use_prev_std_dev:
                if multiscale_reform:
                    prev_std_dev = infer_param['epsilon'] / np.sqrt(2)
                elif use_multiscale_variance:
                    #prev_std_dev = np.sqrt(infer_param['multiscale_variances'])
                    pass

            if 'epsilon' in infer_param:
                epsilon_max = infer_param['epsilon']  # FIXME??

            if 'lowres_exit' in mods:
                exit(0)

        return struct_, infer_param


def pastis_poisson(counts, lengths, ploidy, outdir='', chromosomes=None,
                   chrom_subset=None, alpha=None, seed=0, normalize=True,
                   filter_threshold=0.04, alpha_init=-3., max_alpha_loop=20,
                   beta=None, multiscale_rounds=1, use_multiscale_variance=True,
                   max_iter=30000, factr=1e7, pgtol=1e-05,
                   alpha_factr=1e12, bcc_lambda=0., hsc_lambda=0.,
                   bcc_version='2019', hsc_version='2019',
                   counts_interchrom=None, est_hmlg_sep=None, hsc_min_beads=5,
                   hsc_perc_diff=None, struct_draft_fullres=None,
                   callback_fxns=None, print_freq=100, history_freq=100,
                   save_freq=None, piecewise=False, piecewise_step=None,
                   piecewise_chrom=None, piecewise_min_beads=5,
                   piecewise_fix_homo=False, piecewise_opt_orient=True,
                   piecewise_step3_multiscale=False,
                   piecewise_step1_accuracy=1,
                   multiscale_reform=False, epsilon_coord_descent=False, alpha_true=None,
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
    TODO add bcc_version, hsc_version, & counts_interchrom
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

    if mods is None:  # TODO remove
        mods = []
    elif isinstance(mods, str):
        mods = mods.lower().split('.')
    else:
        mods = [x.lower() for x in mods]

    if not isinstance(counts, list):
        counts = [counts]
    if verbose:
        print("\nRANDOM SEED = %03d" % seed)
        if all([isinstance(c, str) for c in counts]):
            print('COUNTS: %s' % counts[0])
            if len(counts) > 1:
                print('\n'.join(['        %s' % c for c in counts[1:]]))
        print('')

    lengths_full = lengths
    chrom_full = chromosomes
    callback_freq = {'print': print_freq, 'history': history_freq,
                     'save': save_freq}

    counts, lengths_subset, chrom_subset, lengths_full, chrom_full, struct_true = load_data(
        counts=counts, lengths_full=lengths_full, ploidy=ploidy,
        chrom_full=chrom_full, chrom_subset=chrom_subset,
        exclude_zeros=exclude_zeros, struct_true=struct_true)

    outdir = _output_subdir(
        outdir=outdir, chrom_full=chrom_full, chrom_subset=chrom_subset,
        null=null)

    if (not piecewise) or len(chrom_subset) == 1:
        struct_, infer_param = infer(
            counts_raw=counts, outdir=outdir, lengths=lengths_subset,
            ploidy=ploidy, alpha=alpha, seed=seed, normalize=normalize,
            filter_threshold=filter_threshold, alpha_init=alpha_init,
            max_alpha_loop=max_alpha_loop, beta=beta,
            multiscale_rounds=multiscale_rounds,
            use_multiscale_variance=use_multiscale_variance, init=init,
            max_iter=max_iter, factr=factr, pgtol=pgtol,
            alpha_factr=alpha_factr, bcc_lambda=bcc_lambda,
            hsc_lambda=hsc_lambda, bcc_version=bcc_version,
            hsc_version=hsc_version, counts_interchrom=counts_interchrom,
            est_hmlg_sep=est_hmlg_sep, hsc_min_beads=hsc_min_beads,
            hsc_perc_diff=hsc_perc_diff,
            struct_draft_fullres=struct_draft_fullres,
            callback_fxns=callback_fxns, callback_freq=callback_freq,
            multiscale_reform=multiscale_reform,
            epsilon_coord_descent=epsilon_coord_descent, alpha_true=alpha_true,
            struct_true=struct_true, input_weight=input_weight,
            exclude_zeros=exclude_zeros, null=null, mixture_coefs=mixture_coefs,
            verbose=verbose, mods=mods)
    else:
        from .piecewise_whole_genome import infer_piecewise

        struct_, infer_param = infer_piecewise(
            counts_raw=counts, outdir=outdir, lengths=lengths_subset,
            ploidy=ploidy, chromosomes=chrom_subset, alpha=alpha, seed=seed,
            normalize=normalize, filter_threshold=filter_threshold,
            alpha_init=alpha_init, max_alpha_loop=max_alpha_loop, beta=beta,
            multiscale_rounds=multiscale_rounds,
            use_multiscale_variance=use_multiscale_variance, max_iter=max_iter,
            factr=factr, pgtol=pgtol, alpha_factr=alpha_factr,
            bcc_lambda=bcc_lambda, hsc_lambda=hsc_lambda,
            bcc_version=bcc_version, hsc_version=hsc_version,
            counts_interchrom=counts_interchrom,
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

    if verbose:
        if infer_param['converged']:
            print("INFERENCE COMPLETE: CONVERGED", flush=True)
        else:
            print("INFERENCE COMPLETE: DID NOT CONVERGE", flush=True)

    return struct_, infer_param
