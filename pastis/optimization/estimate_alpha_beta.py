import numpy as np
from scipy import optimize
import warnings
from sklearn.utils import check_random_state

from .utils_poisson import _setup_jax
_setup_jax()
import jax.numpy as ag_np
from jax import grad


from .poisson import _format_X, objective, get_gamma_moments
from .counts import _update_betas_in_counts_matrices
from .multiscale_optimization import decrease_lengths_res


def _estimate_beta_single(structures, counts, alpha, lengths, ploidy, bias=None,
                          multiscale_factor=1, multiscale_variances=None,
                          epsilon=None, mixture_coefs=None):
    """Facilitates estimation of beta for a single counts object.

    Computes the sum of lambda_ij corresponding to a given counts matrix.
    """

    if isinstance(alpha, np.ndarray):
        if len(alpha) > 1:
            raise ValueError("Alpha should be of length 1.")
        alpha = alpha[0]

    if mixture_coefs is not None and len(structures) != len(mixture_coefs):
        raise ValueError("The number of structures (%d) and of mixture"
                         " coefficents (%d) should be identical." %
                         (len(structures), len(mixture_coefs)))
    elif mixture_coefs is None:
        mixture_coefs = [1.]

    if multiscale_variances is not None:
        if isinstance(multiscale_variances, np.ndarray):
            var_per_dis = multiscale_variances[
                counts.row3d] + multiscale_variances[counts.col3d]
        else:
            var_per_dis = multiscale_variances * 2
    else:
        var_per_dis = 0
    num_highres_per_lowres_bins = counts.fullres_per_lowres_dis()

    lambda_intensity_sum = 0.
    for struct, gamma in zip(structures, mixture_coefs):
        dis = ag_np.sqrt((ag_np.square(
            struct[counts.row3d] - struct[counts.col3d])).sum(axis=1))
        if epsilon is None:
            if multiscale_variances is None:
                tmp1 = ag_np.power(dis, alpha)
            else:
                tmp1 = ag_np.power(ag_np.square(dis) + var_per_dis, alpha / 2)
            tmp = tmp1.reshape(-1, counts.nnz).sum(axis=0)
            lambda_intensity_sum += ag_np.sum(gamma * counts.bias_per_bin(
                bias) * num_highres_per_lowres_bins * tmp)
        else:
            tmp = get_gamma_moments(
                dis, epsilon=epsilon, alpha=alpha, beta=1,
                ambiguity=counts.ambiguity, inferring_alpha=True,
                return_var=False)
            if bias is not None and not np.all(bias == 1):
                raise NotImplementedError
            lambda_intensity_sum += ag_np.sum(gamma * counts.bias_per_bin(
                bias) * num_highres_per_lowres_bins * tmp)

    return lambda_intensity_sum


def _estimate_beta(X, counts, alpha, lengths, ploidy, bias=None,
                   reorienter=None, multiscale_factor=1,
                   multiscale_variances=None, multiscale_reform=False,
                   mixture_coefs=None, verbose=False):
    """Estimates beta for all counts matrices.
    """

    if isinstance(alpha, (np.ndarray, ag_np.ndarray)):
        if alpha.size > 1:
            raise ValueError("Alpha should be of length 1.")
        alpha = alpha[0]

    structures, epsilon, mixture_coefs = _format_X(
        X, lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor,
        multiscale_reform=multiscale_reform, mixture_coefs=mixture_coefs)

    if reorienter is not None and reorienter.reorient:
        structures = reorienter.translate_and_rotate(X)

    # Check format of input
    counts = (counts if isinstance(counts, list) else [counts])
    if lengths is None:
        lengths = np.array([min([min(c.shape) for c in counts])])
    lengths = np.array(lengths)
    if bias is None:
        if multiscale_reform:
            bias = np.ones((lengths.sum(),))
        else:
            lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
            bias = np.ones((lengths_lowres.sum(),))
    if not isinstance(structures, list):
        structures = [structures]
    if mixture_coefs is None:
        mixture_coefs = [1.] * len(structures)

    # Estimate beta for each type of counts (ambig, pa, ua)
    counts_sum = {c.ambiguity: c.sum() for c in counts if c.sum() > 0}
    lambda_sum = {c.ambiguity: 0. for c in counts}

    for counts_maps in counts:
        lambda_sum[counts_maps.ambiguity] += _estimate_beta_single(
            structures, counts_maps, alpha=alpha, lengths=lengths,
            ploidy=ploidy, bias=bias, multiscale_factor=multiscale_factor,
            multiscale_variances=multiscale_variances,
            epsilon=epsilon, mixture_coefs=mixture_coefs)

    beta = {x: counts_sum[x] / lambda_sum[x] for x in counts_sum.keys()}
    for ambiguity, beta_maps in beta.items():
        if not ag_np.isfinite(beta_maps) or beta_maps == 0:
            raise ValueError(f"Beta for {ambiguity} counts is {beta_maps}.")

    if verbose:
        print('INFERRED BETA: %s' % ', '.join(['%s=%.3g' %
              (k, v) for k, v in beta.items()]),
              flush=True)

    return beta


def objective_alpha(alpha, counts, X, lengths, ploidy, bias=None,
                    constraints=None, reorienter=None, multiscale_factor=1,
                    multiscale_variances=None, multiscale_reform=False,
                    mixture_coefs=None, return_extras=False, mods=[]):
    """Computes the objective function.

    Computes the negative log likelihood of the poisson model and constraints.

    Parameters
    ----------
    alpha : float, optional
        Biophysical parameter of the transfer function used in converting
        counts to wish distances.
    counts : list of CountsMatrix subclass instances
        Preprocessed counts data.
    X : array of float
        Structure being inferred.
    lengths : array of int
        Number of beads per homolog of each chromosome.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    bias : array of float, optional
        Biases computed by ICE normalization.
    constraints : list of Constraint instances, optional
        Objects to compute constraints at each iteration.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
    multiscale_variances : float or array of float, optional
        For multiscale optimization at low resolution, the variances of each
        group of full-resolution beads corresponding to a single low-resolution
        bead.

    Returns
    -------
    obj : float
        The total negative log likelihood of the poisson model and constraints.
    """

    if isinstance(alpha, (np.ndarray, ag_np.ndarray)):
        if alpha.size > 1:
            raise ValueError("Alpha should be of length 1.")
        alpha = alpha[0]

    return objective(
        X, counts, alpha=alpha, lengths=lengths, ploidy=ploidy, bias=bias,
        constraints=constraints, reorienter=reorienter,
        multiscale_factor=multiscale_factor,
        multiscale_variances=multiscale_variances,
        multiscale_reform=multiscale_reform, mixture_coefs=mixture_coefs,
        return_extras=return_extras, inferring_alpha=True, mods=mods)


gradient_alpha = grad(objective_alpha)


def objective_wrapper_alpha(alpha, counts, X, lengths, ploidy, bias=None,
                            constraints=None, reorienter=None,
                            multiscale_factor=1, multiscale_variances=None,
                            multiscale_reform=False, mixture_coefs=None,
                            callback=None, mods=[]):
    """Objective function wrapper to match scipy.optimize's interface.
    """

    new_beta = _estimate_beta(
        X, counts, alpha=alpha, lengths=lengths, ploidy=ploidy, bias=bias,
        multiscale_factor=multiscale_factor,
        multiscale_variances=multiscale_variances,
        multiscale_reform=multiscale_reform, reorienter=reorienter,
        mixture_coefs=mixture_coefs)
    counts = _update_betas_in_counts_matrices(counts=counts, beta=new_beta)

    new_obj, obj_logs, structures, alpha, epsilon = objective_alpha(
        alpha, counts=counts, X=X, lengths=lengths, ploidy=ploidy, bias=bias,
        constraints=constraints, reorienter=reorienter,
        multiscale_factor=multiscale_factor,
        multiscale_variances=multiscale_variances,
        multiscale_reform=multiscale_reform,
        mixture_coefs=mixture_coefs, return_extras=True, mods=mods)

    if callback is not None:
        callback.on_iter_end(
            obj_logs=obj_logs, structures=structures, alpha=alpha, Xi=X,
            epsilon=epsilon)

    return new_obj


def fprime_wrapper_alpha(alpha, counts, X, lengths, ploidy, bias=None,
                         constraints=None, reorienter=None, multiscale_factor=1,
                         multiscale_variances=None, multiscale_reform=False,
                         mixture_coefs=None, callback=None, mods=[]):
    """Gradient function wrapper to match scipy.optimize's interface.
    """

    new_beta = _estimate_beta(
        X, counts, alpha=alpha, lengths=lengths, ploidy=ploidy, bias=bias,
        multiscale_factor=multiscale_factor,
        multiscale_variances=multiscale_variances,
        multiscale_reform=multiscale_reform, reorienter=reorienter,
        mixture_coefs=mixture_coefs)
    counts = _update_betas_in_counts_matrices(counts=counts, beta=new_beta)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message='Using a non-tuple sequence for multidimensional'
                              ' indexing is deprecated', category=FutureWarning)
        new_grad = np.array(gradient_alpha(
            alpha, counts=counts, X=X, lengths=lengths, ploidy=ploidy,
            bias=bias, constraints=constraints, reorienter=reorienter,
            multiscale_factor=multiscale_factor,
            multiscale_variances=multiscale_variances,
            multiscale_reform=multiscale_reform,
            mixture_coefs=mixture_coefs, mods=mods)).flatten()

    return np.asarray(new_grad, dtype=np.float64)


def estimate_alpha(counts, X, alpha_init, lengths, ploidy, bias=None,
                   constraints=None, multiscale_factor=1,
                   multiscale_variances=None, epsilon=None,
                   random_state=None, max_iter=30000, max_fun=None,
                   factr=1e7, pgtol=1e-05, callback=None, alpha_loop=None,
                   reorienter=None, mixture_coefs=None, verbose=True, mods=[]):
    """Estimates alpha, given current structure.

    Parameters
    ----------
    Given a chromatin structure, infer alpha from Hi-C contact counts data for
    haploid or diploid organisms at a given resolution.

    Parameters
    ----------
    counts : list of CountsMatrix subclass instances
        Preprocessed counts data.
    X : array_like of float
        3D chromatin structure.
    alpha_init : float
        Initialization of alpha, the biophysical parameter of the transfer
        function used in converting counts to wish distances.
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    bias : array_like of float, optional
        Biases computed by ICE normalization.
    constraints : list of Constraint instances, optional
        Objects to compute constraints at each iteration.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
    multiscale_variances : float or array_like of float, optional
        For multiscale optimization at low resolution, the variances of each
        group of full-resolution beads corresponding to a single low-resolution
        bead.
    max_iter : int, optional
        Maximum number of iterations per optimization.
    max_fun : int, optional
        Maximum number of function evaluations per optimization. If not
        supplied, defaults to same value as `max_iter`.
    factr : float, optional
        factr for scipy's L-BFGS-B, alters convergence criteria.
    pgtol : float, optional
        pgtol for scipy's L-BFGS-B, alters convergence criteria.
    callback : pastis.callbacks.Callback object, optional
        Object to perform callback at each iteration and before and after
        optimization.
    alpha_loop : int, optional
        Current iteration of alpha/structure optimization.

    Returns
    -------
    alpha : float
        Output of the optimization, the biophysical parameter of the transfer
        function used in converting counts to wish distances.
    obj : float
        Final objective value.
    converged : bool
        Whether the optimization successfully converged.
    callback.history : list of dict
        History generated by the callback, containing information about the
        objective function during optimization.
    """

    multiscale_reform = (epsilon is not None)
    if multiscale_factor > 1 and multiscale_reform:
        X = np.append(X.flatten(), epsilon)

    # Check format of input
    counts = (counts if isinstance(counts, list) else [counts])
    if 'no0alpha' in mods:  # TODO
        counts = [c for c in counts if c.sum() != 0]
    lengths = np.array(lengths)
    if bias is None:
        if multiscale_reform:
            bias = np.ones((lengths.sum(),))
        else:
            lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
            bias = np.ones((lengths_lowres.sum(),))

    # Initialize alpha if necessary
    if random_state is None:
        random_state = np.random.RandomState(seed=0)
    random_state = check_random_state(random_state)
    if alpha_init is None:
        alpha_init = random_state.uniform(low=-4, high=-1)

    if verbose:
        #print('=' * 30, flush=True)
        print('\nRUNNING THE L-BFGS-B CODE\n\n           * * *\n\n Machine'
              ' precision = %.4g\n' % np.finfo(np.float).eps, flush=True)

    if callback is not None:
        if reorienter is not None and reorienter.reorient:
            opt_type = 'alpha.chrom_reorient'
        else:
            opt_type = 'alpha'
        callback.on_training_begin(opt_type=opt_type, alpha_loop=alpha_loop)
        objective_wrapper_alpha(
            alpha=alpha_init, counts=counts, X=X.flatten(), lengths=lengths,
            ploidy=ploidy, bias=bias, constraints=constraints,
            reorienter=reorienter, multiscale_factor=multiscale_factor,
            multiscale_variances=multiscale_variances,
            multiscale_reform=multiscale_reform, mixture_coefs=mixture_coefs,
            callback=callback, mods=mods)

    if max_fun is None:
        max_fun = max_iter
    results = optimize.fmin_l_bfgs_b(
        objective_wrapper_alpha,
        x0=np.float64(alpha_init),
        fprime=fprime_wrapper_alpha,
        iprint=0,
        maxiter=max_iter,
        maxfun=max_fun,
        pgtol=pgtol,
        factr=factr,
        bounds=np.array([[-4.5, -0.5]]),
        args=(counts, X.flatten(), lengths, ploidy, bias, constraints,
              reorienter, multiscale_factor, multiscale_variances,
              multiscale_reform, mixture_coefs, callback, mods))

    history = None
    if callback is not None:
        callback.on_training_end()
        history = callback.history

    alpha, obj, d = results
    converged = d['warnflag'] == 0
    # TODO add conv_desc to main branch
    conv_desc = d['task']
    if isinstance(conv_desc, bytes):
        conv_desc = conv_desc.decode('utf8')

    if verbose:
        print(f'INIT ALPHA: {alpha_init:.3g},  FINAL ALPHA: {float(alpha):.3g}',
              flush=True)
        if converged:
            print('CONVERGED\n', flush=True)
        else:
            print('OPTIMIZATION DID NOT CONVERGE', flush=True)
            print(conv_desc + '\n', flush=True)

    return float(alpha), obj, converged, history, conv_desc
