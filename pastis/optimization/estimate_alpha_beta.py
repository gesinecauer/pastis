import numpy as np
from scipy import optimize
import warnings

from .utils_poisson import _setup_jax
_setup_jax()
import jax.numpy as ag_np
from jax import grad

from .poisson import _format_X, objective
from .utils_poisson import _euclidean_distance


def _estimate_beta_single(structures, counts, alpha, lengths, ploidy, bias=None,
                          mixture_coefs=None):
    """Facilitates estimation of beta for a single counts object.

    Computes the sum of lambda_ij corresponding to a given counts matrix.
    """

    if isinstance(alpha, np.ndarray):
        if len(alpha) > 1:
            raise ValueError("Alpha should be of length 1.")
        alpha = alpha[0]

    if mixture_coefs is not None and len(structures) != len(mixture_coefs):
        raise ValueError(
            f"The number of structures ({len(structures)}) and of mixture"
            f" coefficents ({len(mixture_coefs)}) should be identical.")
    elif mixture_coefs is None:
        mixture_coefs = [1.]

    lambda_pois_sum = 0.
    for struct, mix_coef in zip(structures, mixture_coefs):
        dis = _euclidean_distance(struct, row=counts.row3d, col=counts.col3d)
        tmp1 = ag_np.power(dis, alpha)
        tmp = tmp1.reshape(-1, counts.nbins).sum(axis=0)
        lambda_pois_sum += ag_np.sum(mix_coef * counts.bias_per_bin(
            bias) * tmp)

    return lambda_pois_sum


def _estimate_beta(X, counts, alpha, lengths, ploidy, bias=None,
                   reorienter=None, mixture_coefs=None, verbose=False):
    """Estimates beta for all counts matrices.
    """

    if isinstance(alpha, (np.ndarray, ag_np.ndarray)):
        if alpha.size > 1:
            raise ValueError("Alpha should be of length 1.")
        alpha = alpha[0]

    structures, _, mixture_coefs = _format_X(
        X, lengths=lengths, ploidy=ploidy, multiscale_factor=1,
        mixture_coefs=mixture_coefs)

    if reorienter is not None and reorienter.reorient:
        structures = reorienter.translate_and_rotate(X)

    # Check format of input
    counts = (counts if isinstance(counts, list) else [counts])
    if lengths is None:
        lengths = np.array([min([min(c.shape) for c in counts])])
    lengths = np.array(lengths, copy=False, ndmin=1, dtype=int).ravel()
    if not isinstance(structures, list):
        structures = [structures]
    if mixture_coefs is None:
        mixture_coefs = [1.] * len(structures)

    # Estimate beta for each type of counts (ambig, pa, ua)
    counts_sum = {c.ambiguity: c.sum() for c in counts}
    lambda_sum = {c.ambiguity: 0. for c in counts}

    for i in range(len(counts)):
        for counts_bins in counts[i].bins:
            lambda_sum[counts[i].ambiguity] += _estimate_beta_single(
                structures, counts_bins, alpha=alpha, lengths=lengths,
                ploidy=ploidy, bias=bias, mixture_coefs=mixture_coefs)

    beta = {x: counts_sum[x] / lambda_sum[x] for x in counts_sum.keys()}
    for ambiguity, beta_maps in beta.items():
        if not ag_np.isfinite(beta_maps) or beta_maps == 0:
            raise ValueError(f"Beta for {ambiguity} counts is {beta_maps}.")

    if verbose:
        print('INFERRED BETA: ' + ', '.join(
              [f'{k}={v:.3g}' for k, v in beta.items()]), flush=True)

    return beta


def objective_alpha(alpha, counts, X, lengths, ploidy, bias=None,
                    constraints=None, reorienter=None, mixture_coefs=None,
                    return_extras=False, mods=[]):
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
        mixture_coefs=mixture_coefs, return_extras=return_extras,
        inferring_alpha=True, mods=mods)


gradient_alpha = grad(objective_alpha)


def objective_wrapper_alpha(alpha, counts, X, lengths, ploidy, bias=None,
                            constraints=None, reorienter=None,
                            mixture_coefs=None, callback=None, mods=[]):
    """Objective function wrapper to match scipy.optimize's interface.
    """

    beta_new = _estimate_beta(
        X, counts, alpha=alpha, lengths=lengths, ploidy=ploidy, bias=bias,
        reorienter=reorienter, mixture_coefs=mixture_coefs)
    counts = [c.update_beta(beta_new) for c in counts]

    new_obj, obj_logs, structures, alpha, _ = objective_alpha(
        alpha, counts=counts, X=X, lengths=lengths, ploidy=ploidy, bias=bias,
        constraints=constraints, reorienter=reorienter,
        mixture_coefs=mixture_coefs, return_extras=True, mods=mods)

    if callback is not None:
        callback.on_iter_end(
            obj_logs=obj_logs, structures=structures, alpha=alpha, Xi=X)

    return new_obj


def fprime_wrapper_alpha(alpha, counts, X, lengths, ploidy, bias=None,
                         constraints=None, reorienter=None,
                         mixture_coefs=None, callback=None, mods=[]):
    """Gradient function wrapper to match scipy.optimize's interface.
    """

    beta_new = _estimate_beta(
        X, counts, alpha=alpha, lengths=lengths, ploidy=ploidy, bias=bias,
        reorienter=reorienter, mixture_coefs=mixture_coefs)
    counts = [c.update_beta(beta_new) for c in counts]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message='Using a non-tuple sequence for multidimensional'
                              ' indexing is deprecated', category=FutureWarning)
        new_grad = np.array(gradient_alpha(
            alpha, counts=counts, X=X, lengths=lengths, ploidy=ploidy,
            bias=bias, constraints=constraints, reorienter=reorienter,
            mixture_coefs=mixture_coefs, mods=mods)).flatten()

    return np.asarray(new_grad, dtype=np.float64)


def estimate_alpha(counts, X, alpha_init, lengths, ploidy, bias=None,
                   constraints=None,
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

    # Check format of input
    counts = (counts if isinstance(counts, list) else [counts])
    lengths = np.array(lengths, copy=False, ndmin=1, dtype=int).ravel()

    # Initialize alpha if necessary
    if random_state is None:
        random_state = np.random.RandomState(seed=0)
    if alpha_init is None:
        alpha_init = random_state.uniform(low=-4, high=-1)

    if verbose:
        print("\nRUNNING THE L-BFGS-B CODE\n\n           * * *\n\nMachine"
              f" precision = {np.finfo(float).eps:.4g}\n", flush=True)

    beta_init = {c.ambiguity: c.beta for c in counts}

    if callback is not None:
        if reorienter is not None and reorienter.reorient:
            opt_type = 'alpha.chrom_reorient'
        else:
            opt_type = 'alpha'
        callback.on_training_begin(opt_type=opt_type, alpha_loop=alpha_loop)
        objective_wrapper_alpha(
            alpha=alpha_init, counts=counts, X=X.flatten(), lengths=lengths,
            ploidy=ploidy, bias=bias, constraints=constraints,
            reorienter=reorienter, mixture_coefs=mixture_coefs,
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
              reorienter, mixture_coefs, callback, mods))

    history = None
    if callback is not None:
        callback.on_training_end()
        history = callback.history

    alpha, obj, d = results
    converged = d['warnflag'] == 0
    conv_desc = d['task']
    if isinstance(conv_desc, bytes):
        conv_desc = conv_desc.decode('utf8')

    beta_new = _estimate_beta(
        X, counts, alpha=alpha, lengths=lengths, ploidy=ploidy, bias=bias,
        reorienter=reorienter, mixture_coefs=mixture_coefs)
    counts = [c.update_beta(beta_new) for c in counts]

    if verbose:
        print(f'INITIAL ALPHA: {alpha_init:.3g},  INFERRED ALPHA:'
              f' {float(alpha):.3g}', flush=True)
        print('INITIAL BETA:  ' + ', '.join(
              [f'{k}={v:.3g}' for k, v in beta_init.items()]), flush=True)
        print('INFERRED BETA: ' + ', '.join(
              [f'{k}={v:.3g}' for k, v in beta_new.items()]), flush=True)
        if converged:
            print('CONVERGED\n', flush=True)
        else:
            print('OPTIMIZATION DID NOT CONVERGE', flush=True)
            print(conv_desc + '\n', flush=True)

    return float(alpha), obj, converged, history, conv_desc
