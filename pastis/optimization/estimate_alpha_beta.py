import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import numpy as np
from scipy import optimize
from functools import partial

from .utils_poisson import _setup_jax
_setup_jax()
import jax.numpy as jnp
from jax import grad, jit

from .poisson import _format_X, objective, _check_input
from .utils_poisson import _euclidean_distance


def _estimate_beta_single(structures, counts, alpha, lengths, ploidy, bias=None,
                          mixture_coefs=None):
    """Estimate beta for a single counts matrix."""

    # Check format of input
    if isinstance(alpha, (np.ndarray, jnp.ndarray)):
        if alpha.size > 1:
            raise ValueError("Alpha must be a float or array of size 1.")

    if mixture_coefs is not None and len(structures) != len(mixture_coefs):
        raise ValueError(
            f"The number of structures ({len(structures)}) and of mixture"
            f" coefficents ({len(mixture_coefs)}) should be identical.")
    elif mixture_coefs is None:
        mixture_coefs = [1]

    lambda_pois_sum = 0
    for counts_bins in counts.bins:
        for struct, mix_coef in zip(structures, mixture_coefs):
            dis = _euclidean_distance(
                struct, row=counts_bins.row3d, col=counts_bins.col3d)
            tmp = jnp.power(dis, alpha).reshape(
                -1, counts_bins.nbins).sum(axis=0)
            if bias is not None:
                tmp = tmp * counts_bins.bias_per_bin(bias)
            lambda_pois_sum = lambda_pois_sum + jnp.sum(mix_coef * tmp)

    beta = counts.sum() / lambda_pois_sum

    if type(beta).__name__ in ('DeviceArray', 'ndarray') and (
            (not jnp.isfinite(beta)) or beta <= 0):
        raise ValueError(f"Beta for {counts.ambiguity} counts is {beta}.")

    return beta


def _estimate_beta(X, counts, alpha, lengths, ploidy, bias=None,
                   reorienter=None, mixture_coefs=None):
    """Estimates betas for all counts matrices."""

    # Check format of input
    if isinstance(alpha, (np.ndarray, jnp.ndarray)):
        if alpha.size > 1:
            raise ValueError("Alpha must be a float or array of size 1.")
    if not isinstance(counts, (list, tuple)):
        counts = [counts]
    if not isinstance(lengths, (tuple, np.ndarray, jnp.ndarray)):
        lengths = np.array(lengths, copy=False, ndmin=1, dtype=int).ravel()

    structures, _, mixture_coefs = _format_X(
        X, lengths=lengths, ploidy=ploidy, multiscale_factor=1,
        mixture_coefs=mixture_coefs)

    if reorienter is not None and reorienter.reorient:
        structures = reorienter.translate_and_rotate(X)

    # Estimate beta for each counts matrix
    betas = jnp.zeros(len(counts))
    for i in range(len(counts)):
        beta_i = _estimate_beta_single(
            structures, counts=counts[i], alpha=alpha, lengths=lengths,
            ploidy=ploidy, bias=bias, mixture_coefs=mixture_coefs)
        betas = betas.at[i].set(beta_i)
    return betas


def objective_alpha(alpha, beta, counts, X, lengths, ploidy, bias=None,
                    constraints=None, reorienter=None, mixture_coefs=None, mods=()):
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

    if isinstance(alpha, (np.ndarray, jnp.ndarray)):
        if alpha.size > 1:
            raise ValueError("Alpha must be a float or array of size 1.")

    return objective(
        X, counts, alpha=alpha, lengths=lengths, ploidy=ploidy, beta=beta,
        bias=bias, constraints=constraints, reorienter=reorienter,
        mixture_coefs=mixture_coefs, inferring_alpha=True, mods=mods)


_estimate_beta_jit = jit(_estimate_beta, static_argnames=[
    'counts', 'lengths', 'ploidy', 'reorienter', 'mixture_coefs'])
objective_alpha_jit = jit(objective_alpha, static_argnames=[
    'counts', 'lengths', 'ploidy', 'constraints', 'reorienter', 'mixture_coefs', 'mods'])
gradient_alpha = grad(objective_alpha_jit, has_aux=True)


def objective_wrapper_alpha(alpha, counts, X, lengths, ploidy, bias=None,
                            constraints=None, reorienter=None,
                            mixture_coefs=None, callback=None, mods=()):
    """Objective function wrapper to match scipy.optimize's interface."""

    checked = _check_input(
        lengths=lengths, alpha=alpha, counts=counts, constraints=constraints,
        bias=bias, mixture_coefs=mixture_coefs, mods=mods)
    (lengths, alpha, counts, constraints, bias, mixture_coefs, mods) = checked

    beta_new = _estimate_beta_jit(
        X, counts, alpha=alpha, lengths=lengths, ploidy=ploidy, bias=bias,
        reorienter=reorienter, mixture_coefs=mixture_coefs)

    new_obj, (obj_logs, structures, alpha, _) = objective_alpha_jit(
        alpha, beta=beta_new, counts=counts, X=X, lengths=lengths,
        ploidy=ploidy, bias=bias, constraints=constraints,
        reorienter=reorienter, mixture_coefs=mixture_coefs, mods=mods)

    if callback is not None:
        callback.on_iter_end(
            obj_logs=obj_logs, structures=structures, alpha=alpha, X=X)

    return new_obj


def fprime_wrapper_alpha(alpha, counts, X, lengths, ploidy, bias=None,
                         constraints=None, reorienter=None,
                         mixture_coefs=None, callback=None, mods=()):
    """Gradient function wrapper to match scipy.optimize's interface."""

    checked = _check_input(
        lengths=lengths, alpha=alpha, counts=counts, constraints=constraints,
        bias=bias, mixture_coefs=mixture_coefs, mods=mods)
    (lengths, alpha, counts, constraints, bias, mixture_coefs, mods) = checked

    beta_new = _estimate_beta_jit(
        X, counts, alpha=alpha, lengths=lengths, ploidy=ploidy, bias=bias,
        reorienter=reorienter, mixture_coefs=mixture_coefs)

    new_grad = np.array(gradient_alpha(
        alpha, beta=beta_new, counts=counts, X=X, lengths=lengths,
        ploidy=ploidy, bias=bias, constraints=constraints,
        reorienter=reorienter, mixture_coefs=mixture_coefs, mods=mods)[0]).flatten()

    return np.asarray(new_grad, dtype=np.float64)


def estimate_alpha(counts, X, alpha_init, lengths, ploidy, bias=None,
                   constraints=None,
                   random_state=None, max_iter=30000, max_fun=None,
                   factr=1e7, pgtol=1e-05, callback=None, alpha_loop=None,
                   reorienter=None, mixture_coefs=None, verbose=True, mods=()):
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
    callback.log : list of dict
        Log generated by the callback, containing information about the
        objective function during optimization.
    """

    # Check format of input; convert lists to tuples for jax jit
    checked = _check_input(
        lengths=lengths, alpha=alpha_init, counts=counts,
        constraints=constraints, bias=bias, mixture_coefs=mixture_coefs, mods=mods)
    (lengths, alpha_init, counts, constraints, bias, mixture_coefs, mods) = checked

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
        callback.on_optimization_begin(inferring='alpha', alpha_loop=alpha_loop)
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
        bounds=np.array([[-4, -1]]),
        args=(counts, X.flatten(), lengths, ploidy, bias, constraints,
              reorienter, mixture_coefs, callback, mods))

    log = None
    if callback is not None:
        callback.on_optimization_end()
        log = callback.log

    alpha, obj, d = results
    alpha = float(alpha)
    converged = d['warnflag'] == 0
    conv_desc = d['task']
    if isinstance(conv_desc, bytes):
        conv_desc = conv_desc.decode('utf8')

    beta_new = _estimate_beta(
        X, counts, alpha=alpha, lengths=lengths, ploidy=ploidy,
        bias=bias, reorienter=reorienter, mixture_coefs=mixture_coefs)._value
    counts = [counts[i].update_beta(beta_new[i]) for i in range(len(counts))]

    if verbose:
        print(f'INITIAL ALPHA: {alpha_init:.3g},  INFERRED ALPHA:'
              f' {alpha:.3g}', flush=True)
        print('INITIAL BETA:  ' + ', '.join(
              [f'{k}={v:.3g}' for k, v in beta_init.items()]), flush=True)
        print('INFERRED BETA: ' + ', '.join(
              [f'{c.ambiguity}={c.beta:.3g}' for c in counts]), flush=True)
        if converged:
            print('CONVERGED\n', flush=True)
        else:
            print('OPTIMIZATION DID NOT CONVERGE', flush=True)
            print(conv_desc + '\n', flush=True)

    return alpha, obj, converged, log, conv_desc
