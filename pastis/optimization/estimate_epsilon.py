import numpy as np
from scipy import optimize
import warnings
from autograd import grad
from .poisson import objective


def objective_epsilon(X, counts, alpha, lengths, structures=None, epsilon=None,
                      bias=None, constraints=None, reorienter=None,
                      multiscale_factor=1, mixture_coefs=None,
                      return_extras=False, obj_type=None):
    """Computes the objective function.

    Computes the negative log likelihood of the poisson model and constraints.

    Parameters
    ----------
    X : array of float
        Structure being inferred.
    counts : list of CountsMatrix subclass instances
        Preprocessed counts data.
    alpha : float, optional
        Biophysical parameter of the transfer function used in converting
        counts to wish distances.
    lengths : array of int
        Number of beads per homolog of each chromosome.
    bias : array of float, optional
        Biases computed by ICE normalization.
    constraints : Constraints instance, optional
        Object to compute constraints at each iteration.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.

    Returns
    -------
    obj : float
        The total negative log likelihood of the poisson model and constraints.
    """

    if structures is None and epsilon is None:
        raise ValueError("structures and epsilon may not both be None")
    elif structures is not None and epsilon is not None:
        raise ValueError("Either structures or epsilon must be None")
    elif epsilon is None:
        # Inferring epsilon, not inferring structure
        my_epsilon = X
        my_struct = structures.flatten()
    else:
        # Inferring structure, not inferring epsilon
        my_epsilon = epsilon
        my_struct = X

    return objective(
        my_struct, counts, alpha=alpha, lengths=lengths, bias=bias,
        constraints=constraints, reorienter=reorienter,
        multiscale_factor=multiscale_factor, multiscale_variances=None,
        multiscale_reform=True, mixture_coefs=mixture_coefs,
        return_extras=return_extras, epsilon=my_epsilon, obj_type=obj_type)


def objective_wrapper_epsilon(X, counts, alpha, lengths, structures=None, epsilon=None,
                              bias=None, constraints=None,
                              reorienter=None, multiscale_factor=1,
                              mixture_coefs=None, callback=None, obj_type=None):
    """Objective function wrapper to match scipy.optimize's interface.
    """

    if structures is None and epsilon is None:
        raise ValueError("structures and epsilon may not both be None")
    if structures is not None and epsilon is not None:
        raise ValueError("Either structures or epsilon must be None")

    new_obj, obj_logs, structures, alpha = objective_epsilon(
        X, counts=counts, alpha=alpha, lengths=lengths, bias=bias,
        structures=structures, epsilon=epsilon,
        constraints=constraints, reorienter=reorienter,
        multiscale_factor=multiscale_factor,
        mixture_coefs=mixture_coefs, return_extras=True, obj_type=obj_type)

    if callback is not None:
        if epsilon is None:
            callback.on_epoch_end(obj_logs, structures, alpha, X, epsilon=X[0])
        else:
            callback.on_epoch_end(obj_logs, structures, alpha, X, epsilon=epsilon)

    if epsilon is None:
        spacer = ' ' * (12 - len(f"{new_obj:.3g}"))
        if X[0] == 0:
            print(f'    obj {new_obj:.3g}{spacer}ε ---', flush=True)
        else:
            print(f'    obj {new_obj:.3g}{spacer}ε {X[0]:.6g}', flush=True)

    return new_obj


gradient_epsilon = grad(objective_epsilon)


def fprime_wrapper_epsilon(X, counts, alpha, lengths, structures=None, epsilon=None,
                           bias=None, constraints=None,
                           reorienter=None, multiscale_factor=1,
                           mixture_coefs=None, callback=None, obj_type=None):
    """Gradient function wrapper to match scipy.optimize's interface.
    """

    if structures is None and epsilon is None:
        raise ValueError("structures and epsilon may not both be None")
    elif structures is not None and epsilon is not None:
        raise ValueError("Either structures or epsilon must be None")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Using a non-tuple sequence for multidimensional"
            " indexing is deprecated", category=FutureWarning)
        new_grad = np.array(gradient_epsilon(
            X, counts=counts, alpha=alpha, lengths=lengths, bias=bias,
            structures=structures, epsilon=epsilon,
            constraints=constraints, reorienter=reorienter,
            multiscale_factor=multiscale_factor,
            mixture_coefs=mixture_coefs, obj_type=obj_type)).flatten()
    if epsilon is None and new_grad[-1] == 0:
        print(f"* * * * EPSILON GRADIENT IS 0 * * * *")

    return new_grad


def estimate_epsilon(counts, init_X, alpha, lengths, bias=None,
                     constraints=None, epsilon=None, structures=None,
                     multiscale_factor=1,
                     epsilon_bounds=None, max_iter=30000, max_fun=None,
                     factr=10000000., pgtol=1e-05, callback=None,
                     alpha_loop=None, epsilon_loop=None,
                     reorienter=None, mixture_coefs=None, verbose=True,
                     obj_type=None):
    """Estimates a 3D structure, given current alpha.

    Infer 3D structure from Hi-C contact counts data for haploid or diploid
    organisms at a given resolution.

    Parameters
    ----------
    counts : list of CountsMatrix subclass instances
        Preprocessed counts data.
    init_X : array_like of float
        Initialization for inference.
    alpha : float, optional
        Biophysical parameter of the transfer function used in converting
        counts to wish distances. If alpha is not specified, it will be
        inferred.
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    bias : array_like of float, optional
        Biases computed by ICE normalization.
    constraints : Constraints instance, optional
        Object to compute constraints at each iteration.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
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
    X : array_like of float
        Output of the optimization (typically a 3D structure).
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
    lengths = np.array(lengths)
    lengths_counts = lengths
    if bias is None:
        bias = np.ones((lengths_counts.sum(),))
    bias = np.array(bias)

    if structures is None and epsilon is None:
        raise ValueError("structures and epsilon may not both be None")
    elif structures is not None and epsilon is not None:
        raise ValueError("Either structures or epsilon must be None")
    elif epsilon is None:
        inferring_epsilon = True
        if isinstance(init_X, float) or isinstance(init_X, int):
            init_X = np.array([init_X])
    else:
        inferring_epsilon = False
        init_X = init_X.flatten()

    if verbose:
        #print('=' * 30, flush=True)
        print("\nRUNNING THE L-BFGS-B CODE\n\n           * * *\n\nMachine"
              " precision = %.4g\n" % np.finfo(np.float).eps, flush=True)

    if callback is not None:
        if inferring_epsilon:
            opt_type = 'epsilon'
        else:
            opt_type = 'structure'
        if reorienter is not None and reorienter.reorient:
            opt_type += '.chrom_reorient'
        callback.on_training_begin(
            opt_type=opt_type, alpha_loop=alpha_loop, epsilon_loop=epsilon_loop)
        obj = objective_wrapper_epsilon(
            init_X, counts=counts, alpha=alpha, lengths=lengths,
            structures=structures, epsilon=epsilon,
            bias=bias, constraints=constraints, reorienter=reorienter,
            multiscale_factor=multiscale_factor,
            mixture_coefs=mixture_coefs, callback=callback, obj_type=obj_type)
    else:
        obj = np.nan

    if inferring_epsilon:
        bounds = np.array(epsilon_bounds).reshape(1, -1)
    else:
        bounds = None

    if max_iter == 0:
        X = init_X
        converged = True
        conv_desc = ''
    else:
        if max_fun is None:
            max_fun = max_iter
        results = optimize.fmin_l_bfgs_b(
            objective_wrapper_epsilon,
            x0=init_X,
            fprime=fprime_wrapper_epsilon,
            iprint=0,
            maxiter=max_iter,
            maxfun=max_fun,
            pgtol=pgtol,
            factr=factr,
            bounds=bounds,
            args=(counts, alpha, lengths, structures, epsilon, bias, constraints,
                  reorienter, multiscale_factor, mixture_coefs, callback, obj_type))
        X, obj, d = results
        converged = d['warnflag'] == 0
        # TODO add conv_desc to main branch
        conv_desc = d['task'].decode('utf8')

    history = None
    if callback is not None:
        callback.on_training_end()
        history = callback.history

    if verbose:
        if inferring_epsilon:
            print(f'INIT EPSILON: {init_X[0]:.3g},  FINAL EPSILON: {X[0]:.3g}',
                  flush=True)
        if converged:
            print('CONVERGED\n', flush=True)
        else:
            print('OPTIMIZATION DID NOT CONVERGE', flush=True)
            print(conv_desc + '\n', flush=True)

    return X, obj, converged, history, conv_desc
