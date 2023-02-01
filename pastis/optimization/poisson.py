import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import numpy as np
from scipy import optimize
import warnings
from timeit import default_timer as timer
from functools import partial

from .utils_poisson import _setup_jax
_setup_jax()
import jax.numpy as jnp
from jax import grad, jit

from .multiscale_optimization import decrease_lengths_res
from .utils_poisson import _euclidean_distance
from .polynomial import _approx_ln_f
from .likelihoods import gamma_poisson_nll, poisson_nll


def get_gamma_moments(struct, epsilon, alpha, beta, row3d, col3d,
                      ambiguity='ua', inferring_alpha=False, mods=()):

    dis = _euclidean_distance(struct, row=row3d, col=col3d)
    dis_alpha = jnp.power(dis, alpha)

    ln_f_mean, ln_f_var = _approx_ln_f(
        dis, epsilon=epsilon, alpha=alpha, inferring_alpha=inferring_alpha,
        mods=mods)

    gamma_mean = jnp.exp(ln_f_mean) * dis_alpha * beta
    gamma_var = jnp.exp(ln_f_var) * jnp.square(dis_alpha * beta)

    if ambiguity != 'ua':
        if ambiguity == 'ambig':
            reshape_0 = 4
        else:
            reshape_0 = 2
        gamma_mean = gamma_mean.reshape(reshape_0, -1).sum(axis=0)
        gamma_var = gamma_var.reshape(reshape_0, -1).sum(axis=0)

    return gamma_mean, gamma_var


def get_gamma_params(struct, epsilon, alpha, beta, row3d, col3d, ambiguity='ua',
                     inferring_alpha=False, mods=()):

    dis = _euclidean_distance(struct, row=row3d, col=col3d)
    dis_alpha = jnp.power(dis, alpha)

    ln_f_mean, ln_f_var = _approx_ln_f(
        dis, epsilon=epsilon, alpha=alpha, inferring_alpha=inferring_alpha,
        mods=mods)

    if ambiguity == 'ua':
        theta_tmp = dis_alpha * jnp.exp(ln_f_var - ln_f_mean)
        k = jnp.exp(2 * ln_f_mean - ln_f_var)
    else:
        gamma_mean = jnp.exp(ln_f_mean) * dis_alpha
        gamma_var = jnp.exp(ln_f_var) * jnp.square(dis_alpha)

        if ambiguity == 'ambig':
            reshape_0 = 4
        else:
            reshape_0 = 2
        gamma_mean = gamma_mean.reshape(reshape_0, -1).sum(axis=0)
        gamma_var = gamma_var.reshape(reshape_0, -1).sum(axis=0)

        theta_tmp = gamma_var / gamma_mean
        k = jnp.square(gamma_mean) / gamma_var

    theta = beta * theta_tmp
    return k, theta


def _multires_negbinom_obj(structures, epsilon, counts, alpha, lengths, ploidy,
                           beta, bias=None, multiscale_factor=1,
                           mixture_coefs=None, mods=()):
    """Computes the multiscale objective function for a given counts matrix.
    """

    obj = 0
    for struct, mix_coef in zip(structures, mixture_coefs):
        k, theta = get_gamma_params(
            struct, epsilon=epsilon, alpha=alpha, beta=beta, row3d=counts.row3d,
            col3d=counts.col3d, ambiguity=counts.ambiguity, mods=mods)
        obj = obj + mix_coef * gamma_poisson_nll(
            theta=theta, k=k, data=counts.data,
            bias_per_bin=counts.bias_per_bin(bias), mask=counts.mask,
            data_per_bin=counts.fullres_per_lowres_dis, mods=mods)

    if type(obj).__name__ in ('DeviceArray', 'ndarray') and not jnp.isfinite(obj):
        raise ValueError(
            "Multires (negative binomial) component of objective function for"
            f" {counts.name} is {obj} at {multiscale_factor}x resolution.")

    return counts.weight * obj


def _poisson_obj(structures, counts, alpha, lengths, ploidy, beta, bias=None,
                 multiscale_factor=1, mixture_coefs=None, mods=()):
    """Computes the Poisson objective function for a given counts matrix.
    """

    lambda_pois = jnp.zeros(counts.nbins)
    for struct, mix_coef in zip(structures, mixture_coefs):
        dis = _euclidean_distance(struct, row=counts.row3d, col=counts.col3d)
        tmp = beta * jnp.power(dis, alpha).reshape(-1, counts.nbins).sum(axis=0)
        if bias is not None:
            tmp = tmp * counts.bias_per_bin(bias)
        lambda_pois = lambda_pois + mix_coef * tmp

    obj = poisson_nll(counts.data, lambda_pois=lambda_pois, mask=counts.mask,
                      data_per_bin=counts.fullres_per_lowres_dis)

    if type(obj).__name__ in ('DeviceArray', 'ndarray') and not jnp.isfinite(obj):
        raise ValueError(
            f"Poisson component of objective function for {counts.name}"
            f" is {obj} at {multiscale_factor}x resolution.")

    return counts.weight * obj


def _obj_single(structures, counts, alpha, lengths, ploidy, beta, bias=None,
                multiscale_factor=1, epsilon=None, mixture_coefs=None, mods=()):
    """Computes the objective function for a given individual counts matrix.
    """

    if counts.nbins == 0 or counts.null:
        return 0
    if (not np.isfinite(counts.weight)) or counts.weight <= 0:
        raise ValueError(f"Counts weight may not be {counts.weight}.")

    if mixture_coefs is not None and len(structures) != len(mixture_coefs):
        raise ValueError(
            f"The number of structures ({len(structures)}) and of mixture"
            f" coefficents ({len(mixture_coefs)}) should be identical.")
    elif mixture_coefs is None:
        mixture_coefs = [1]

    if epsilon is None or multiscale_factor == 1:
        obj = _poisson_obj(
            structures=structures, counts=counts, alpha=alpha, lengths=lengths,
            ploidy=ploidy, beta=beta, bias=bias,
            multiscale_factor=multiscale_factor, mixture_coefs=mixture_coefs,
            mods=mods)
    else:
        obj = _multires_negbinom_obj(
            structures=structures, epsilon=epsilon, counts=counts, alpha=alpha,
            lengths=lengths, ploidy=ploidy, beta=beta, bias=bias,
            multiscale_factor=multiscale_factor, mixture_coefs=mixture_coefs,
            mods=mods)

    return obj


def objective(X, counts, alpha, lengths, ploidy, beta=None, bias=None,
              constraints=None, reorienter=None, multiscale_factor=1,
              multiscale_reform=False, mixture_coefs=None,
              inferring_alpha=False, mods=()):
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
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    bias : array of float, optional
        Biases computed by ICE normalization.
    constraints : list of Constraint instances, optional
        Objects to compute constraints at each iteration.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.

    Returns
    -------
    obj : float
        The total negative log likelihood of the poisson model and constraints.
    """

    # Check format of input
    if not isinstance(counts, (list, tuple)):
        counts = [counts]
    if constraints is not None and not isinstance(constraints, (list, tuple)):
        constraints = [constraints]
    if not isinstance(lengths, (tuple, np.ndarray, jnp.ndarray)):
        lengths = np.array(lengths, copy=False, ndmin=1, dtype=int).ravel()

    # Get beta
    if beta is None:
        if inferring_alpha:
            raise ValueError("Must supply beta when inferring alpha.")
        beta = [c.beta for c in counts]
    else:
        beta = jnp.array(beta, copy=False, ndmin=1).ravel()
        if beta.size != len(counts):
            raise ValueError(
                "Beta needs to contain as many values as there are counts"
                f" matrices ({len(counts)}). It is of size {beta.size}.")
        stored_betas = np.array([c.beta for c in counts])
        if (not inferring_alpha) and (not np.all(stored_betas == None)) and (
                not jnp.array_equal(beta, stored_betas)):
            warnings.warn("Overriding betas stored in counts matrices...")

    # Format X
    if reorienter is None or (not reorienter.reorient):
        X, epsilon, mixture_coefs = _format_X(
            X, lengths=lengths, ploidy=ploidy, multiscale_factor=multiscale_factor,
            multiscale_reform=multiscale_reform, mixture_coefs=mixture_coefs, mods=mods)

    # Optionally translate & rotate structures
    if reorienter is not None and reorienter.reorient:
        reorienter.check_X(X)
        structures = reorienter.translate_and_rotate(X)
    else:
        structures = X

    # Get the constraint terms
    obj_constraints = {}
    if constraints is not None:
        for constraint in constraints:
            constraint_obj = 0
            for struct, mix_coef in zip(structures, mixture_coefs):
                constraint_obj = constraint_obj + mix_coef * constraint.apply(
                    struct=struct, alpha=alpha, epsilon=epsilon,
                    counts=counts, bias=bias, inferring_alpha=inferring_alpha)
            obj_constraints[f"obj_{constraint.abbrev}"] = constraint_obj

    # Get main objective (Poisson/NegBinom of all eligible counts bins)
    obj_main = {}
    obj_main_sum = 0
    for i in range(len(counts)):
        for counts_bins in counts[i].bins:
            obj_counts = _obj_single(
                structures=structures, counts=counts_bins, alpha=alpha,
                lengths=lengths, ploidy=ploidy, beta=beta[i], bias=bias,
                multiscale_factor=multiscale_factor, epsilon=epsilon,
                mixture_coefs=mixture_coefs, mods=mods)
            obj_main[f"obj_{counts_bins.name}"] = obj_counts
            obj_main_sum = obj_main_sum + obj_counts * counts_bins.nbins

    # Take mean of main objective terms, weighted by total number of bins
    obj_main_mean = obj_main_sum / sum([c.nbins for c in counts])

    # Total objective
    obj = obj_main_mean + sum(obj_constraints.values())

    obj_logs = {**obj_main, **obj_constraints,
                **{'obj': obj, 'obj_main': obj_main_mean}}
    return obj, (obj_logs, structures, alpha, epsilon)


def _format_X(X, lengths, ploidy, multiscale_factor=1,
              multiscale_reform=False, mixture_coefs=None, mods=()):
    """Reformat and check X."""

    if mixture_coefs is None:
        mixture_coefs = [1]

    # Get number of beads
    lengths_lowres = decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor)
    nbeads = lengths_lowres.sum() * ploidy

    # Get epsilon
    if multiscale_factor > 1 and multiscale_reform:
        if X.size == nbeads * 3 * len(mixture_coefs) + 1:
            epsilon = X[-1]
            X = X[:-1]
        else:
            raise ValueError(f"Epsilon must be of length 1, {X.shape=}.")
    else:
        epsilon = None

    # Reshape into 3D coordinates
    try:
        X = X.reshape(-1, 3)
    except ValueError:
        raise ValueError("Structures should be composed of 3D bead coordinates,"
                         f" {X.shape=}.")

    # Get list of structures, one per mix_coef
    num_mix = len(mixture_coefs)
    nbeads_ = int(X.shape[0] / num_mix)
    if nbeads_ != X.shape[0] / num_mix:
        raise ValueError("X.shape[0] should be divisible by the length of"
                         f" mixture_coefs, {num_mix}. {X.shape=}.")
    X = [X[i * nbeads_:(i + 1) * nbeads_] for i in range(num_mix)]

    # Check number of beads
    if nbeads_ != nbeads:
        raise ValueError(f"Structures must contain {nbeads}"
                         f" beads. They contain {nbeads_} beads.")

    return X, epsilon, mixture_coefs


objective_jit = jit(objective, static_argnames=[
    'counts', 'alpha', 'lengths', 'ploidy', 'constraints', 'reorienter',
    'multiscale_factor', 'multiscale_reform', 'mixture_coefs', 'mods'])
gradient = grad(objective_jit, has_aux=True)


def objective_wrapper(X, counts, alpha, lengths, ploidy, bias=None,
                      constraints=None, reorienter=None, multiscale_factor=1,
                      multiscale_reform=False, callback=None, mixture_coefs=None, mods=()):
    """Objective function wrapper to match scipy.optimize's interface.
    """

    checked = _check_input(
        lengths=lengths, alpha=alpha, counts=counts, constraints=constraints,
        bias=bias, mixture_coefs=mixture_coefs, mods=mods)
    (lengths, alpha, counts, constraints, bias, mixture_coefs, mods) = checked

    new_obj, (obj_logs, structures, alpha, epsilon) = objective_jit(
        X, counts=counts, alpha=alpha, lengths=lengths, ploidy=ploidy,
        bias=bias, constraints=constraints, reorienter=reorienter,
        multiscale_factor=multiscale_factor, multiscale_reform=multiscale_reform,
        mixture_coefs=mixture_coefs, mods=mods)

    if callback is not None:
        callback.on_iter_end(obj_logs=obj_logs, structures=structures,
                             alpha=alpha, X=X, epsilon=epsilon)

    return new_obj


def fprime_wrapper(X, counts, alpha, lengths, ploidy, bias=None,
                   constraints=None, reorienter=None, multiscale_factor=1,
                   multiscale_reform=False, callback=None, mixture_coefs=None, mods=()):
    """Gradient function wrapper to match scipy.optimize's interface.
    """

    # checked = _check_input(  # TODO remove
    #     lengths=lengths, alpha=alpha, counts=counts, constraints=constraints,
    #     bias=bias, mixture_coefs=mixture_coefs, mods=mods)
    # (lengths, alpha, counts, constraints, bias, mixture_coefs, mods) = checked

    new_grad = np.array(gradient(
        X, counts=counts, alpha=alpha, lengths=lengths, ploidy=ploidy,
        bias=bias, constraints=constraints, reorienter=reorienter,
        multiscale_factor=multiscale_factor, multiscale_reform=multiscale_reform,
        mixture_coefs=mixture_coefs, mods=mods)[0]).flatten()

    return new_grad


def _check_input(lengths, alpha, counts, constraints, bias, mixture_coefs,
                 lengths_as_tuple=True, mods=()):
    """Check format of input; convert lists to tuples for jax JIT compilation"""

    if lengths_as_tuple:
        if not isinstance(lengths, tuple):
            lengths = tuple(
                np.array(lengths, copy=False, ndmin=1, dtype=int).ravel())
    else:
        lengths = np.array(lengths, copy=False, ndmin=1, dtype=int).ravel()

    if alpha is not None and (alpha > -1 or alpha < -4):
        raise ValueError(f"Alpha must be between -4 and -1, {alpha=:.3g}.")

    if isinstance(counts, list):
        counts = tuple(counts)
    elif not isinstance(counts, tuple):
        counts = (counts,)

    if constraints is not None:
        if not isinstance(constraints, (list, tuple)):
            constraints = [constraints]
        if not all([x.setup_completed for x in constraints]):
            # Constraint setup may modify object attributes, so it must be
            # completed before inputting constraints into a jitted function
            if isinstance(constraints, tuple):
                constraints = list(constraints)
            [x.setup(counts=counts, bias=bias) for x in constraints]
        if not isinstance(constraints, tuple):
            constraints = tuple(constraints)

    if bias is not None and np.all(bias == 1):
        bias = None

    if mixture_coefs is not None:
        if isinstance(mixture_coefs, (list, np.ndarray)):
            mixture_coefs = tuple(mixture_coefs)
        elif not isinstance(mixture_coefs, tuple):
            mixture_coefs = (mixture_coefs,)

    if mods is not None:
        if isinstance(mods, (list, np.ndarray)):
            mods = tuple(mods)
        elif not isinstance(mods, tuple):
            mods = (mods,)
    else:
        mods = ()

    return (lengths, alpha, counts, constraints, bias, mixture_coefs, mods)


def estimate_X(counts, init_X, alpha, lengths, ploidy, bias=None,
               constraints=None, multiscale_factor=1, epsilon=None,
               epsilon_bounds=None, max_iter=30000, max_fun=None,
               factr=1e7, pgtol=1e-05, callback=None, alpha_loop=0,
               reorienter=None, mixture_coefs=None, verbose=True, mods=()):
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
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    bias : array_like of float, optional
        Biases computed by ICE normalization.
    constraints : list of Constraint instances, optional
        Objects to compute constraints at each iteration.
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
    callback.log : list of dict
        Log generated by the callback, containing information about the
        objective function during optimization.
    """

    multiscale_reform = (epsilon is not None)

    checked = _check_input(
        lengths=lengths, alpha=alpha, counts=counts, constraints=constraints,
        bias=bias, mixture_coefs=mixture_coefs, mods=mods)
    (lengths, alpha, counts, constraints, bias, mixture_coefs, mods) = checked

    if (multiscale_factor != 1 and multiscale_reform):
        x0 = np.append(init_X.flatten(), epsilon)
    else:
        x0 = init_X.flatten()

    if verbose:
        print("\nRUNNING THE L-BFGS-B CODE\n\n           * * *\n\nMachine"
              f" precision = {np.finfo(float).eps:.4g}\n", flush=True)

    if callback is not None:
        callback.on_optimization_begin(
            inferring='structure', alpha_loop=alpha_loop)
        obj = objective_wrapper(
            x0, counts=counts, alpha=alpha, lengths=lengths, ploidy=ploidy,
            bias=bias, constraints=constraints, reorienter=reorienter,
            multiscale_factor=multiscale_factor, multiscale_reform=multiscale_reform,
            callback=callback, mixture_coefs=mixture_coefs, mods=mods)
    else:
        obj = np.nan

    if multiscale_reform:
        bounds = np.append(
            np.full((init_X.flatten().shape[0], 2), None),
            np.array(epsilon_bounds).reshape(-1, 2), axis=0)
    else:
        bounds = None

    if max_iter == 0:
        X = x0
        converged = True
        conv_desc = ''
    else:
        if max_fun is None:
            max_fun = max_iter
        results = optimize.fmin_l_bfgs_b(
            objective_wrapper, x0=x0, fprime=fprime_wrapper, iprint=0,
            maxiter=max_iter, maxfun=max_fun, pgtol=pgtol, factr=factr,
            bounds=bounds,
            args=(counts, alpha, lengths, ploidy, bias, constraints,
                  reorienter, multiscale_factor, multiscale_reform, callback,
                  mixture_coefs, mods))
        X, obj, d = results
        converged = d['warnflag'] == 0
        conv_desc = d['task']
        if isinstance(conv_desc, bytes):
            conv_desc = conv_desc.decode('utf8')

    log = None
    if callback is not None:
        callback.on_optimization_end()
        log = callback.log

    if verbose:
        if multiscale_reform:
            _, final_epsilon, _ = _format_X(
                X, lengths=lengths, ploidy=ploidy,
                multiscale_factor=multiscale_factor,
                multiscale_reform=multiscale_reform,
                mixture_coefs=mixture_coefs, mods=mods)
            print(f'INITIAL EPSILON: {epsilon:.3g},'
                  f'  INFERRED EPSILON: {final_epsilon:.3g}', flush=True)
        if converged:
            print('CONVERGED\n', flush=True)
        else:
            print('OPTIMIZATION DID NOT CONVERGE', flush=True)
            print(conv_desc + '\n', flush=True)

    return X, obj, converged, log, conv_desc


def _convergence_criteria(f_k, f_kplus1, factr=1e7):
    """Convergence criteria for joint inference of alpha & structure.
    """
    if f_k is None:
        return False
    else:
        return np.abs(f_k - f_kplus1) / max(np.abs(f_k), np.abs(
            f_kplus1), 1) <= factr * np.finfo(float).eps


class PastisPM(object):
    """Infer 3D structures with PASTIS.

    Infer 3D structure from Hi-C contact counts data for haploid or diploid
    organisms at a given resolution. Optionally, jointly infer alpha alongside
    structure.

    Parameters
    ----------
    counts : list of CountsMatrix subclass instances
        Preprocessed counts data.
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    alpha : float
        Biophysical parameter of the transfer function used in converting
        counts to wish distances. If alpha is not specified, it will be
        inferred.
    init : array_like of float
        Initialization for inference.
    bias : array_like of float, optional
        Biases computed by ICE normalization.
    constraints : list of Constraint instances, optional
        Objects to compute constraints at each iteration.
    callback : pastis.callbacks.Callback object, optional
        Object to perform callback at each iteration and before and after
        optimization.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
    alpha_init : float, optional
        For PM2, the initial value of alpha to use.
    max_alpha_loop : int, optional
        For PM2, Number of times alpha and structure are inferred.
    max_iter : int, optional
        Maximum number of iterations per optimization.
    factr : float, optional
        factr for scipy's L-BFGS-B, alters convergence criteria.
    pgtol : float, optional
        pgtol for scipy's L-BFGS-B, alters convergence criteria.
    alpha_factr : float, optional
        factr for convergence criteria of joint alpha/structure inference.

    Attributes
    ----------
    X_ : array_like of float
        Output of the optimization (typically a 3D structure).
    alpha_ : float
        Inferred alpha (or inputted alpha if alpha is not inferred).
    beta_ : list of float
        Estimated beta (or inputted beta if alpha is not inferred).
    obj_ : float
        Final objective value.
    converged_ : bool
        Whether the optimization successfully converged.
    log_ : list of dict
        Log generated by the callback, containing information about the
        objective function during optimization.
    struct_ : array_like of float
        3D structure resulting from the optimization.
    """

    def __init__(self, counts, lengths, ploidy, alpha, init, bias=None,
                 constraints=None, callback=None, multiscale_factor=1,
                 epsilon=None, epsilon_bounds=None, alpha_init=None,
                 max_alpha_loop=20, max_iter=30000, factr=1e7, pgtol=1e-05,
                 alpha_factr=1e12, reorienter=None, null=False,
                 mixture_coefs=None, verbose=True, mods=()):

        # Check input
        checked = _check_input(
            lengths=lengths, alpha=alpha, counts=counts,
            constraints=constraints, bias=bias, mixture_coefs=mixture_coefs, mods=mods)
        (lengths, alpha, counts, constraints, bias, mixture_coefs,
            mods) = checked
        if alpha_init is not None and (alpha_init > -1 or alpha_init < -4):
            raise ValueError(
                f"Alpha must be between -4 and -1, {alpha_init=:.3g}.")
        if bias is not None:
            if multiscale_factor > 1 and (epsilon is None):
                lengths_lowres = decrease_lengths_res(
                    lengths, multiscale_factor=multiscale_factor)
                if bias.size != lengths_lowres.sum():
                    raise ValueError(
                        "Bias size must be equal to the sum of low-res"
                        f" chromosome lengths ({lengths_lowres.sum()})."
                        f" It is of size {bias.size}.")
            elif bias.size != sum(lengths):
                raise ValueError("Bias size must be equal to the sum of the"
                                 f" chromosome lengths ({sum(lengths)}). It is"
                                 f" of size {bias.size}.")

        if reorienter is not None:
            reorienter.set_multiscale_factor(multiscale_factor)

        self.counts = counts
        self.lengths = lengths
        self.ploidy = ploidy
        self.alpha = alpha
        self.beta_init = [c.beta for c in self.counts]
        self.init_X = init
        self.bias = bias
        self.constraints = constraints
        self.callback = callback
        self.multiscale_factor = multiscale_factor
        self.multiscale_reform = multiscale_factor > 1 and epsilon is not None
        self.epsilon = epsilon
        self.epsilon_bounds = epsilon_bounds
        self.alpha_init = alpha_init
        self.max_alpha_loop = max_alpha_loop
        self.max_iter = max_iter
        self.factr = factr
        self.pgtol = pgtol
        self.alpha_factr = alpha_factr
        self.reorienter = reorienter
        self.null = null
        self.mixture_coefs = mixture_coefs
        self.verbose = verbose
        self.mods = mods

        if self.null:
            if self.verbose:
                print('GENERATING NULL STRUCTURE', flush=True)
            # Exclude the counts data from the primary objective function.
            # Counts are still used in the calculation of the constraints.
            self.counts = (sum(self.counts).as_null(),)

        self._clear()

    def _clear(self):
        self.X_ = self.init_X
        if self.alpha is not None:
            self.alpha_ = self.alpha
        else:
            self.alpha_ = self.alpha_init
        self.beta_ = self.beta_init
        self.epsilon_ = self.epsilon
        self.obj_ = None
        self.alpha_obj_ = None
        self.epsilon_obj_ = None
        self.converged_ = None
        self.alpha_converged_ = None
        self.log_ = None
        self.struct_ = None
        self.orientation_ = None
        self.time_elapsed_ = 0

    def _infer_beta(self, update_counts=True):
        """Estimate beta, given current structure and alpha."""
        # TODO remove this function? It's unused...

        from .estimate_alpha_beta import _estimate_beta

        beta_new = _estimate_beta(
            self.X_.ravel(), self.counts, alpha=self.alpha_,
            lengths=self.lengths, ploidy=self.ploidy, bias=self.bias,
            mixture_coefs=self.mixture_coefs, reorienter=self.reorienter)._value
        if update_counts:
            self.counts = [self.counts[i].update_beta(
                beta_new[i]) for i in range(len(self.counts))]
        return list(beta_new)

    def fit_structure(self, alpha_loop=None):
        """Fit structure to counts data, given current alpha.
        """

        time_start = timer()
        self.X_, self.obj_, self.converged_, self.log_, self.conv_desc_ = estimate_X(
            counts=self.counts,
            init_X=self.X_.ravel(),
            alpha=self.alpha_,
            lengths=self.lengths,
            ploidy=self.ploidy,
            bias=self.bias,
            constraints=self.constraints,
            multiscale_factor=self.multiscale_factor,
            epsilon=self.epsilon_,
            epsilon_bounds=self.epsilon_bounds,
            max_iter=self.max_iter,
            factr=self.factr,
            pgtol=self.pgtol,
            callback=self.callback,
            alpha_loop=alpha_loop,
            reorienter=self.reorienter,
            mixture_coefs=self.mixture_coefs,
            verbose=self.verbose,
            mods=self.mods)
        self.time_elapsed_ += timer() - time_start

        if self.reorienter is not None and self.reorienter.reorient:
            self.orientation_ = self.X_
            self.struct_ = self.reorienter.translate_and_rotate(self.X_)[
                0].reshape(-1, 3)
        else:
            self.struct_, self.epsilon_, _ = _format_X(
                self.X_, lengths=self.lengths, ploidy=self.ploidy,
                multiscale_factor=self.multiscale_factor,
                multiscale_reform=self.multiscale_reform,
                mixture_coefs=self.mixture_coefs, mods=self.mods)
            if self.epsilon_ is not None and isinstance(
                    self.epsilon_, jnp.ndarray):
                self.epsilon_ = self.epsilon_._value
            if len(self.struct_) == 1:
                self.struct_ = self.struct_[0]

    def fit_alpha(self, alpha_loop=None):
        """Fit alpha to counts data, given current structure.
        """

        from .estimate_alpha_beta import estimate_alpha

        if self.multiscale_factor > 1:
            raise ValueError(
                "Alpha can only be inferred using full-resolution structures.")

        time_start = timer()
        self.alpha_, self.alpha_obj_, self.alpha_converged_, self.log_, self.conv_desc_ = estimate_alpha(
            counts=self.counts,
            X=self.X_.ravel(),
            alpha_init=self.alpha_,
            lengths=self.lengths,
            ploidy=self.ploidy,
            bias=self.bias,
            constraints=self.constraints,
            random_state=None,
            max_iter=self.max_iter,
            factr=self.factr,
            pgtol=self.pgtol,
            callback=self.callback,
            alpha_loop=alpha_loop,
            reorienter=self.reorienter,
            mixture_coefs=self.mixture_coefs,
            verbose=self.verbose,
            mods=self.mods)
        self.time_elapsed_ += timer() - time_start

        self.beta_ = [c.beta for c in self.counts]
