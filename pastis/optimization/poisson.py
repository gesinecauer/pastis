import sys
import numpy as np

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

from .utils_poisson import _setup_jax
_setup_jax()
import jax.numpy as ag_np
from jax import grad

from scipy import optimize
import warnings
from timeit import default_timer as timer
from datetime import timedelta

from .multiscale_optimization import decrease_lengths_res
from .counts import _update_betas_in_counts_matrices, NullCountsMatrix
from .utils_poisson import _print_code_header
from .polynomial import _approx_ln_f
from .likelihoods import _masksum, gamma_poisson_nll, poisson_nll


def get_eps_types(stretch_fullres_beads):
    if stretch_fullres_beads is None:
        return None
    eps_types = np.flip(np.sort(np.unique(stretch_fullres_beads)))
    mask = np.isin(eps_types, [0, 1, 2], assume_unique=True, invert=True)
    eps_types = eps_types[mask]
    return eps_types


def get_epsilon_per_bead(epsilon, multiscale_factor, stretch_fullres_beads=None,
                         mean_fullres_nghbr_dis=None):
    if stretch_fullres_beads is None:
        return epsilon

    # Setup epsilon per low-res BEAD
    if ag_np.asarray(epsilon).size == 1:
        epsilon_per_bead = ag_np.full(stretch_fullres_beads.size, epsilon)
        adjusted_eps = None
    elif epsilon.size == stretch_fullres_beads.size:
        epsilon_per_bead = ag_np.asarray(epsilon)
        adjusted_eps = None
    else:
        epsilon_per_bead = ag_np.full(stretch_fullres_beads.size, epsilon[0])
        adjusted_eps = epsilon[1:]

    # If there's only 1 full-res bead per low-res bead:
    # epsilon for that low-res bead is 0
    epsilon_per_bead = epsilon_per_bead.at[stretch_fullres_beads == 1].set(0)

    # If there's 2 consecutive full-res beads per low-res beads:
    # epsilon for that low-res bead is set via the mean dis btwn full-res nghbr
    if multiscale_factor != 2 and mean_fullres_nghbr_dis is not None:
        epsilon_per_bead = epsilon_per_bead.at[stretch_fullres_beads == 2].set(
            mean_fullres_nghbr_dis / np.sqrt(6))

    # If there's > 2 but < multiscale_factor^2 full-res beads per low-res beads:
    # epsilon for that low-res bead is set via adjusted inferred epsilon
    if adjusted_eps is not None:
        eps_types = get_eps_types(stretch_fullres_beads)[1:]
        for i in range(eps_types.size):
            epsilon_per_bead = epsilon_per_bead.at[
                stretch_fullres_beads == eps_types[i]].set(adjusted_eps[i])

    return epsilon_per_bead


def get_epsilon_per_bin(epsilon, row3d, col3d, multiscale_factor,
                        stretch_fullres_beads=None,
                        mean_fullres_nghbr_dis=None):

    if stretch_fullres_beads is None:
        if ag_np.asarray(epsilon).size == 1:
            return epsilon
        else:
            epsilon_per_bin = ag_np.sqrt(
                ag_np.square(epsilon[row3d]) + ag_np.square(
                    epsilon[col3d])) / np.sqrt(2)
            return epsilon_per_bin

    if ag_np.asarray(epsilon).size == stretch_fullres_beads.size:
        epsilon_per_bead = epsilon
    else:
        epsilon_per_bead = get_epsilon_per_bead(
            epsilon, multiscale_factor=multiscale_factor,
            stretch_fullres_beads=stretch_fullres_beads,
            mean_fullres_nghbr_dis=mean_fullres_nghbr_dis)

    # Get epsilon per low-res distance matrix BIN
    epsilon_per_bin = ag_np.sqrt(
        ag_np.square(epsilon_per_bead[row3d]) + ag_np.square(
            epsilon_per_bead[col3d])) / np.sqrt(2)
    return epsilon_per_bin


def get_gamma_moments(struct, epsilon, alpha, beta, row3d, col3d,
                      multiscale_factor, ambiguity='ua',
                      stretch_fullres_beads=None, mean_fullres_nghbr_dis=None,
                      return_mean=True, return_var=True,
                      inferring_alpha=False, mods=[]):

    dis = ag_np.sqrt((ag_np.square(
        struct[row3d] - struct[col3d])).sum(axis=1))
    dis_alpha = ag_np.power(dis, alpha)

    if 'adjust_eps' in mods or ag_np.asarray(epsilon).size > 1:
        epsilon = get_epsilon_per_bin(
            epsilon, row3d=row3d, col3d=col3d,
            multiscale_factor=multiscale_factor,
            stretch_fullres_beads=stretch_fullres_beads,
            mean_fullres_nghbr_dis=mean_fullres_nghbr_dis)
    ln_f_mean, ln_f_var = _approx_ln_f(
        dis, epsilon=epsilon, alpha=alpha, inferring_alpha=inferring_alpha,
        return_mean=return_mean, return_var=return_var)

    gamma_mean = gamma_var = 0
    if return_mean:
        gamma_mean = ag_np.exp(ln_f_mean) * dis_alpha * beta
    if return_var:
        gamma_var = ag_np.exp(ln_f_var) * ag_np.square(dis_alpha * beta)

    if ambiguity != 'ua':
        if ambiguity == 'ambig':
            reshape_0 = 4
        else:
            reshape_0 = 2
        if return_mean:
            gamma_mean = gamma_mean.reshape(reshape_0, -1).sum(axis=0)
        if return_var:
            gamma_var = gamma_var.reshape(reshape_0, -1).sum(axis=0)

    if return_mean and return_var:
        return gamma_mean, gamma_var
    elif return_mean:
        return gamma_mean
    else:
        return gamma_var


def get_gamma_params(struct, epsilon, alpha, beta, row3d, col3d,
                     multiscale_factor, ambiguity='ua',
                     stretch_fullres_beads=None, mean_fullres_nghbr_dis=None,
                     inferring_alpha=False, mods=[]):

    dis = ag_np.sqrt((ag_np.square(
        struct[row3d] - struct[col3d])).sum(axis=1))
    dis_alpha = ag_np.power(dis, alpha)

    eps_gt0 = None
    dis_alpha_eps0 = ag_np.array([])
    if 'adjust_eps' in mods or ag_np.asarray(epsilon).size > 1:
        epsilon = get_epsilon_per_bin(
            epsilon, row3d=row3d, col3d=col3d,
            multiscale_factor=multiscale_factor,
            stretch_fullres_beads=stretch_fullres_beads,
            mean_fullres_nghbr_dis=mean_fullres_nghbr_dis)
    if ('adjust_eps' in mods) and ('eps0' not in mods):
        print('not eps0 mod?'); exit(1)
        eps_gt0 = (stretch_fullres_beads[row3d] * stretch_fullres_beads[col3d] > 1)
        dis_alpha_eps0 = dis_alpha[~eps_gt0]
        dis_alpha = dis_alpha[eps_gt0]
        dis = dis[eps_gt0]
        epsilon = epsilon[eps_gt0]

    ln_f_mean, ln_f_var = _approx_ln_f(
        dis, epsilon=epsilon, alpha=alpha, inferring_alpha=inferring_alpha)

    if ambiguity == 'ua':
        theta_tmp = dis_alpha * ag_np.exp(ln_f_var - ln_f_mean)

        k = ag_np.exp(2 * ln_f_mean - ln_f_var)
    else:
        gamma_mean = ag_np.exp(ln_f_mean) * dis_alpha
        gamma_var = ag_np.exp(ln_f_var) * ag_np.square(dis_alpha)

        if ambiguity == 'ambig':
            reshape_0 = 4
        else:
            reshape_0 = 2
        gamma_mean = gamma_mean.reshape(reshape_0, -1).sum(axis=0)
        gamma_var = gamma_var.reshape(reshape_0, -1).sum(axis=0)

        theta_tmp = gamma_var / gamma_mean
        k = ag_np.square(gamma_mean) / gamma_var

    theta = beta * theta_tmp
    return k, theta, dis_alpha_eps0, eps_gt0


def _multires_negbinom_obj(structures, epsilon, counts, alpha, lengths, ploidy,
                           bias=None, multiscale_factor=1,
                           inferring_alpha=False, stretch_fullres_beads=None,
                           mean_fullres_nghbr_dis=None,
                           mixture_coefs=None, mods=[]):
    """Computes the multiscale objective function for a given counts matrix.
    """

    data_per_bin = counts.fullres_per_lowres_dis().reshape(1, -1)
    bias_per_bin = counts.bias_per_bin(bias)  # TODO

    obj = 0
    for struct in structures:
        k, theta, dis_alpha_eps0, eps_gt0 = get_gamma_params(
            struct, epsilon=epsilon, alpha=alpha, beta=counts.beta,
            row3d=counts.row3d, col3d=counts.col3d,
            multiscale_factor=multiscale_factor, ambiguity=counts.ambiguity,
            stretch_fullres_beads=stretch_fullres_beads,
            mean_fullres_nghbr_dis=mean_fullres_nghbr_dis,
            inferring_alpha=inferring_alpha, mods=mods)
        if dis_alpha_eps0.size == 0:
            obj = obj + gamma_poisson_nll(
                theta=theta, k=k, data=counts.data, bias_per_bin=bias_per_bin,
                mask=counts.mask, mods=mods)
        else:
            if eps_gt0.sum() > 0:
                obj = obj + gamma_poisson_nll(
                    theta=theta, k=k, data=counts.data[:, eps_gt0],
                    bias=bias_per_bin, mask=counts.mask[:, eps_gt0], mods=mods)
            if eps_gt0.sum() < eps_gt0.size:
                obj = obj + poisson_nll(
                    data=counts.data[:, ~eps_gt0], lambda_pois=dis_alpha_eps0,
                    mask=counts.mask[:, ~eps_gt0],
                    data_per_bin=data_per_bin[:, ~eps_gt0])

    if not ag_np.isfinite(obj):
        raise ValueError(
            "Multires (negative binomial) component of objective function for"
            f" {counts.name} is {obj} at {multiscale_factor}x resolution.")

    return counts.weight * obj


def _poisson_obj(structures, counts, alpha, lengths, ploidy, bias=None,
                 multiscale_factor=1, multiscale_variances=None,
                 mixture_coefs=None, mods=[]):
    """Computes the Poisson objective function for a given counts matrix.
    """

    if multiscale_variances is not None:
        if isinstance(multiscale_variances, np.ndarray):
            var_per_dis = multiscale_variances[
                counts.row3d] + multiscale_variances[counts.col3d]
        else:
            var_per_dis = multiscale_variances * 2
    else:
        var_per_dis = 0
    data_per_bin = counts.fullres_per_lowres_dis()

    lambda_intensity = ag_np.zeros(counts.nnz)
    for struct, mix_coef in zip(structures, mixture_coefs):
        dis = ag_np.sqrt((ag_np.square(
            struct[counts.row3d] - struct[counts.col3d])).sum(axis=1))
        if multiscale_variances is None:
            tmp1 = ag_np.power(dis, alpha)
        else:
            tmp1 = ag_np.power(ag_np.square(dis) + var_per_dis, alpha / 2)
        tmp = tmp1.reshape(-1, counts.nnz).sum(axis=0)
        lambda_intensity = lambda_intensity + mix_coef * counts.bias_per_bin(
            bias) * counts.beta * tmp

    # Sum main objective function  # TODO use function in likelihoods.py
    obj = (lambda_intensity * data_per_bin).mean()
    if counts.type != 'zero':
        if lambda_intensity.shape == counts.data.shape:
            counts_data = counts.data
        else:
            counts_data = _masksum(counts.data, mask=counts.mask, axis=0)
        obj = obj - (counts_data * ag_np.log(lambda_intensity)).mean()

    if not ag_np.isfinite(obj):
        raise ValueError(
            f"Poisson component of objective function for {counts.name}"
            f" is {obj} at {multiscale_factor}x resolution.")

    return counts.weight * obj


def _obj_single(structures, counts, alpha, lengths, ploidy, bias=None,
                multiscale_factor=1, multiscale_variances=None, epsilon=None,
                inferring_alpha=False, stretch_fullres_beads=None,
                mean_fullres_nghbr_dis=None, mixture_coefs=None, mods=[]):
    """Computes the objective function for a given individual counts matrix.
    """

    if (bias is not None and bias.sum() == 0) or counts.nnz == 0 or counts.null:
        return 0.
    if np.isnan(counts.weight) or np.isinf(counts.weight) or counts.weight == 0:
        raise ValueError(f"Counts weight may not be {counts.weight}.")

    if mixture_coefs is not None and len(structures) != len(mixture_coefs):
        raise ValueError("The number of structures (%d) and of mixture"
                         " coefficents (%d) should be identical." %
                         (len(structures), len(mixture_coefs)))
    elif mixture_coefs is None:
        mixture_coefs = [1.]

    if epsilon is None or counts.multiscale_factor == 1 or np.all(epsilon == 0):
        obj = _poisson_obj(
            structures=structures, counts=counts, alpha=alpha, lengths=lengths,
            ploidy=ploidy, bias=bias,
            multiscale_factor=counts.multiscale_factor,
            multiscale_variances=multiscale_variances,
            mixture_coefs=mixture_coefs, mods=mods)
    else:
        obj = _multires_negbinom_obj(
            structures=structures, epsilon=epsilon, counts=counts, alpha=alpha,
            lengths=lengths, ploidy=ploidy, bias=bias,
            multiscale_factor=multiscale_factor,
            stretch_fullres_beads=stretch_fullres_beads,
            mean_fullres_nghbr_dis=mean_fullres_nghbr_dis,
            inferring_alpha=inferring_alpha, mixture_coefs=mixture_coefs,
            mods=mods)
        if 'msv' in mods:
            multiscale_variances = 3 / 2 * ag_np.square(epsilon)
            obj_msv = _poisson_obj(
                structures=structures, counts=counts, alpha=alpha, lengths=lengths,
                ploidy=ploidy, bias=bias,
                multiscale_factor=counts.multiscale_factor,
                multiscale_variances=multiscale_variances,
                mixture_coefs=mixture_coefs, mods=mods)
            # print('+++', obj, obj_msv)
            obj = (obj + obj_msv) / 2

    return obj


def objective(X, counts, alpha, lengths, ploidy, bias=None, constraints=None,
              reorienter=None, multiscale_factor=1, multiscale_variances=None,
              multiscale_reform=False, mixture_coefs=None, return_extras=False,
              inferring_alpha=False, stretch_fullres_beads=None,
              mean_fullres_nghbr_dis=None,
              epsilon=None, mods=[]):  # FIXME epsilon shouldn't be defined here unless inferring struct/eps separately
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
    multiscale_variances : float or array of float, optional
        For multiscale optimization at low resolution, the variances of each
        group of full-resolution beads corresponding to a single low-resolution
        bead.

    Returns
    -------
    obj : float
        The total negative log likelihood of the poisson model and constraints.
    """

    # Check format of input
    counts = (counts if isinstance(counts, list) else [counts])
    if lengths is None:
        lengths = np.array([min([min(counts_maps.shape)
                                 for counts_maps in counts])])
    lengths = np.asarray(lengths)
    if bias is None:
        if multiscale_reform:
            bias = np.ones((lengths.sum(),))
        else:
            lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
            bias = np.ones((lengths_lowres.sum(),))

    # Format X
    if reorienter is None or (not reorienter.reorient):
        X, epsilon, mixture_coefs = _format_X(
            X, lengths=lengths, ploidy=ploidy,
            multiscale_factor=multiscale_factor,
            multiscale_reform=multiscale_reform,
            stretch_fullres_beads=stretch_fullres_beads,
            epsilon=epsilon, mixture_coefs=mixture_coefs, mods=mods)
    if epsilon is not None and ('adjust_eps' in mods):
        epsilon_tmp = get_epsilon_per_bead(
            epsilon, multiscale_factor=multiscale_factor,
            stretch_fullres_beads=stretch_fullres_beads,
            mean_fullres_nghbr_dis=mean_fullres_nghbr_dis)
    else:
        epsilon_tmp = epsilon

    # Optionally translate & rotate structures
    if reorienter is not None and reorienter.reorient:
        reorienter.check_X(X)
        structures = reorienter.translate_and_rotate(X)
    else:
        structures = X

    # Check format of structures and mixture_coefs
    if not isinstance(structures, list):
        structures = [structures]
    if mixture_coefs is None:
        mixture_coefs = [1.] * len(structures)
    # nbeads = decrease_lengths_res(lengths, multiscale_factor).sum() * ploidy
    # if structures[0].shape[0] != nbeads:  # TODO fix this
    #     raise ValueError(
    #         f"Expected {nbeads} beads in structure at multiscale_factor="
    #         f"{multiscale_factor}, found {structures[0].shape[0]} beads")

    obj_constraints = {}
    if constraints is not None:
        for constraint in constraints:
            constraint_obj = 0.
            for struct, mix_coef in zip(structures, mixture_coefs):
                constraint_obj = constraint_obj + mix_coef * constraint.apply(
                    struct=struct, alpha=alpha, epsilon=epsilon_tmp,
                    counts=counts, bias=bias, inferring_alpha=inferring_alpha)
            obj_constraints[f"obj_{constraint.abbrev}"] = constraint_obj

    obj_poisson = {}
    obj_poisson_sum = 0.

    for counts_maps in counts:
        obj_counts = _obj_single(
            structures=structures, counts=counts_maps, alpha=alpha,
            lengths=lengths, ploidy=ploidy, bias=bias,
            multiscale_factor=multiscale_factor,
            multiscale_variances=multiscale_variances, epsilon=epsilon_tmp,
            stretch_fullres_beads=stretch_fullres_beads,
            mean_fullres_nghbr_dis=mean_fullres_nghbr_dis,
            inferring_alpha=inferring_alpha, mixture_coefs=mixture_coefs,
            mods=mods)
        obj_poisson[f"obj_{counts_maps.name}"] = obj_counts
        obj_poisson_sum = obj_poisson_sum + obj_counts * counts_maps.nnz

    # Take weighted mean of poisson/negbinom obj terms
    obj_poisson_mean = obj_poisson_sum / sum([c.nnz for c in counts])

    obj = obj_poisson_mean + sum(obj_constraints.values())

    # if type(obj).__name__ in ('DeviceArray', 'ndarray'):
    #     print(f"COUNTS OBJ: {obj_poisson_mean:g}")

    # if type(obj).__name__ in ('DeviceArray', 'ndarray'):
    #     print("OBJ\t" + '\t'.join([f"{k.replace('obj_', '')}: {v._value:.3g}" for k, v in list(obj_poisson.items()) + list(obj_constraints.items())]))  # + f"\tepsilon: {epsilon:.3g}"
    # elif type(obj).__name__ == 'JVPTracer':
    #     print("GRAD\t" + '\t'.join([f"{k.replace('obj_', '')}: {v.primal._value:.3g}" for k, v in obj_poisson.items()]))

    if return_extras:
        obj_logs = {**obj_poisson, **obj_constraints,
                    **{'obj': obj, 'obj_poisson': obj_poisson_mean}}
        return obj, obj_logs, structures, alpha, epsilon
    else:
        return obj


def _format_X(X, lengths=None, ploidy=None, multiscale_factor=1,
              multiscale_reform=False, stretch_fullres_beads=None, epsilon=None,
              mixture_coefs=None, mods=[]):
    """Reformat and check X.
    """
    # TODO epsilon shouldn't be inputted to here unless inferring struct/eps separately

    if mixture_coefs is None:
        mixture_coefs = [1]

    # Get number of beads
    if lengths is None or ploidy is None:
        nbeads = None
    else:
        lengths_lowres = decrease_lengths_res(
            lengths, multiscale_factor=multiscale_factor)
        nbeads = lengths_lowres.sum() * ploidy

    # Get epsilon
    if multiscale_factor > 1 and multiscale_reform and epsilon is None:  # TODO epsilon
        if nbeads is None:
            raise ValueError("Must input `lengths` and `ploidy`.")
        eps_types = get_eps_types(stretch_fullres_beads)

        if X.size == nbeads * 3 * len(mixture_coefs) + 1:
            epsilon = X[-1]
            X = X[:-1]
        elif X.size == nbeads * 3 * len(mixture_coefs) + nbeads:
            epsilon = X[-nbeads:]
            X = X[:-nbeads]
        elif eps_types is not None and (
                X.size == nbeads * 3 * len(mixture_coefs) + eps_types.size):
            tmp = X[-eps_types.size:]
            epsilon_primary = tmp[0]
            epsilon_adjust = ag_np.cumprod(tmp[1:])
            epsilon = ag_np.append(
                epsilon_primary, epsilon_adjust * epsilon_primary)
            X = X[:-eps_types.size]
        else:
            raise ValueError(
                f"Epsilon must be of length 1 or equal to the number of beads"
                f" ({nbeads}). X.shape = ({', '.join(map(str, X.shape))}).")
    else:
        #epsilon = None  # TODO epsilon
        pass

    try:
        X = X.reshape(-1, 3)
    except ValueError:
        raise ValueError("X should contain k 3D structures, X.shape ="
                         f" ({', '.join(map(str, X.shape))})")

    k = len(mixture_coefs)
    n = int(X.shape[0] / k)
    if n != X.shape[0] / k:
        raise ValueError("X.shape[0] should be divisible by the length of"
                         f" mixture_coefs, {k}. X.shape ="
                         f" ({', '.join(map(str, X.shape))})")
    if nbeads is not None and n != nbeads:
        raise ValueError(f"Structures must be of length {nbeads}. They are"
                         f" of length {n}.")
    X = [X[i * n:(i + 1) * n] for i in range(k)]

    return X, epsilon, mixture_coefs


def objective_wrapper(X, counts, alpha, lengths, ploidy, bias=None,
                      constraints=None, reorienter=None, multiscale_factor=1,
                      multiscale_variances=None, multiscale_reform=False,
                      stretch_fullres_beads=None, mean_fullres_nghbr_dis=None,
                      callback=None, mixture_coefs=None, mods=[]):
    """Objective function wrapper to match scipy.optimize's interface.
    """

    new_obj, obj_logs, structures, alpha, epsilon = objective(
        X, counts=counts, alpha=alpha, lengths=lengths, ploidy=ploidy,
        bias=bias, constraints=constraints, reorienter=reorienter,
        multiscale_factor=multiscale_factor,
        multiscale_variances=multiscale_variances,
        multiscale_reform=multiscale_reform,
        stretch_fullres_beads=stretch_fullres_beads,
        mean_fullres_nghbr_dis=mean_fullres_nghbr_dis,
        mixture_coefs=mixture_coefs, return_extras=True, mods=mods)

    if callback is not None:
        callback.on_iter_end(obj_logs=obj_logs, structures=structures,
                             alpha=alpha, Xi=X, epsilon=epsilon)

    return new_obj


gradient = grad(objective)


def fprime_wrapper(X, counts, alpha, lengths, ploidy, bias=None,
                   constraints=None, reorienter=None, multiscale_factor=1,
                   multiscale_variances=None, multiscale_reform=False,
                   stretch_fullres_beads=None, mean_fullres_nghbr_dis=None,
                   callback=None, mixture_coefs=None, mods=[]):
    """Gradient function wrapper to match scipy.optimize's interface.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Using a non-tuple sequence for multidimensional"
            " indexing is deprecated", category=FutureWarning)
        new_grad = np.array(gradient(
            X, counts=counts, alpha=alpha, lengths=lengths, ploidy=ploidy,
            bias=bias, constraints=constraints, reorienter=reorienter,
            multiscale_factor=multiscale_factor,
            multiscale_variances=multiscale_variances,
            multiscale_reform=multiscale_reform,
            stretch_fullres_beads=stretch_fullres_beads,
            mean_fullres_nghbr_dis=mean_fullres_nghbr_dis,
            mixture_coefs=mixture_coefs, mods=mods)).flatten()

    return new_grad


def estimate_X(counts, init_X, alpha, lengths, ploidy, bias=None,
               constraints=None, multiscale_factor=1, multiscale_variances=None,
               epsilon=None, epsilon_bounds=None, stretch_fullres_beads=None,
               mean_fullres_nghbr_dis=None, max_iter=30000, max_fun=None,
               factr=1e7, pgtol=1e-05, callback=None, alpha_loop=0, epsilon_loop=0,
               reorienter=None, mixture_coefs=None,
               verbose=True, mods=[]):
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

    # from jax import random; x = random.uniform(random.PRNGKey(0), (1000,), dtype=ag_np.float64); print(x.dtype); exit(0)

    multiscale_reform = (epsilon is not None)

    # Check format of input
    counts = (counts if isinstance(counts, list) else [counts])
    lengths = np.array(lengths)
    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
    if bias is None:
        if multiscale_reform:
            bias = np.ones((lengths.sum(),))
        else:
            bias = np.ones((lengths_lowres.sum(),))
    bias = np.array(bias)

    if (multiscale_factor != 1 and multiscale_reform):
        x0 = np.append(init_X.flatten(), epsilon)
    else:
        x0 = init_X.flatten()

    if verbose:
        #print('=' * 30, flush=True) # TODO removed
        print("\nRUNNING THE L-BFGS-B CODE\n\n           * * *\n\nMachine"
              " precision = %.4g\n" % np.finfo(np.float).eps, flush=True)

    if callback is not None:
        if reorienter is not None and reorienter.reorient:
            opt_type = 'structure.chrom_reorient'
        else:
            opt_type = 'structure'
        callback.on_training_begin(
            opt_type=opt_type, alpha_loop=alpha_loop, epsilon_loop=epsilon_loop)
        obj = objective_wrapper(
            x0, counts=counts, alpha=alpha, lengths=lengths, ploidy=ploidy,
            bias=bias, constraints=constraints, reorienter=reorienter,
            multiscale_factor=multiscale_factor,
            multiscale_variances=multiscale_variances,
            multiscale_reform=multiscale_reform,
            stretch_fullres_beads=stretch_fullres_beads,
            mean_fullres_nghbr_dis=mean_fullres_nghbr_dis, callback=callback,
            mixture_coefs=mixture_coefs, mods=mods)
    else:
        obj = np.nan

    if multiscale_reform:
        bounds = np.append(
            np.full((init_X.flatten().shape[0], 2), None),
            np.array(epsilon_bounds).reshape(-1, 2), axis=0)
        if ag_np.asarray(epsilon).size > 1:
            eps_types = get_eps_types(stretch_fullres_beads)
            bounds_xtra = np.tile(np.array([1e-6, 1]), (eps_types.size - 1, 1))
            bounds = np.append(bounds, bounds_xtra, axis=0)
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
                  reorienter, multiscale_factor, multiscale_variances,
                  multiscale_reform, stretch_fullres_beads,
                  mean_fullres_nghbr_dis, callback, mixture_coefs, mods))
        X, obj, d = results
        converged = d['warnflag'] == 0
        conv_desc = d['task']
        if isinstance(conv_desc, bytes):
            conv_desc = conv_desc.decode('utf8')

    history = None
    if callback is not None:
        callback.on_training_end()
        history = callback.history

    if verbose:
        if multiscale_reform:
            _, final_epsilon, _ = _format_X(
                X, lengths=lengths, ploidy=ploidy,
                multiscale_factor=multiscale_factor,
                multiscale_reform=multiscale_reform,
                stretch_fullres_beads=stretch_fullres_beads,
                mixture_coefs=mixture_coefs, mods=mods)
            # print(f'INIT EPSILON: {ag_np.asarray(epsilon).mean():.3g},'
            #       f'  FINAL EPSILON: {ag_np.asarray(final_epsilon).mean():.3g}',
            #       flush=True)  # Removed because is confusing with epsilon_per_bead
        if converged:
            print('CONVERGED\n', flush=True)
        else:
            print('OPTIMIZATION DID NOT CONVERGE', flush=True)
            print(conv_desc + '\n', flush=True)

    return X, obj, converged, history, conv_desc


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
    multiscale_variances : float or array_like of float, optional
        For multiscale optimization at low resolution, the variances of each
        group of full-resolution beads corresponding to a single low-resolution
        bead.
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
    history_ : list of dict
        History generated by the callback, containing information about the
        objective function during optimization.
    struct_ : array_like of float of shape (lengths.sum() * ploidy, 3)
        3D structure resulting from the optimization.
    """

    def __init__(self, counts, lengths, ploidy, alpha, init, bias=None,
                 constraints=None, callback=None, multiscale_factor=1,
                 multiscale_variances=None, epsilon=None, epsilon_bounds=None,
                 stretch_fullres_beads=None, mean_fullres_nghbr_dis=None,
                 epsilon_coord_descent=False, alpha_init=None, max_alpha_loop=20, max_iter=30000,
                 factr=1e7, pgtol=1e-05, alpha_factr=1e12,
                 reorienter=None, null=False, mixture_coefs=None, verbose=True,
                 mods=[]):

        from .piecewise_whole_genome import ChromReorienter
        from .callbacks import Callback

        lengths = np.asarray(lengths)

        if constraints is None:
            constraints = []
        if callback is None:
            callback = Callback(
                lengths=lengths, ploidy=ploidy, counts=counts,
                multiscale_factor=multiscale_factor,
                multiscale_reform=(epsilon is not None),
                frequency={'print': 100, 'history': 100, 'save': None})
        # if reorienter is None:
        #     reorienter = ChromReorienter(lengths=lengths, ploidy=ploidy)
        if reorienter is not None:
            reorienter.set_multiscale_factor(multiscale_factor)

        self.counts = counts
        self.lengths = lengths
        self.ploidy = ploidy
        self.alpha = alpha
        self.init_X = init
        self.bias = bias
        self.constraints = constraints
        self.callback = callback
        self.multiscale_factor = multiscale_factor
        self.multiscale_variances = multiscale_variances
        self.multiscale_reform = multiscale_factor > 1 and epsilon is not None
        self.epsilon = epsilon
        self.epsilon_bounds = epsilon_bounds
        self.stretch_fullres_beads = stretch_fullres_beads
        self.mean_fullres_nghbr_dis = mean_fullres_nghbr_dis
        self.epsilon_coord_descent = epsilon_coord_descent
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
            print('GENERATING NULL STRUCTURE', flush=True)
            # Dummy counts need to be inputted because we need to know which
            # row/col to include in calculations of constraints
            self.counts = [NullCountsMatrix(
                counts=self.counts, lengths=self.lengths, ploidy=self.ploidy,
                multiscale_factor=self.multiscale_factor)]

        self._clear()

    def _clear(self):
        self.X_ = self.init_X
        if self.alpha is not None:
            self.alpha_ = self.alpha
        else:
            self.alpha_ = self.alpha_init
        self.beta_ = [c.beta for c in self.counts if c.sum() != 0]
        self.epsilon_ = self.epsilon
        self.obj_ = None
        self.alpha_obj_ = None
        self.epsilon_obj_ = None
        self.converged_ = None
        self.alpha_converged_ = None
        self.history_ = None
        self.struct_ = None
        self.orientation_ = None
        self.time_elapsed_ = 0

    def _infer_beta(self, update_counts=True, verbose=True):
        """Estimate beta, given current structure and alpha.
        """

        from .estimate_alpha_beta import _estimate_beta

        if (self.multiscale_factor > 1 and self.multiscale_reform):
            X_ = np.append(self.X_.flatten(), self.epsilon)
        else:
            X_ = self.X_.flatten()

        new_beta = _estimate_beta(
            X_, self.counts, alpha=self.alpha_,
            lengths=self.lengths, ploidy=self.ploidy, bias=self.bias,
            multiscale_factor=self.multiscale_factor,
            multiscale_variances=self.multiscale_variances,
            multiscale_reform=self.multiscale_reform,
            mixture_coefs=self.mixture_coefs,
            reorienter=self.reorienter, verbose=verbose)
        if update_counts:
            self.counts = _update_betas_in_counts_matrices(
                counts=self.counts, beta=new_beta)
        return list(new_beta.values())

    def _fit_structure(self, alpha_loop=None):
        """Fit structure to counts data, given current alpha.
        """

        # if alpha_loop is not None:
        #     _print_code_header([
        #         "Jointly inferring structure & alpha",
        #         f"Inferring STRUCTURE #{alpha_loop}"], max_length=50)

        time_start = timer()
        self.X_, self.obj_, self.converged_, self.history_, self.conv_desc_ = estimate_X(
            counts=self.counts,
            init_X=self.X_.flatten(),
            alpha=self.alpha_,
            lengths=self.lengths,
            ploidy=self.ploidy,
            bias=self.bias,
            constraints=self.constraints,
            multiscale_factor=self.multiscale_factor,
            multiscale_variances=self.multiscale_variances,
            epsilon=self.epsilon_,
            epsilon_bounds=self.epsilon_bounds,
            stretch_fullres_beads=self.stretch_fullres_beads,
            mean_fullres_nghbr_dis=self.mean_fullres_nghbr_dis,
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

        self.X_, self.epsilon_, _ = _format_X(
            self.X_, lengths=self.lengths, ploidy=self.ploidy,
            multiscale_factor=self.multiscale_factor,
            multiscale_reform=self.multiscale_reform,
            stretch_fullres_beads=self.stretch_fullres_beads,
            mixture_coefs=self.mixture_coefs, mods=self.mods)
        if self.epsilon_ is not None and isinstance(
                self.epsilon_, ag_np.ndarray):
            self.epsilon_ = self.epsilon_._value

        if self.reorienter is not None and self.reorienter.reorient:
            self.orientation_ = self.X_
            self.struct_ = self.reorienter.translate_and_rotate(self.X_)[
                0].reshape(-1, 3)
        else:
            self.struct_ = [struct.reshape(-1, 3) for struct in self.X_]
            if len(self.struct_) == 1:
                self.struct_ = self.struct_[0]

    def _fit_alpha(self, alpha_loop=None):
        """Fit alpha to counts data, given current structure.
        """

        from .estimate_alpha_beta import estimate_alpha

        # if alpha_loop is not None:
        #     _print_code_header([
        #         "Jointly inferring structure & alpha",
        #         f"Inferring ALPHA #{alpha_loop}"], max_length=50)

        time_start = timer()
        self.alpha_, self.alpha_obj_, self.alpha_converged_, self.history_, self.conv_desc_ = estimate_alpha(
            counts=self.counts,
            X=self.X_.flatten(),
            alpha_init=self.alpha_,
            lengths=self.lengths,
            ploidy=self.ploidy,
            bias=self.bias,
            constraints=self.constraints,
            multiscale_factor=self.multiscale_factor,
            multiscale_variances=self.multiscale_variances,
            epsilon=self.epsilon_,
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

        self.beta_ = self._infer_beta()

    # def _fit_epsilon(self, inferring_epsilon, alpha_loop=0, epsilon_loop=0):
    #     """Fit structure/epsilon to counts, given current structure/epsilon.
    #     """
    #     from .estimate_epsilon import estimate_epsilon

    #     if inferring_epsilon:
    #         init_X = self.epsilon_
    #         epsilon = None
    #         structures = self.X_.flatten()
    #     else:
    #         init_X = self.X_.flatten()
    #         epsilon = self.epsilon_
    #         structures = None

    #     new_X_, self.epsilon_obj_, self.converged_, self.history_, self.conv_desc_ = estimate_epsilon(
    #         counts=self.counts,
    #         init_X=init_X,
    #         alpha=self.alpha_,
    #         lengths=self.lengths,
    #         ploidy=self.ploidy,
    #         bias=self.bias,
    #         constraints=self.constraints,
    #         multiscale_factor=self.multiscale_factor,
    #         epsilon=epsilon,
    #         structures=structures,
    #         epsilon_bounds=self.epsilon_bounds,
    #         stretch_fullres_beads=self.stretch_fullres_beads,
    #         mean_fullres_nghbr_dis=self.mean_fullres_nghbr_dis,
    #         max_iter=self.max_iter,
    #         factr=self.factr,
    #         pgtol=self.pgtol,
    #         callback=self.callback,
    #         alpha_loop=alpha_loop,
    #         epsilon_loop=epsilon_loop,
    #         reorienter=self.reorienter,
    #         mixture_coefs=self.mixture_coefs,
    #         verbose=self.verbose,
    #         mods=self.mods)

    #     if inferring_epsilon:
    #         self.epsilon_ = new_X_
    #         if self.epsilon_.size == 1:
    #             self.epsilon_ = self.epsilon_[0]
    #     else:
    #         self.X_ = new_X_

    # def _fit_naive_multiscale(self, alpha_loop=0):
    #     """Fit structure to counts data, given current alpha. TODO
    #     """

    #     if self.multiscale_factor == 1:
    #         return False

    #     if self.epsilon is None and (self.multiscale_variances is None or self.multiscale_variances == 0):
    #         return False

    #     if not self.multiscale_reform:
    #         return False

    #     _print_code_header(
    #         "Inferring with naive multiscale", max_length=50, blank_lines=1)

    #     self.X_, self.obj_, self.converged_, self.history_, self.conv_desc_ = estimate_X(
    #         counts=self.counts,
    #         init_X=self.X_.flatten(),
    #         alpha=self.alpha_,
    #         lengths=self.lengths,
    #         ploidy=self.ploidy,
    #         bias=self.bias,
    #         constraints=self.constraints,
    #         multiscale_factor=self.multiscale_factor,
    #         multiscale_variances=None,
    #         epsilon=None,
    #         epsilon_bounds=None,
    #         stretch_fullres_beads=self.stretch_fullres_beads,
    #         mean_fullres_nghbr_dis=self.mean_fullres_nghbr_dis,
    #         max_iter=self.max_iter,
    #         factr=self.factr,
    #         pgtol=self.pgtol,
    #         callback=self.callback,
    #         alpha_loop=alpha_loop,
    #         epsilon_loop=-1,
    #         reorienter=self.reorienter,
    #         mixture_coefs=self.mixture_coefs,
    #         verbose=self.verbose,
    #         mods=self.mods)

    #     if self.callback.epsilon_true is not None:
    #         self.history_['epsilon_nrmse'] = [None] * len(self.history_['iter'])

    #     return True

    def _fit_struct_alpha_jointly(self, time_start, infer_structure_first=True):
        """Jointly fit structure & alpha to counts data.
        """
        if self.alpha is not None:
            self._fit_structure()
            return

        if infer_structure_first:
            _print_code_header([
                "Jointly inferring structure & alpha",
                f"Inferring STRUCTURE #0, initial alpha={self.alpha_init:.3g}"],
                max_length=50)
            self._fit_structure(alpha_loop=0)
            if not self.converged_:
                return

        prev_alpha_obj = None
        for alpha_loop in range(1, self.max_alpha_loop + 1):
            # time_current = str(
            #     timedelta(seconds=round(timer() - time_start)))
            # _print_code_header([
            #     "Jointly inferring structure & alpha",
            #     f"Inferring ALPHA #{alpha_loop},"
            #     f" total time={time_current}"], max_length=50)
            self._fit_alpha(alpha_loop=alpha_loop)
            if not self.alpha_converged_:
                break
            # time_current = str(
            #     timedelta(seconds=round(timer() - time_start)))
            # _print_code_header([
            #     "Jointly inferring structure & alpha",
            #     f"Inferring STRUCTURE #{alpha_loop},"
            #     f" total time={time_current}"], max_length=50)
            self._fit_structure(alpha_loop=alpha_loop)
            if not self.converged_:
                break
            if _convergence_criteria(
                    f_k=prev_alpha_obj, f_kplus1=self.alpha_obj_,
                    factr=self.alpha_factr):
                break
            prev_alpha_obj = self.alpha_obj_

    # def _fit_struct_epsilon_jointly(self, time_start, alpha_loop=0,
    #                                 infer_structure_first=True):
    #     """Jointly fit structure & epsilon to counts data.
    #     """
    #     # FIXME this is obviously temporary...
    #     self.max_epsilon_loop = max_alpha_loop
    #     self.epsilon_factr = alpha_factr

    #     if self.multiscale_reform and self.verbose:
    #         print(f"Epsilon init = {self.epsilon:.3g}, bounds = ["
    #               f"{self.epsilon_bounds[0]:.3g},"
    #               f" {self.epsilon_bounds[1]:.3g}]", flush=True)

    #     fit_naive_multiscale = 'naive1st' in self.mods
    #     if fit_naive_multiscale:
    #         fit_naive_multiscale = self._fit_naive_multiscale()

    #     if not (self.multiscale_reform and self.epsilon_coord_descent):
    #         if fit_naive_multiscale:
    #             _print_code_header(
    #                 "Inferring with NegBinom multiscale", max_length=50,
    #                 blank_lines=1)
    #         self._fit_structure()
    #         return

    #     only_infer_epsilon_once = False
    #     if only_infer_epsilon_once:
    #         self._fit_epsilon(
    #             inferring_epsilon=True, alpha_loop=alpha_loop,
    #             epsilon_loop=1)
    #         return

    #     if infer_structure_first and not fit_naive_multiscale:
    #         _print_code_header([
    #             "Jointly inferring structure & epsilon",
    #             f"Inferring STRUCTURE #0, initial epsilon={self.epsilon:.3g}"],
    #             max_length=50)
    #         self._fit_epsilon(
    #             inferring_epsilon=False, alpha_loop=alpha_loop, epsilon_loop=0)
    #         if not self.converged_:
    #             return

    #     prev_epsilon_obj = None
    #     for epsilon_loop in range(1, self.max_epsilon_loop + 1):
    #         time_current = str(
    #             timedelta(seconds=round(timer() - time_start)))
    #         _print_code_header([
    #             "Jointly inferring structure & epsilon",
    #             f"Inferring EPSILON #{epsilon_loop},"
    #             f" total time={time_current}"], max_length=50)
    #         self._fit_epsilon(
    #             inferring_epsilon=True, alpha_loop=alpha_loop,
    #             epsilon_loop=epsilon_loop)
    #         if not self.converged_:
    #             break
    #         time_current = str(
    #             timedelta(seconds=round(timer() - time_start)))
    #         _print_code_header([
    #             "Jointly inferring structure & epsilon",
    #             f"Inferring STRUCTURE #{epsilon_loop},"
    #             f" total time={time_current}"], max_length=50)
    #         self._fit_epsilon(
    #             inferring_epsilon=False, alpha_loop=alpha_loop,
    #             epsilon_loop=epsilon_loop)
    #         if not self.converged_:
    #             break
    #         if _convergence_criteria(
    #                 f_k=prev_epsilon_obj, f_kplus1=self.epsilon_obj_,
    #                 factr=self.epsilon_factr):
    #             break
    #         prev_epsilon_obj = self.epsilon_obj_

    def fit(self):
        """Fit structure to counts data, optionally estimate alpha & epsilon.

        Returns
        -------
        self : returns an instance of self.
        """

        self._clear()

        # Infer structure
        time_start = timer()
        self._fit_struct_alpha_jointly(time_start)
        self.time_elapsed_ = timer() - time_start
        time_current = str(timedelta(seconds=round(self.time_elapsed_)))
        print(f"OPTIMIZATION AT {self.multiscale_factor}X RESOLUTION COMPLETE,"
              f" TOTAL ELAPSED TIME={time_current}", flush=True)

        # if self.reorienter.reorient:
        #     self.orientation_ = self.X_
        #     self.struct_ = self.reorienter.translate_and_rotate(self.X_)[
        #         0].reshape(-1, 3)
        # else:
        #     self.struct_ = self.X_.reshape(-1, 3)

        return self
