import sys
import numpy as np

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

from scipy import optimize
import warnings
from timeit import default_timer as timer
from datetime import timedelta
import autograd.numpy as ag_np
from autograd.builtins import SequenceBox
from autograd import grad
from autograd.extend import primitive, defvjp
from autograd.scipy.special import gammaln as ag_gammaln
from autograd.scipy.special import gamma as ag_gamma
from .multiscale_optimization import decrease_lengths_res
from .counts import _update_betas_in_counts_matrices, NullCountsMatrix
from .constraints import Constraints
from .callbacks import Callback
from .utils_poisson import _print_code_header


def my_polevl(x, coefs):
    """TODO
    """
    ans = 0
    power = len(coefs) - 1
    for coef in coefs:
        ans = ans + coef * ag_np.power(x, power)
        power = power - 1
    return ans


def stirling_polyval_good(z):
    """TODO
    """
    sterling_coefs = [
        8.11614167470508450300E-4, -5.95061904284301438324E-4,
        7.93650340457716943945E-4, -2.77777777730099687205E-3,
        8.33333333333331927722E-2]
    z_sq_inv = 1. / (z * z)
    return my_polevl(z_sq_inv, coefs=sterling_coefs) / z


@primitive
def _stirling_polyval(z):
    """TODO
    """
    sterling_coefs = [
        8.11614167470508450300E-4, -5.95061904284301438324E-4,
        7.93650340457716943945E-4, -2.77777777730099687205E-3,
        8.33333333333331927722E-2]
    z_sq_inv = 1. / (z * z)
    return ag_np.polyval(sterling_coefs, z_sq_inv) / z


def _stirling_polyval_deriv(z):
    """TODO
    """
    # FIXME if z >= 1e17 then return 0
    sterling_coefs_deriv = [
        8.33333333333333333333E-2, -2.10927960927960927961E-2,
        7.57575757575757575758E-3, -4.16666666666666666667E-3,
        3.96825396825396825397E-3, -8.33333333333333333333E-3,
        8.33333333333333333333E-2]
    z_sq_inv = 1. / (z * z)
    return - ag_np.polyval(sterling_coefs_deriv, z_sq_inv) * z_sq_inv


defvjp(_stirling_polyval, lambda ans, x: lambda g: g * _stirling_polyval_deriv(x))


def _pois_sum(arr, nnz):
    """TODO
    """

    return arr.reshape(-1, nnz).sum(axis=0)


def _multiscale_reform_obj(structures, epsilon, counts, alpha, lengths,
                           bias=None, multiscale_factor=1, mixture_coefs=None):
    """Computes the multiscale objective function for a given counts matrix.
    """

    #epsilon = 1e-50 # FIXME

    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
    ploidy = int(structures[0].shape[0] / lengths_lowres.sum())

    num_highres_per_lowres_bins = counts.count_fullres_per_lowres_bins(
        multiscale_factor).reshape(1, -1)

    bias_per_bin = counts.bias_per_bin(bias, ploidy)  # TODO

    epsilon_sq = ag_np.square(epsilon)
    alpha_sq = ag_np.square(alpha)

    assume_epsilon_small = False
    taylor_theta = True  # FIXME (obj errors if False)
    taylor_k = False  # FIXME (obj errors if True)
    mu_is_theta_k = True  # FIXME (obj errors if False)
    mu = ag_np.zeros((1, counts.nnz_lowres))
    theta = ag_np.zeros((1, counts.nnz_lowres))
    k = ag_np.zeros((1, counts.nnz_lowres))
    lambda_intensity = ag_np.zeros(counts.nnz_lowres)
    for struct, mix_coef in zip(structures, mixture_coefs):
        dis = ag_np.sqrt((ag_np.square(
            struct[counts.row3d] - struct[counts.col3d])).sum(axis=1))
        dis_sq = ag_np.square(dis)
        if assume_epsilon_small:
            dis_alpha = ag_np.power(dis, alpha)
            if counts.ambiguity == 'ua':
                mu_tmp = 1 + 1.5 * alpha * epsilon_sq / dis_sq
                theta_tmp1 = alpha_sq * epsilon_sq * dis_alpha / dis_sq
                if taylor_theta:
                    theta_tmp2 = 1 - 1.5 * alpha * epsilon_sq / dis_sq
                else:
                    theta_tmp2 = 1 / mu_tmp
                k_tmp1 = dis_sq / alpha_sq / epsilon_sq
                if taylor_k:
                    k_tmp2 = 1 + 3 * alpha * epsilon_sq / dis_sq
                else:
                    k_tmp2 = ag_np.square(mu_tmp)
            else:
                raise ValueError("lol.")
                sum_dis_2alpha_neg2 = _pois_sum(
                    ag_np.power(dis, 2 * alpha - 2), counts.nnz_lowres)
                sum_dis_alpha = _pois_sum(dis_alpha, counts.nnz_lowres)
                sum_dis_alpha_neg2 = _pois_sum(
                    dis_alpha / dis_sq, counts.nnz_lowres)
                mu_tmp = 1 + 1.5 * alpha * epsilon_sq * sum_dis_alpha_neg2 / sum_dis_alpha
                theta_tmp1 = alpha_sq * epsilon_sq * sum_dis_2alpha_neg2 / sum_dis_alpha
                if taylor_theta:
                    theta_tmp2 = 1 - 1.5 * alpha * epsilon_sq * sum_dis_alpha_neg2 / sum_dis_alpha
                else:
                    theta_tmp2 = 1 / mu_tmp
                k_tmp1 = ag_np.square(sum_dis_alpha) / alpha_sq / epsilon_sq / sum_dis_2alpha_neg2
                if taylor_k:
                    k_tmp2 = 1 + 3 * alpha * epsilon_sq * sum_dis_alpha_neg2 / sum_dis_alpha
                else:
                    k_tmp2 = ag_np.square(mu_tmp)
            theta = theta + mix_coef * counts.beta * theta_tmp1 * theta_tmp2
            k = k + mix_coef * k_tmp1 * k_tmp2
            mu = mu + mix_coef * counts.beta * dis_alpha * mu_tmp
        else:
            gamma_tmp = dis_sq + 3 * epsilon_sq
            gamma_mean = ag_np.power(gamma_tmp, alpha / 2)
            gamma_var = (alpha_sq / 4) * ag_np.power(gamma_tmp, alpha - 2) * (
                4 * epsilon_sq * dis_sq + 6 * ag_np.power(epsilon, 4))
            theta = theta + mix_coef * counts.beta * (gamma_var / gamma_mean)
            k = k + mix_coef * (ag_np.square(gamma_mean) / gamma_var)
            mu = mu + mix_coef * counts.beta * gamma_mean
        # below is just for lambda_intensity, can remove
        tmp1 = ag_np.power(dis, alpha)
        tmp = tmp1.reshape(-1, counts.nnz_lowres).sum(axis=0)
        lambda_intensity = lambda_intensity + mix_coef * counts.bias_per_bin(
            bias, ploidy) * counts.beta * tmp

    if assume_epsilon_small and mu_is_theta_k:
        mu = theta * k

    # if type(theta).__name__ != 'ArrayBox' and not np.allclose(theta * k, mu) and counts.type != 'zero':
    #     from topsy.utils.misc import printvars  # FIXME
    #     print(f"\nθk != μ for ε={epsilon}")
    #     printvars({
    #         # 'θ': theta, 'k': k,
    #         'θk': theta * k, 'μ': mu, 'λ': lambda_intensity,
    #         'θk-μ': theta * k - mu})

    if epsilon < 1e-10:
        obj_pois_mu = (num_highres_per_lowres_bins * mu).sum() - (
            counts.data_grouped.sum(axis=0) * ag_np.log(mu)).sum()

        if ag_np.isnan(obj_pois_mu) or ag_np.isinf(obj_pois_mu):
            from topsy.utils.misc import printvars  # FIXME
            printvars({
                'ε': epsilon, 'dis': dis, 'θ': theta, 'k': k, 'λ': lambda_intensity, 'μ': mu,
                'ag_np.log(μ)': ag_np.log(mu)})
            raise ValueError(
                f"Poisson component of objective function for {counts.name}"
                f" is {obj_pois_mu}.")

        return counts.weight * obj_pois_mu

    eps = 1e-6
    check_instability = False

    log1p_theta = ag_np.log1p(theta)
    obj_tmp1 = -num_highres_per_lowres_bins * k * log1p_theta
    if counts.type == 'zero':
        obj = ag_np.sum(obj_tmp1)
    else:
        obj_tmp2 = -num_highres_per_lowres_bins * stirling_polyval_good(k)
        obj_tmp3 = ag_np.sum(stirling_polyval_good(counts.data_grouped + k), axis=0)
        obj_tmp4 = ag_np.sum(counts.data_grouped, axis=0) * (
            ag_np.log(mu) - log1p_theta - 1)
        if check_instability:
            possible_instability = counts.data_grouped / k < eps
            stable_rows, stable_cols = ag_np.where(~possible_instability)
            stable_counts = counts.data_grouped[stable_rows, stable_cols]
            stable_k = k[:, stable_cols]
            obj_tmp5 = counts.data_grouped[possible_instability].sum() + (
                (stable_counts + stable_k - 0.5) * ag_np.log1p(
                    stable_counts / stable_k)).sum()
            # FIXME make sure dimensions of obj_tmp5 are correct here (1, -1)
        else:
            obj_tmp5 = ((counts.data_grouped + k - 0.5) * ag_np.log1p(
                counts.data_grouped / k)).sum(axis=0)
        obj = ag_np.sum(obj_tmp1 + obj_tmp2 + obj_tmp3 + obj_tmp4 + obj_tmp5)

    from topsy.utils.misc import printvars
    if False and type(theta).__name__ != 'ArrayBox' and counts.type != 'zero' and np.allclose(epsilon, 1e-50):
        obj_tmp4_theta0 = ag_np.sum(counts.data_grouped, axis=0) * (
            ag_np.log(mu) - 0 - 1)
        nxny_mu = num_highres_per_lowres_bins * mu
        obj_pois = (num_highres_per_lowres_bins * lambda_intensity).sum() - (counts.data_grouped.sum(axis=0) * ag_np.log(
            lambda_intensity)).sum()
        obj_pois_mu = nxny_mu.sum() - (counts.data_grouped.sum(axis=0) * ag_np.log(mu)).sum()

        log1p_theta = ag_np.log1p(theta)
        obj_tmp1 = -num_highres_per_lowres_bins * k * log1p_theta

        actual_pois_obj = _poisson_obj_single(
            structures=structures, counts=counts, alpha=alpha, lengths=lengths,
            bias=bias, multiscale_factor=multiscale_factor,
            multiscale_variances=None, epsilon=0, mixture_coefs=mixture_coefs)
        print()
        printvars({
            'obj_pois λ': -obj_pois,
            'obj_pois μ': -obj_pois_mu,
            'fix2_pois μ': - (nxny_mu.sum() - (counts.data_grouped * ag_np.log(mu)).sum()),  # fixes: nxny_mu (doesn't matter if you sum(axis=0) counts data grouped before mult by ln(μ))
            '(∑cij)lnμ -nxnyμ': (counts.data_grouped * ag_np.log(mu)).sum() - \
            nxny_mu.sum(),
            'obj': obj,
            'actual pois obj': -actual_pois_obj})  # 'obj + ∑cij': obj + counts.data_grouped.sum())
        print()
        A_ideal_mu = counts.data_grouped.sum(axis=0) * (ag_np.log(mu) - 1)
        BpC_ideal_mu = -num_highres_per_lowres_bins * mu
        D_ideal = counts.data_grouped.sum(axis=0)
        printvars({
            'ε': epsilon, 'dis': dis, 'θ': theta, 'k': k, 'nxny': num_highres_per_lowres_bins,
            'λ': lambda_intensity, 'μ': mu, 'nxny μ': nxny_mu,
            '': np.array([]),
            # 'A': obj_tmp4,
            # 'A θ=0': obj_tmp4_theta0,
            # 'A ideal μ': A_ideal_mu,
            # 'A ideal λ': counts.data_grouped.sum(axis=0) * (ag_np.log(lambda_intensity) - 1),
            # 'B+C': obj_tmp1 + obj_tmp2 + obj_tmp3,
            # 'B+C ideal μ': BpC_ideal_mu,
            # 'B+C ideal λ': -num_highres_per_lowres_bins * lambda_intensity,
            # 'tmp2+tmp3': obj_tmp2 + obj_tmp3,
            # '    ': np.array([]),
            # '-λ': -lambda_intensity,
            # '-μ': -mu,
            # '-k * log(1+θ)': -k * log1p_theta,
            # 'D+E': obj_tmp5,
            # 'D+E ideal': D_ideal,
            # ' ': np.array([]),
            'obj_tmp1': obj_tmp1, 'obj_tmp2': obj_tmp2, 'obj_tmp3': obj_tmp3,
            'obj_tmp4': obj_tmp4, 'obj_tmp5': obj_tmp5,
            'ag_np.log(μ)': ag_np.log(mu), 'ag_np.log1p(θ)': ag_np.log1p(theta),

            '   ': np.array([]),
            '(∑cij)lnμ -nxnyμ': (counts.data_grouped.sum(axis=0) * ag_np.log(mu)) - \
            (num_highres_per_lowres_bins * mu),
            'obj': (obj_tmp1 + obj_tmp2 + obj_tmp3) + obj_tmp4 + obj_tmp5,
            'obj ideal': A_ideal_mu + BpC_ideal_mu + D_ideal,
            '     ': np.array([]),
            '∑cij': counts.data_grouped.sum(axis=0)})

        print()
        print(type(counts.data_grouped), counts.data_grouped.dtype)

        exit(0)

    if ag_np.isnan(obj) or ag_np.isinf(obj):
        from topsy.utils.misc import printvars  # FIXME
        if counts.type == 'zero':
            printvars({
                'ε': epsilon, 'dis': dis, 'θ': theta, 'k': k, 'λ': lambda_intensity, 'μ': mu,
                'obj_tmp1': obj_tmp1})
        else:
            printvars({
                'ε': epsilon, 'dis': dis, 'θ': theta, 'k': k, 'λ': lambda_intensity, 'μ': mu,
                'ln(μ)': ag_np.log(mu),
                'ln(1+θ)': ag_np.log1p(theta),
                'tmp1': obj_tmp1, 'tmp2': obj_tmp2,
                'tmp3': obj_tmp3, 'tmp4': obj_tmp4,
                'tmp5': obj_tmp5})
        raise ValueError(
            f"Poisson component of objective function for {counts.name}"
            f" is {- obj}.")

    #return counts.weight * (- obj) # FIXME FIXME FIXME
    return counts.weight * (- obj - (counts.data_grouped.sum(axis=0) * ag_np.log(num_highres_per_lowres_bins)).sum()) # FIXME FIXME FIXME


def _poisson_obj_single(structures, counts, alpha, lengths, bias=None,
                        multiscale_factor=1, multiscale_variances=None,
                        epsilon=None, mixture_coefs=None):
    """Computes the Poisson objective function for a given counts matrix.
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

    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
    ploidy = int(structures[0].shape[0] / lengths_lowres.sum())

    if multiscale_variances is not None:
        if isinstance(multiscale_variances, np.ndarray):
            var_per_dis = multiscale_variances[
                counts.row3d] + multiscale_variances[counts.col3d]
        else:
            var_per_dis = multiscale_variances * 2
    else:
        var_per_dis = 0
    num_highres_per_lowres_bins = counts.count_fullres_per_lowres_bins(
        multiscale_factor)

    lambda_intensity = ag_np.zeros(counts.nnz_lowres)
    for struct, mix_coef in zip(structures, mixture_coefs):
        dis = ag_np.sqrt((ag_np.square(
            struct[counts.row3d] - struct[counts.col3d])).sum(axis=1))
        if multiscale_variances is None:
            tmp1 = ag_np.power(dis, alpha)
        else:
            tmp1 = ag_np.power(ag_np.square(dis) + var_per_dis, alpha / 2)
        tmp = tmp1.reshape(-1, counts.nnz_lowres).sum(axis=0)
        lambda_intensity = lambda_intensity + mix_coef * counts.bias_per_bin(
            bias, ploidy) * counts.beta * num_highres_per_lowres_bins * tmp

    # Sum main objective function
    if counts.type == 'zero':
        obj = lambda_intensity.sum()
    elif lambda_intensity.shape == counts.data.shape:  # TODO decide: elif epsilon is None:
        obj = lambda_intensity.sum() - (counts.data * ag_np.log(
            lambda_intensity)).sum()
        # obj0 = lambda_intensity.sum() - (counts.data * ag_np.log(
        #     lambda_intensity)).sum()
        # obj1 = lambda_intensity.sum() - (counts.data * ag_np.log(
        #     lambda_intensity / num_highres_per_lowres_bins)).sum()
        # obj2 = lambda_intensity.sum() - (counts.data * ag_np.log(
        #     lambda_intensity / num_highres_per_lowres_bins)).sum() - (
        #     counts.data * ag_np.log(
        #         num_highres_per_lowres_bins)).sum()
        # print(obj0); print(obj1); print(obj2); exit(0)
    else:
        obj = lambda_intensity.sum() - (
            counts.data_grouped * ag_np.log(lambda_intensity / num_highres_per_lowres_bins)).sum()

    #print('\n', obj, '\n'); exit(0) # TODO

    if ag_np.isnan(obj) or ag_np.isinf(obj):
        from topsy.utils.misc import printvars
        if type(dis).__name__ != 'ArrayBox':
            print(counts.name)
            printvars({
                'struct': structures[0], 'dis': dis,
                'lambda_intensity': lambda_intensity,
                'counts.data': counts.data})
        raise ValueError(
            f"Poisson component of objective function for {counts.name}"
            f" is {obj}.")

    return counts.weight * obj


def _obj_single(structures, counts, alpha, lengths, bias=None,
                multiscale_factor=1, multiscale_variances=None, epsilon=None,
                mixture_coefs=None):
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

    if epsilon is None or multiscale_factor == 1 or epsilon == 0:
        obj = _poisson_obj_single(
            structures=structures, counts=counts, alpha=alpha, lengths=lengths,
            bias=bias, multiscale_factor=multiscale_factor,
            multiscale_variances=multiscale_variances, epsilon=epsilon,
            mixture_coefs=mixture_coefs)
        return obj
    else:
        obj = _multiscale_reform_obj(
            structures=structures, epsilon=epsilon, counts=counts, alpha=alpha,
            lengths=lengths, bias=bias, multiscale_factor=multiscale_factor,
            mixture_coefs=mixture_coefs)
        return obj


def objective(X, counts, alpha, lengths, bias=None, constraints=None,
              reorienter=None, multiscale_factor=1, multiscale_variances=None,
              multiscale_reform=False, mixture_coefs=None, return_extras=False,
              inferring_alpha=False, epsilon=None):  # FIXME epsilon shouldn't be defined here unless inferring struct/eps separately
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
    multiscale_variances : float or array of float, optional
        For multiscale optimization at low resolution, the variances of each
        group of full-resolution beads corresponding to a single low-resolution
        bead.

    Returns
    -------
    obj : float
        The total negative log likelihood of the poisson model and constraints.
    """

    X, epsilon, mixture_coefs = _format_X(
        X, reorienter=reorienter,
        multiscale_reform=(multiscale_factor != 1 and multiscale_reform),
        mixture_coefs=mixture_coefs, epsilon=epsilon)  # FIXME epsilon

    # Optionally translate & rotate structures
    if reorienter is not None and reorienter.reorient:
        structures = reorienter.translate_and_rotate(X)
    else:
        structures = X

    # Check format of input
    counts = (counts if isinstance(counts, list) else [counts])
    if lengths is None:
        lengths = np.array([min([min(counts_maps.shape)
                                 for counts_maps in counts])])
    lengths = np.array(lengths)
    if bias is None:
        bias = np.ones((min([min(counts_maps.shape)
                             for counts_maps in counts]),))
    if not (isinstance(structures, list) or isinstance(structures, SequenceBox)):
        structures = [structures]
    if mixture_coefs is None:
        mixture_coefs = [1.] * len(structures)

    if constraints is None:
        obj_constraints = {}
    else:
        obj_constraints = constraints.apply(
            structures, alpha=alpha, inferring_alpha=inferring_alpha,
            mixture_coefs=mixture_coefs)
    obj_poisson = {}
    for counts_maps in counts:
        obj_poisson['obj_' + counts_maps.name] = _obj_single(
            structures=structures, counts=counts_maps, alpha=alpha,
            lengths=lengths, bias=bias, multiscale_factor=multiscale_factor,
            multiscale_variances=multiscale_variances, epsilon=epsilon,
            mixture_coefs=mixture_coefs)
    obj_poisson_sum = sum(obj_poisson.values())
    obj = obj_poisson_sum + sum(obj_constraints.values())

    if return_extras:
        obj_logs = {**obj_poisson, **obj_constraints,
                    **{'obj': obj, 'obj_poisson': obj_poisson_sum}}
        return obj, obj_logs, structures, alpha
    else:
        return obj


def _format_X(X, reorienter=None, multiscale_reform=False, mixture_coefs=None, epsilon=None):  # FIXME epsilon shouldn't be defined here unless inferring struct/eps separately
    """Reformat and check X.
    """

    if mixture_coefs is None:
        mixture_coefs = [1]

    if multiscale_reform and epsilon is None:  # FIXME epsilon
        epsilon = X[-1]
        X = X[:-1]
    else:
        #epsilon = None  # FIXME epsilon
        pass

    if reorienter is not None and reorienter.reorient:
        reorienter.check_X(X)
    else:
        try:
            X = X.reshape(-1, 3)
        except ValueError:
            raise ValueError(
                f"X should contain k 3D structures, X.shape = ({X.shape[0]},)")
        k = len(mixture_coefs)
        n = int(X.shape[0] / k)

        X = [X[i * n:(i + 1) * n] for i in range(k)]

    return X, epsilon, mixture_coefs


def objective_wrapper(X, counts, alpha, lengths, bias=None, constraints=None,
                      reorienter=None, multiscale_factor=1,
                      multiscale_variances=None, multiscale_reform=False,
                      mixture_coefs=None, callback=None):
    """Objective function wrapper to match scipy.optimize's interface.
    """

    '''X, epsilon, mixture_coefs = _format_X(
        X, reorienter=reorienter, multiscale_reform=multiscale_reform,
        mixture_coefs=mixture_coefs)
    X = X + [epsilon]'''

    new_obj, obj_logs, structures, alpha = objective(
        X, counts=counts, alpha=alpha, lengths=lengths, bias=bias,
        constraints=constraints, reorienter=reorienter,
        multiscale_factor=multiscale_factor,
        multiscale_variances=multiscale_variances,
        multiscale_reform=multiscale_reform,
        mixture_coefs=mixture_coefs, return_extras=True)

    if callback is not None:
        if multiscale_reform:
            epsilon = X[-1]
        else:
            epsilon = None
        callback.on_epoch_end(obj_logs, structures, alpha, X, epsilon=epsilon)

    if epsilon is not None:
        spacer = ' ' * (12 - len(f"{new_obj:.3g}"))
        if epsilon == 0:
            print(f'    obj {new_obj:.3g}{spacer}ε ---', flush=True)
        else:
            print(f'    obj {new_obj:.3g}{spacer}ε {epsilon:.6g}', flush=True)

    return new_obj


gradient = grad(objective)


def fprime_wrapper(X, counts, alpha, lengths, bias=None, constraints=None,
                   reorienter=None, multiscale_factor=1,
                   multiscale_variances=None, multiscale_reform=False,
                   mixture_coefs=None, callback=None):
    """Gradient function wrapper to match scipy.optimize's interface.
    """

    '''X, epsilon, mixture_coefs = _format_X(
        X, reorienter=reorienter, multiscale_reform=multiscale_reform,
        mixture_coefs=mixture_coefs)
    X = X + [epsilon]'''

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Using a non-tuple sequence for multidimensional"
            " indexing is deprecated", category=FutureWarning)
        new_grad = np.array(gradient(
            X, counts=counts, alpha=alpha, lengths=lengths, bias=bias,
            constraints=constraints, reorienter=reorienter,
            multiscale_factor=multiscale_factor,
            multiscale_variances=multiscale_variances,
            multiscale_reform=multiscale_reform,
            mixture_coefs=mixture_coefs)).flatten()
    if multiscale_reform and new_grad[-1] == 0:
        print(f"* * * * EPSILON GRADIENT IS 0 * * * *       (mean |other grad| = {np.mean(np.abs(new_grad[:-1]))})", flush=True)
    # elif multiscale_reform:
    #     print(f"*                 epsilon grad = {new_grad[-1]:.2g}   (mean |other grad| = {np.mean(np.abs(new_grad[:-1]))})", flush=True)

    return new_grad


def estimate_X(counts, init_X, alpha, lengths, bias=None, constraints=None,
               multiscale_factor=1, multiscale_variances=None,
               epsilon=None, epsilon_bounds=None, max_iter=30000, max_fun=None,
               factr=10000000., pgtol=1e-05, callback=None, alpha_loop=0, epsilon_loop=0,
               reorienter=None, mixture_coefs=None, verbose=True):
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

    multiscale_reform = (epsilon is not None)

    # Check format of input
    counts = (counts if isinstance(counts, list) else [counts])
    lengths = np.array(lengths)
    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
    if multiscale_reform:
        lengths_counts = lengths
    else:
        lengths_counts = lengths_lowres
    if bias is None:
        bias = np.ones((lengths_counts.sum(),))
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
            x0, counts=counts, alpha=alpha, lengths=lengths,
            bias=bias, constraints=constraints, reorienter=reorienter,
            multiscale_factor=multiscale_factor,
            multiscale_variances=multiscale_variances,
            multiscale_reform=multiscale_reform, mixture_coefs=mixture_coefs,
            callback=callback)
    else:
        obj = np.nan

    if multiscale_reform:
        bounds = np.append(
            np.full((init_X.flatten().shape[0], 2), None),
            np.array(epsilon_bounds).reshape(1, -1), axis=0)
    else:
        bounds = None

    if max_iter == 0:
        X = x0
        converged = True
    else:
        if max_fun is None:
            max_fun = max_iter
        results = optimize.fmin_l_bfgs_b(
            objective_wrapper,
            x0=x0,
            fprime=fprime_wrapper,
            iprint=0,
            maxiter=max_iter,
            maxfun=max_fun,
            pgtol=pgtol,
            factr=factr,
            bounds=bounds,
            args=(counts, alpha, lengths, bias, constraints,
                  reorienter, multiscale_factor, multiscale_variances,
                  multiscale_reform, mixture_coefs, callback))
        X, obj, d = results
        converged = d['warnflag'] == 0

    history = None
    if callback is not None:
        callback.on_training_end()
        history = callback.history

    if verbose:
        if multiscale_reform:
            print(f'INIT EPSILON: {epsilon:.3g},  FINAL EPSILON: {X[-1]:.3g}',
                  flush=True)
        if converged:
            print('CONVERGED\n', flush=True)
        else:
            print('OPTIMIZATION DID NOT CONVERGE', flush=True)
            print(d['task'].decode('utf8') + '\n', flush=True)

    return X, obj, converged, history


def _convergence_criteria(f_k, f_kplus1, factr=10000000.):
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
    constraints : Constraints instance, optional
        Object to compute constraints at each iteration.
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
                 epsilon_coord_descent=False, alpha_init=-3., max_alpha_loop=20, max_iter=30000,
                 factr=10000000., pgtol=1e-05, alpha_factr=1000000000000.,
                 reorienter=None, null=False, mixture_coefs=None, verbose=True):

        from .piecewise_whole_genome import ChromReorienter

        #print('%s\n%s 3D STRUCTURAL INFERENCE' %
        #      ('=' * 30, {2: 'DIPLOID', 1: 'HAPLOID'}[ploidy]), flush=True) # TODO removed

        lengths = np.array(lengths)

        if constraints is None:
            constraints = Constraints(
                counts=counts, lengths=lengths, ploidy=ploidy,
                multiscale_factor=multiscale_factor,
                multiscale_reform=(epsilon is not None))
        if callback is None:
            callback = Callback(
                lengths=lengths, ploidy=ploidy, counts=counts,
                multiscale_factor=multiscale_factor,
                multiscale_reform=(epsilon is not None),
                frequency={'print': 100, 'history': 100, 'save': None})
        if reorienter is None:
            reorienter = ChromReorienter(lengths=lengths, ploidy=ploidy)
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
        self.multiscale_reform = (epsilon is not None)
        self.epsilon = epsilon
        self.epsilon_bounds = epsilon_bounds
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

        # FIXME this is obviously temporary...
        self.max_epsilon_loop = max_alpha_loop
        self.epsilon_factr = alpha_factr

        # TODO update this on the main branch too...?
        self.X_ = self.init_X
        if self.alpha is not None:
            self.alpha_ = self.alpha
        else:
            self.alpha_ = self.alpha_init
        self.beta_ = [c.beta for c in self.counts if c.sum() != 0]
        self.epsilon_ = self.epsilon
        self.obj_ = None
        self.alpha_obj_ = None  # TODO update this on the main branch too
        self.epsilon_obj_ = None
        self.converged_ = None
        self.history_ = None
        self.struct_ = None
        self.orientation_ = None

    def _infer_beta(self, update_counts=True, verbose=True):
        """Estimate beta, given current structure and alpha.
        """

        from .estimate_alpha_beta import _estimate_beta

        new_beta = _estimate_beta(
            self.X_.flatten(), self.counts, alpha=self.alpha_,
            lengths=self.lengths, bias=self.bias,
            multiscale_factor=self.multiscale_factor,
            multiscale_variances=self.multiscale_variances,
            epsilon=self.epsilon_,
            mixture_coefs=self.mixture_coefs,
            reorienter=self.reorienter, verbose=verbose)
        if update_counts:
            self.counts = _update_betas_in_counts_matrices(
                counts=self.counts, beta=new_beta)
        return list(new_beta.values())

    def _fit_structure(self, alpha_loop=0):
        """Fit structure to counts data, given current alpha.
        """

        self.X_, self.obj_, self.converged_, self.history_ = estimate_X(  # TODO check self.history_
            counts=self.counts,
            init_X=self.X_.flatten(),
            alpha=self.alpha_,
            lengths=self.lengths,
            bias=self.bias,
            constraints=self.constraints,
            multiscale_factor=self.multiscale_factor,
            multiscale_variances=self.multiscale_variances,
            epsilon=self.epsilon_,
            epsilon_bounds=self.epsilon_bounds,
            max_iter=self.max_iter,
            factr=self.factr,
            pgtol=self.pgtol,
            callback=self.callback,
            alpha_loop=alpha_loop,
            reorienter=self.reorienter,
            mixture_coefs=self.mixture_coefs,
            verbose=self.verbose)

        # if len(history_) > 1:
        #     if self.history_ is None:
        #         self.history_ = history_
        #     else:
        #         for k, v in history_.items():
        #             self.history_[k].extend(v)

    def _fit_alpha(self, alpha_loop=0):
        """Fit alpha to counts data, given current structure.
        """

        from .estimate_alpha_beta import estimate_alpha

        self.alpha_, self.alpha_obj_, self.converged_, self.history_ = estimate_alpha(  # TODO check self.history_
            counts=self.counts,
            X=self.X_.flatten(),
            alpha_init=self.alpha_,
            lengths=self.lengths,
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
            verbose=self.verbose)

        # if len(history_) > 1:
        #     if self.history_ is None:
        #         self.history_ = history_
        #     else:
        #         for k, v in history_.items():
        #             self.history_[k].extend(v)

    def _fit_epsilon(self, inferring_epsilon, alpha_loop=0, epsilon_loop=0):
        """Fit structure/epsilon to counts, given current structure/epsilon.
        """
        from .estimate_epsilon import estimate_epsilon

        if inferring_epsilon:
            init_X = self.epsilon_
            epsilon = None
            structures = self.X_.flatten()
        else:
            init_X = self.X_.flatten()
            epsilon = self.epsilon_
            structures = None

        new_X_, self.epsilon_obj_, self.converged_, self.history_ = estimate_epsilon(  # TODO check self.history_
            counts=self.counts,
            init_X=init_X,
            alpha=self.alpha_,
            lengths=self.lengths,
            bias=self.bias,
            constraints=self.constraints,
            multiscale_factor=self.multiscale_factor,
            multiscale_variances=self.multiscale_variances,
            epsilon=epsilon,
            structures=structures,
            epsilon_bounds=self.epsilon_bounds,
            max_iter=self.max_iter,
            factr=self.factr,
            pgtol=self.pgtol,
            callback=self.callback,
            alpha_loop=alpha_loop,
            epsilon_loop=epsilon_loop,
            reorienter=self.reorienter,
            mixture_coefs=self.mixture_coefs,
            verbose=self.verbose)

        if inferring_epsilon:
            self.epsilon_ = new_X_[0]
        else:
            self.X_ = new_X_

        # if len(history_) > 1:
        #     if self.history_ is None:
        #         self.history_ = history_
        #     else:
        #         for k, v in history_.items():
        #             self.history_[k].extend(v)

    def _fit_naive_multiscale(self, alpha_loop=0):
        """Fit structure to counts data, given current alpha. TODO
        """

        if self.multiscale_factor == 1:
            return False

        if self.epsilon is None and (self.multiscale_variances is None or self.multiscale_variances == 0):
            return False

        if not self.multiscale_reform:
            return False

        _print_code_header(
            "Inferring with naive multiscale", max_length=50, blank_lines=1)

        self.X_, self.obj_, self.converged_, self.history_ = estimate_X(  # TODO check self.history_
            counts=self.counts,
            init_X=self.X_.flatten(),
            alpha=self.alpha_,
            lengths=self.lengths,
            bias=self.bias,
            constraints=self.constraints,
            multiscale_factor=self.multiscale_factor,
            multiscale_variances=None,
            epsilon=None,
            epsilon_bounds=None,
            max_iter=self.max_iter,
            factr=self.factr,
            pgtol=self.pgtol,
            callback=self.callback,
            alpha_loop=alpha_loop,
            epsilon_loop=-1,
            reorienter=self.reorienter,
            mixture_coefs=self.mixture_coefs,
            verbose=self.verbose)

        return True

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
            time_current = str(
                timedelta(seconds=round(timer() - time_start)))
            _print_code_header([
                f"Jointly inferring structure & alpha",
                f"Inferring ALPHA #{alpha_loop},"
                f" total time={time_current}"], max_length=50)
            self._fit_alpha(alpha_loop=alpha_loop)
            self.beta_ = self._infer_beta()
            if not self.converged_:
                break
            time_current = str(
                timedelta(seconds=round(timer() - time_start)))
            _print_code_header([
                f"Jointly inferring structure & alpha",
                f"Inferring STRUCTURE #{alpha_loop},"
                f" total time={time_current}"], max_length=50)
            self._fit_structure(alpha_loop=alpha_loop)
            if not self.converged_:
                break
            if _convergence_criteria(
                    f_k=prev_alpha_obj, f_kplus1=self.alpha_obj_,
                    factr=self.alpha_factr):
                break
            prev_alpha_obj = self.alpha_obj_

    def _fit_struct_epsilon_jointly(self, time_start, alpha_loop=0,
                                    infer_structure_first=True):
        """Jointly fit structure & epsilon to counts data.
        """

        # if self.multiscale_reform and self.verbose:
        #     print(f"Epsilon init = {self.epsilon:.3g}, bounds = ["
        #           f"{self.epsilon_bounds[0]:.3g},"
        #           f" {self.epsilon_bounds[1]:.3g}]", flush=True)

        only_infer_epsilon_once = True
        if only_infer_epsilon_once:
            self._fit_epsilon(
                inferring_epsilon=True, alpha_loop=alpha_loop,
                epsilon_loop=1)
            return

        fit_naive_multiscale = False
        #fit_naive_multiscale = self._fit_naive_multiscale()

        if not self.multiscale_reform or not self.epsilon_coord_descent:
            self._fit_structure()
            return

        if infer_structure_first and not fit_naive_multiscale:
            _print_code_header([
                "Jointly inferring structure & epsilon",
                f"Inferring STRUCTURE #0, initial epsilon={self.epsilon:.3g}"],
                max_length=50)
            self._fit_epsilon(
                inferring_epsilon=False, alpha_loop=alpha_loop, epsilon_loop=0)
            if not self.converged_:
                return

        prev_epsilon_obj = None
        for epsilon_loop in range(1, self.max_epsilon_loop + 1):
            time_current = str(
                timedelta(seconds=round(timer() - time_start)))
            _print_code_header([
                f"Jointly inferring structure & epsilon",
                f"Inferring EPSILON #{epsilon_loop},"
                f" total time={time_current}"], max_length=50)
            self._fit_epsilon(
                inferring_epsilon=True, alpha_loop=alpha_loop,
                epsilon_loop=epsilon_loop)
            if not self.converged_:
                break
            time_current = str(
                timedelta(seconds=round(timer() - time_start)))
            _print_code_header([
                f"Jointly inferring structure & epsilon",
                f"Inferring STRUCTURE #{epsilon_loop},"
                f" total time={time_current}"], max_length=50)
            self._fit_epsilon(
                inferring_epsilon=False, alpha_loop=alpha_loop,
                epsilon_loop=epsilon_loop)
            if not self.converged_:
                break
            if _convergence_criteria(
                    f_k=prev_epsilon_obj, f_kplus1=self.epsilon_obj_,
                    factr=self.epsilon_factr):
                break
            prev_epsilon_obj = self.epsilon_obj_

    def fit(self):
        """Fit structure to counts data, optionally estimate alpha & epsilon.

        Returns
        -------
        self : returns an instance of self.
        """

        if self.null:
            print('GENERATING NULL STRUCTURE', flush=True)
            # Dummy counts need to be inputted because we need to know which
            # row/col to include in calculations of constraints
            self.counts = [NullCountsMatrix(
                counts=self.counts, lengths=self.lengths, ploidy=self.ploidy,
                multiscale_factor=self.multiscale_factor,
                multiscale_reform=self.multiscale_reform)]

        if any([b is None for b in self.beta_]):
            if all([b is None for b in self.beta_]):
                self.beta_ = self._infer_beta()
            else:
                raise ValueError("Some but not all values in beta are None.")

        # Infer structure
        self.history_ = None
        time_start = timer()
        #self._fit_naive_multiscale()
        self._fit_struct_epsilon_jointly(time_start)
        #self._fit_struct_alpha_jointly(time_start)  # FIXME duh, temporary
        time_current = str(timedelta(seconds=round(timer() - time_start)))
        print("OPTIMIZATION AT %dX RESOLUTION COMPLETE, TOTAL ELAPSED TIME=%s" %
              (self.multiscale_factor, time_current), flush=True)

        if self.multiscale_reform and not self.epsilon_coord_descent:
            self.epsilon_ = self.X_[-1]
            X_ = self.X_[:-1]
        else:
            X_ = self.X_

        if self.reorienter.reorient:
            self.orientation_ = X_
            self.struct_ = self.reorienter.translate_and_rotate(X_)[
                0].reshape(-1, 3)
        else:
            self.struct_ = X_.reshape(-1, 3)

        return self
