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
from autograd.scipy.special import gammaln as ag_gammaln
from .multiscale_optimization import decrease_lengths_res
from .multiscale_optimization import _repeat_struct_multiscale
from .counts import _update_betas_in_counts_matrices, NullCountsMatrix
from .constraints import Constraints
from .callbacks import Callback


def printvars(variables, print_array=False):
    import pandas as pd
    out_arr = []
    out_const = []
    for name, var in variables.items():
        if type(var).__name__ == 'ArrayBox':
            var = var._value
        if isinstance(var, np.ndarray):
            var_size = var.flatten().shape[0]
            if var_size == 0:
                out_arr.append({
                    'Name': name, 'Empty': True,
                    'Shape': f"({','.join(map(str, var.shape))})"})
            else:
                perc0 = (var == 0).sum() / var_size * 100
                percNaN = np.isnan(var).sum() / var_size * 100
                percInf = np.isinf(var).sum() / var_size * 100
                out_arr.append({
                    'Name': name, 'Shape': f"({','.join(map(str, var.shape))})",
                    'Mean': f"{var.mean():.3g}", 'Med': f"{np.median(var):.3g}",
                    'Sum': f"{var.sum():.3g}",
                    'Max': f"{var.max():.3g}", 'Min': f"{var.min():.3g}",
                    '%0': f"{perc0:.3g}", '%nan': f"{percNaN:.3g}",
                    '%inf': f"{percInf:.3g}", 'Empty': False})
                if print_array:
                    print(name)
                    with np.printoptions(formatter={'float': '{: 0.3g}'.format}):
                        print(var)
        else:
            out_const.append({'Name': name, 'Const': var})
    df_const = pd.DataFrame(out_const)
    df_arr = pd.DataFrame(out_arr)
    cols_replace_nan = [c for c in df_arr.columns if c not in ('Name', 'Shape')]
    df_arr.loc[df_arr.Empty.values,
               cols_replace_nan] = df_arr.loc[df_arr.Empty.values,
                                              cols_replace_nan].fillna('.')
    df_arr.drop('Empty', axis=1, inplace=True)
    print(df_const.to_string(index=False, header=False))
    print(df_arr.to_string(index=False))


def _pois_sum(arr, nnz):
    """TODO
    """

    return arr.reshape(-1, nnz).sum(axis=0)  # ***************


def _multiscale_reform_obj(structures, epsilon, counts, alpha, lengths,
                           bias=None, multiscale_factor=1, mixture_coefs=None):
    """TODO
    """

    lengths_lowres = decrease_lengths_res(lengths, multiscale_factor)
    ploidy = int(structures[0].shape[0] / lengths_lowres.sum())

    num_highres_per_lowres_bins = counts.count_fullres_per_lowres_bins(
        multiscale_factor)

    bias_per_bin = counts.bias_per_bin(bias, ploidy)  # TODO

    epsilon_sq = ag_np.square(epsilon)
    alpha_sq = ag_np.square(alpha)

    taylor = False

    theta = ag_np.zeros(counts.nnz_lowres)
    k = ag_np.zeros(counts.nnz_lowres)
    for struct, mix_coef in zip(structures, mixture_coefs):
        dis = ag_np.sqrt((ag_np.square(
            struct[counts.row3d] - struct[counts.col3d])).sum(axis=1))
        dis_sq = ag_np.square(dis)
        dis_alpha = ag_np.power(dis, alpha)
        if ploidy == 1:
            theta_tmp1 = alpha_sq * epsilon_sq * dis_alpha / dis_sq
            theta_tmp2 = 1 - 1.5 * alpha * epsilon_sq / dis_sq
            #theta_tmp2 = 1 / (1 + 1.5 * alpha * epsilon_sq / dis_sq)
            k_tmp1 = dis_sq / alpha_sq / epsilon_sq
            if taylor: k_tmp2 = 1 + 3 * alpha * epsilon_sq / dis_sq
            else: k_tmp2 = 1 + 1.5 * alpha * epsilon_sq / dis_sq
        else:
            raise ValueError("lol.")
            sum_dis_2alpha_min2 = _pois_sum(
                ag_np.power(dis, 2 * alpha - 2), counts.nnz_lowres)
            sum_dis_alpha = _pois_sum(dis_alpha, counts.nnz_lowres)
            sum_dis_alpha_min2 = _pois_sum(
                dis_alpha / dis_sq, counts.nnz_lowres)
            theta_tmp1 = alpha_sq * epsilon_sq * sum_dis_2alpha_min2 / sum_dis_alpha
            theta_tmp2 = 1 - 1.5 * alpha * epsilon_sq * sum_dis_alpha_min2 / sum_dis_alpha
            k_tmp1 = ag_np.square(sum_dis_alpha) / alpha_sq / epsilon_sq / sum_dis_2alpha_min2
            if taylor: k_tmp2 = 1 + 3 * alpha * epsilon_sq * sum_dis_alpha_min2 / sum_dis_alpha
            else: k_tmp2 = 1 + 1.5 * alpha * epsilon_sq * sum_dis_alpha_min2 / sum_dis_alpha
        theta = theta + mix_coef * counts.beta * theta_tmp1 * theta_tmp2
        if taylor: k = k + mix_coef * k_tmp1 * k_tmp2
        else: k = k + mix_coef * k_tmp1 * ag_np.square(k_tmp2)

    theta = theta.reshape(1, -1)
    k = k.reshape(1, -1)
    num_highres_per_lowres_bins = num_highres_per_lowres_bins.reshape(1, -1)

    alternative_obj = 'tryo'
    huge_k_over_c = ag_np.array([0])

    if counts.type == 'zero':
        if alternative_obj == 'try1':
            obj_tmp2a = num_highres_per_lowres_bins * k * (ag_np.log(theta / (1 - theta)))
        elif alternative_obj == 'try2':
            obj_tmp2a = num_highres_per_lowres_bins * k * (ag_np.log(theta) - ag_np.log(1 + theta))
        else:
            obj_tmp2a = num_highres_per_lowres_bins * k * ag_np.log(1 + theta)
        obj = ag_np.sum(obj_tmp2a)
        obj_2b_3 = 0
    elif alternative_obj == 'try1':
        # TRY 1
        obj_tmp1 = ag_np.sum(counts.data_grouped * (ag_np.log(1 - theta / (1 - theta))), axis=0)
        obj_tmp2a = num_highres_per_lowres_bins * k * (ag_np.log(theta / (1 - theta)))
        obj_tmp2b = - num_highres_per_lowres_bins * ag_gammaln(k)
        obj_tmp3 = ag_np.sum(ag_gammaln(counts.data_grouped + k), axis=0)
        obj = ag_np.sum(obj_tmp1) + ag_np.sum(obj_tmp2a) + ag_np.sum(obj_tmp2b + obj_tmp3)
        obj_2b_3 = obj_tmp2b + obj_tmp3
    elif alternative_obj == 'try2':
        obj_tmp1 = - ag_np.sum(counts.data_grouped * ag_np.log(1 + theta), axis=0).reshape(1, -1)
        obj_tmp2a = num_highres_per_lowres_bins * k * (ag_np.log(theta) - ag_np.log(1 + theta))
        obj_tmp2b = - num_highres_per_lowres_bins * ag_gammaln(k)
        obj_tmp3 = ag_np.sum(ag_gammaln(counts.data_grouped + k), axis=0).reshape(1, -1)

        try_new_approach = False
        if ag_np.min(dis) > ag_np.inf:
            obj_2b_3 = ag_np.sum(counts.data_grouped * ag_np.log(k), axis=0).reshape(1, -1)
            obj_2b_3_sum = ag_np.sum(obj_2b_3)
            obj_2b_3_sum_bad = (obj_tmp2b + obj_tmp3).sum()
            if ag_np.min(dis) > 1e3 and obj_2b_3_sum_bad > obj_2b_3_sum:
                print('-' * 60)
                printvars({'data_grouped': counts.data_grouped, 'dis': dis, 'k': k})
                print('2b vs 3', (-obj_tmp2b).sum(), obj_tmp3.sum())
                print('sum bad', obj_2b_3_sum_bad)
                print('sum good', obj_2b_3_sum)
                exit(0)
        elif not try_new_approach:
            obj_2b_3 = obj_tmp2b + obj_tmp3
            obj_2b_3_sum = ag_np.sum(obj_2b_3)
        else:
            k_over_c_cutoff = 1e10
            k_over_c = k.flatten() / counts.data_grouped.min(axis=0)
            huge_k_over_c = k_over_c > k_over_c_cutoff
            obj_2b_3_part1 = obj_tmp2b.flatten()[~huge_k_over_c] + obj_tmp3[~huge_k_over_c]
            obj_2b_3_sum = ag_np.sum(obj_2b_3_part1)
            if ag_np.sum(huge_k_over_c) != 0:
                obj_2b_3_part2 = ag_np.sum(counts.data_grouped[:, huge_k_over_c] * ag_np.log(k[0, huge_k_over_c]), axis=0)
                obj_2b_3_sum = obj_2b_3_sum + ag_np.sum(obj_2b_3_part2)
                obj_2b_3 = np.concatenate([obj_2b_3_part1, obj_2b_3_part2]).reshape(1, -1)
            else:
                obj_2b_3 = obj_2b_3_part1.reshape(1, -1)

        obj = ag_np.sum(obj_tmp1) + ag_np.sum(obj_tmp2a) + obj_2b_3_sum
    else:
        obj_tmp1 = ag_np.sum(counts.data_grouped * ag_np.log(theta / (1 + theta)), axis=0)
        obj_tmp2a = - num_highres_per_lowres_bins * k * ag_np.log(1 + theta)
        obj_tmp2b = - num_highres_per_lowres_bins * ag_gammaln(k)
        obj_tmp3 = ag_np.sum(ag_gammaln(counts.data_grouped + k), axis=0)
        obj = ag_np.sum(obj_tmp1) + ag_np.sum(obj_tmp2a) + ag_np.sum(obj_tmp2b + obj_tmp3)
        obj_2b_3 = obj_tmp2b + obj_tmp3

    if type(obj).__name__ != 'ArrayBox':
        print('=' * 75)
        #kneg1i = (k == -1).flatten(); print(k_tmp1[kneg1i]); print(k_tmp2[kneg1i])
        printvars({
            'dis': dis, 'data_grouped': counts.data_grouped, 'k': k,
            'theta': theta, 'obj_tmp1': obj_tmp1, 'obj_tmp2a': obj_tmp2a,
            'obj_2b_3': obj_2b_3, 'epsilon': epsilon, 'obj': obj,
            'huge_k_over_c': ag_np.sum(huge_k_over_c)})
        #if dis.min() > 1e20: print('jfc.'); exit(0)
        dont_stop_believin = True
        perc0_2b_3 = ag_np.sum(obj_2b_3 == 0) / max(obj_2b_3.shape) * 100
        if perc0_2b_3 > 0 and not dont_stop_believin:
            is0i_2b_3 = np.where(obj_2b_3 == 0)[1]
            non0i_2b_3 = np.where(obj_2b_3 != 0)[1]
            got0_min = counts.data_grouped.min(axis=0)[is0i_2b_3]
            non0_min = counts.data_grouped.min(axis=0)[non0i_2b_3]
            got0_sum = counts.data_grouped.sum(axis=0)[is0i_2b_3]
            non0_sum = counts.data_grouped.sum(axis=0)[non0i_2b_3]
            printvars({
                'got0_min': got0_min, 'non0_min': non0_min,
                'got0_sum': got0_sum, 'non0_sum': non0_sum,
                'got0_c': counts.data_grouped.T[is0i_2b_3], 'non0_c': counts.data_grouped.T[non0i_2b_3],
                'got0_k': k.T[is0i_2b_3], 'non0_k': k.T[non0i_2b_3]})
            exit(0)

    if ag_np.isnan(obj) or ag_np.isinf(obj):
        print(f"\nEpsilon: {epsilon:.3g}")
        printvars({
            'obj_tmp1': obj_tmp1,
            'obj_tmp2a': obj_tmp2a,
            'obj_tmp2b': obj_tmp2b,
            'obj_tmp3': obj_tmp3,
            'k': k,
            'data_grouped': counts.data_grouped,
            'data_grouped + k': counts.data_grouped + k,
            'Γ(data_grouped + k)': ag_gammaln(counts.data_grouped + k),
            'Γ(k)': ag_gammaln(k),
            'log Γ(k)': ag_gammaln(k),
            'epsilon': epsilon,
        })
        raise ValueError(f"Poisson component of objective function is {obj}.")
    if np.isnan(counts.weight) or np.isinf(counts.weight) or counts.weight == 0:
        raise ValueError(f"Counts weight may not be {counts.weight}.")

    return counts.weight * (- obj)


def _poisson_obj_single(structures, counts, alpha, lengths, bias=None,
                        multiscale_factor=1, multiscale_variances=None,
                        epsilon=None, mixture_coefs=None):
    """Computes the poisson objective function for each counts matrix.
    """

    if (bias is not None and bias.sum() == 0) or counts.nnz == 0 or counts.null:
        return 0.

    if mixture_coefs is not None and len(structures) != len(mixture_coefs):
        raise ValueError("The number of structures (%d) and of mixture"
                         " coefficents (%d) should be identical." %
                         (len(structures), len(mixture_coefs)))
    elif mixture_coefs is None:
        mixture_coefs = [1.]

    if epsilon is not None and epsilon != 0 and multiscale_factor != 1:
        return _multiscale_reform_obj(
            structures=structures, epsilon=epsilon, counts=counts, alpha=alpha,
            lengths=lengths, bias=bias, multiscale_factor=multiscale_factor,
            mixture_coefs=mixture_coefs)

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
        bar = ag_np.square(struct[counts.row3d] - struct[counts.col3d])
        foo = (bar).sum(axis=1)
        #print(struct.shape)
        #print(type(struct), struct.dtype, type(struct[0, 0]))
        #print(type(foo), foo.dtype, type(foo[0]))
        #print(bar.shape, foo.shape)
        #print((foo ** 0.5).shape)
        #print(np.sqrt(foo).shape)
        #print(ag_np.sqrt(foo).shape)
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
    if epsilon is None:
        if counts.type == 'zero':
            obj = lambda_intensity.sum()
        else:
            obj = lambda_intensity.sum() - (counts.data * ag_np.log(
                lambda_intensity)).sum()
    else:
        obj = lambda_intensity.sum() - (counts.data_grouped * ag_np.log(
            lambda_intensity)).sum()

    if ag_np.isnan(obj) or ag_np.isinf(obj):
        raise ValueError(f"Poisson component of objective function is {obj}.")
    if np.isnan(counts.weight) or np.isinf(counts.weight) or counts.weight == 0:
        raise ValueError(f"Counts weight may not be {counts.weight}.")

    return counts.weight * obj


def objective(X, counts, alpha, lengths, bias=None, constraints=None,
              reorienter=None, multiscale_factor=1, multiscale_variances=None,
              multiscale_reform=False, mixture_coefs=None, return_extras=False,
              inferring_alpha=False):
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

    '''if multiscale_reform:
        epsilon = X[-1]
        X = X[:-1]
    else:
        epsilon = None'''

    X, epsilon, mixture_coefs = _format_X(
        X, reorienter=reorienter,
        multiscale_reform=(multiscale_factor != 1 and multiscale_reform),
        mixture_coefs=mixture_coefs)

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
        obj_poisson['obj_' + counts_maps.name] = _poisson_obj_single(
            structures, counts_maps, alpha=alpha, lengths=lengths, bias=bias,
            multiscale_factor=multiscale_factor,
            multiscale_variances=multiscale_variances, epsilon=epsilon,
            mixture_coefs=mixture_coefs)
    obj_poisson_sum = sum(obj_poisson.values())
    obj = obj_poisson_sum + sum(obj_constraints.values())
    #if isinstance(obj, float) and epsilon is not None:  # FIXME
    #    print(f"{obj:.3g}\t\t{epsilon:.3g}")

    if return_extras:
        obj_logs = {**obj_poisson, **obj_constraints, **{'obj': obj, 'obj_poisson': obj_poisson_sum}}
        return obj, obj_logs, structures, alpha
    else:
        return obj


def _format_X(X, reorienter=None, multiscale_reform=False, mixture_coefs=None):
    """Reformat and check X.
    """

    if mixture_coefs is None:
        mixture_coefs = [1]

    #print('a', X.dtype)

    if multiscale_reform:
        epsilon = X[-1]
        X = X[:-1]
    else:
        epsilon = None

    #print('b', X.dtype)

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

    #print('c', X[0].dtype)

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

    return new_grad


def estimate_X(counts, init_X, alpha, lengths, bias=None, constraints=None,
               multiscale_factor=1, multiscale_variances=None,
               epsilon=None, epsilon_bounds=None, max_iter=30000, max_fun=None,
               factr=10000000., pgtol=1e-05, callback=None, alpha_loop=None,
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
        print('=' * 30, flush=True)
        print("\nRUNNING THE L-BFGS-B CODE\n\n           * * *\n\nMachine"
              " precision = %.4g\n" % np.finfo(np.float).eps, flush=True)

    if callback is not None:
        if reorienter is not None and reorienter.reorient:
            opt_type = 'chrom_reorient'
        else:
            opt_type = 'structure'
        callback.on_training_begin(opt_type=opt_type, alpha_loop=alpha_loop)
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
                 alpha_init=-3., max_alpha_loop=20, max_iter=30000,
                 factr=10000000., pgtol=1e-05, alpha_factr=1000000000000.,
                 reorienter=None, null=False, mixture_coefs=None, verbose=True):

        from .piecewise_whole_genome import ChromReorienter

        print('%s\n%s 3D STRUCTURAL INFERENCE' %
              ('=' * 30, {2: 'DIPLOID', 1: 'HAPLOID'}[ploidy]), flush=True)

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

        self.X_ = None
        self.alpha_ = None
        self.beta_ = None
        self.obj_ = None
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
            epsilon=self.epsilon,
            mixture_coefs=self.mixture_coefs,
            reorienter=self.reorienter, verbose=verbose)
        if update_counts:
            self.counts = _update_betas_in_counts_matrices(
                counts=self.counts, beta=new_beta)
        return list(new_beta.values())

    def _fit_structure(self, alpha_loop=None):
        """Fit structure to counts data, given current alpha.
        """

        self.X_, self.obj_, self.converged_, history_ = estimate_X(
            counts=self.counts,
            init_X=self.X_.flatten(),
            alpha=self.alpha_,
            lengths=self.lengths,
            bias=self.bias,
            constraints=self.constraints,
            multiscale_factor=self.multiscale_factor,
            multiscale_variances=self.multiscale_variances,
            epsilon=self.epsilon,
            epsilon_bounds=self.epsilon_bounds,
            max_iter=self.max_iter,
            factr=self.factr,
            pgtol=self.pgtol,
            callback=self.callback,
            alpha_loop=alpha_loop,
            reorienter=self.reorienter,
            mixture_coefs=self.mixture_coefs,
            verbose=self.verbose)

        if len(history_) > 1:
            if self.history_ is None:
                self.history_ = history_
            else:
                for k, v in history_.items():
                    self.history_[k].extend(v)

    def _fit_alpha(self, alpha_loop=None):
        """Fit alpha to counts data, given current structure.
        """

        from .estimate_alpha_beta import estimate_alpha

        self.alpha_, self.alpha_obj_, self.converged_, history_ = estimate_alpha(
            counts=self.counts,
            X=self.X_.flatten(),
            alpha_init=self.alpha_,
            lengths=self.lengths,
            bias=self.bias,
            constraints=self.constraints,
            multiscale_factor=self.multiscale_factor,
            multiscale_variances=self.multiscale_variances,
            epsilon=self.epsilon,
            random_state=None,
            max_iter=self.max_iter,
            factr=self.factr,
            pgtol=self.pgtol,
            callback=self.callback,
            alpha_loop=alpha_loop,
            reorienter=self.reorienter,
            mixture_coefs=self.mixture_coefs,
            verbose=self.verbose)

        if len(history_) > 1:
            if self.history_ is None:
                self.history_ = history_
            else:
                for k, v in history_.items():
                    self.history_[k].extend(v)

    def fit(self):
        """Fit structure to counts data, optionally estimate alpha.

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

        self.X_ = self.init_X
        if self.alpha is not None:
            self.alpha_ = self.alpha
        else:
            self.alpha_ = self.alpha_init
        self.beta_ = [c.beta for c in self.counts if c.sum() != 0]
        if any([b is None for b in self.beta_]):
            if all([b is None for b in self.beta_]):
                self.beta_ = self._infer_beta()
            else:
                raise ValueError("Some but not all values in beta are None.")

        # Infer structure
        self.history_ = None
        time_start = timer()
        if self.alpha is not None:
            self._fit_structure()
        else:
            print("JOINTLY INFERRING STRUCTURE + ALPHA: inferring structure,"
                  " with initial guess of alpha=%.3g"
                  % self.alpha_init, flush=True)
            self._fit_structure(alpha_loop=0)
            prev_alpha_obj = None
            if self.converged_:
                for alpha_loop in range(1, self.max_alpha_loop + 1):
                    time_current = str(
                        timedelta(seconds=round(timer() - time_start)))
                    print("JOINTLY INFERRING STRUCTURE + ALPHA (#%d):"
                          " inferring alpha, total elapsed time=%s" %
                          (alpha_loop, time_current), flush=True)
                    self._fit_alpha(alpha_loop=alpha_loop)
                    self.beta_ = self._infer_beta()
                    if not self.converged_:
                        break
                    time_current = str(
                        timedelta(seconds=round(timer() - time_start)))
                    print("JOINTLY INFERRING STRUCTURE + ALPHA (#%d): inferring"
                          " structure, total elapsed time=%s" %
                          (alpha_loop, time_current), flush=True)
                    self._fit_structure(alpha_loop=alpha_loop)
                    if not self.converged_:
                        break
                    if _convergence_criteria(
                            f_k=prev_alpha_obj, f_kplus1=self.alpha_obj_,
                            factr=self.alpha_factr):
                        break
                    prev_alpha_obj = self.alpha_obj_
        time_current = str(timedelta(seconds=round(timer() - time_start)))
        print("OPTIMIZATION AT %dX RESOLUTION COMPLETE, TOTAL ELAPSED TIME=%s" %
              (self.multiscale_factor, time_current), flush=True)

        if self.multiscale_reform:
            self.epsilon_ = self.X_[-1]
            X_ = self.X_[:-1]
        else:
            self.epsilon_ = None
            X_ = self.X_

        if self.reorienter.reorient:
            self.orientation_ = X_
            self.struct_ = self.reorienter.translate_and_rotate(X_)[
                0].reshape(-1, 3)
        else:
            self.struct_ = X_.reshape(-1, 3)

        return self
