import sys
import numpy as np

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

from absl import logging as absl_logging
absl_logging.set_verbosity('error')
from jax.config import config as jax_config
jax_config.update("jax_platform_name", "cpu")
jax_config.update("jax_enable_x64", True)
from jax import custom_jvp, lax
import jax.numpy as ag_np
from jax.nn import relu
from jax.scipy.special import gammaln

from tensorflow_probability.substrates import jax as tfp
from typing import Any
from scipy.special import iv, ivp
from .utils_poisson import jax_max


def _polyval(x, coefs):  # TODO maybe move to utils_poisson.py
    """Analagous to np.polyval (which is not differentiable with jax)"""
    # TODO are we sure jax.numpy.polyval still isn't differentiable?
    ans = 0
    power = len(coefs) - 1
    for coef in coefs:
        ans = ans + coef * ag_np.power(x, power)
        power = power - 1
    return ans


def _stirling(z):
    """Computes B_N(z)
    """
    sterling_coefs = [
        8.11614167470508450300E-4, -5.95061904284301438324E-4,
        7.93650340457716943945E-4, -2.77777777730099687205E-3,
        8.33333333333331927722E-2]
    z_sq_inv = 1. / (z * z)
    return _polyval(z_sq_inv, coefs=sterling_coefs) / z


def _masksum(x, mask=None, axis=None):
    """Sum of masked array (for jax)"""
    if mask is None:
        return ag_np.sum(x, axis=axis)
    else:
        return ag_np.sum(ag_np.where(mask, x, 0), axis=axis)


def relu_min(x1, x2):
    # TODO this is temporary, remove this and switch to jax_min
    # returns min(x1, x2)
    return - (relu((-x1) - (-x2)) + (-x2))


def gamma_poisson_nll(theta, k, data, num_fullres_per_lowres_bins, bias=None,
                      mask=None, mods=[]):
    if num_fullres_per_lowres_bins is None:  # TODO really?
        num_fullres_per_lowres_bins = data.shape[1]

    if not (bias is None or np.all(bias == 1)):
        raise NotImplementedError("aaaaa.")

    log1p_theta = ag_np.log1p(theta)
    tmp1 = -num_fullres_per_lowres_bins * k * log1p_theta
    if data.sum() == 0:
        log_likelihood = tmp1
    else:
        k_plus_1 = k + 1
        tmp2 = -num_fullres_per_lowres_bins * _stirling(k_plus_1)
        tmp3 = _masksum(_stirling(
            data + k_plus_1), mask=mask, axis=0)
        tmp4 = _masksum(data, mask=mask, axis=0) * (
            ag_np.log(theta) + ag_np.log1p(k) - log1p_theta - 1)
        tmp5 = _masksum((data + k + 0.5) * ag_np.log1p(
            data / k_plus_1), mask=mask, axis=0)
        tmp6 = -_masksum(
            ag_np.log1p(data / k), mask=mask, axis=0)
        log_likelihood = (tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6)

    if 'ij_sum' not in mods:
        log_likelihood = log_likelihood / num_fullres_per_lowres_bins

    log_likelihood = ag_np.mean(log_likelihood)
    return -log_likelihood


@custom_jvp
def log_modified_bessel_1st_kind(v: Any, z: Any) -> Any:
    """Modified Bessel function of the first kind of real order."""
    # res = ag_np.log(iv(v, z))
    # # isinf = res == np.inf
    # # res[isinf] = ag_np.abs(z[isinf])
    # return res
    return lax.select(z < 350, ag_np.log(iv(v, z)), ag_np.abs(z))


def deriv_log_modified_bessel_1st_kind(v: Any, z: Any) -> Any:
    """Modified Bessel function of the first kind of real order."""
    # res = ag_np.log(ivp(v, z))
    # # isinf = res == np.inf
    # # res[isinf] = ag_np.abs(z[isinf])
    # return res
    # 1/iv(2., 0.5) * ivp(2., 0.5)
    return lax.select(z < 350, ivp(v, z) / iv(v, z), lax.full_like(z, 1))


log_modified_bessel_1st_kind.defjvps(  # FIXME check this
    lambda g1, ans, v, z: lax.full_like(g1, 0),
    lambda g2, ans, v, z: deriv_log_modified_bessel_1st_kind(v, z) * g2)


# @custom_jvp
# def log_modified_bessel1_mean(v: Any, z: Any) -> Any:
#     """Modified Bessel function of the first kind of real order."""
# def deriv_log_modified_bessel1_mean(v: Any, z: Any) -> Any:
#     """Modified Bessel function of the first kind of real order."""
# log_modified_bessel1_mean.defjvps(  # FIXME check this
#     lambda g1, ans, v, z: deriv_log_modified_bessel1_mean(v, z) * g1,
#     lambda g2, ans, v, z: deriv_log_modified_bessel1_mean(v, z) * g2)


# def mean_log_bessel_iv(z, v):
#     return ag_np.mean(tfp.math.log_bessel_ive(v, z) + ag_np.abs(z))

# grad_mean_log_bessel_ive = grad(mean_log_bessel_iv)

# @custom_jvp
# def mean_log_mod_bessel_1st(v: Any, z: Any) -> Any:
#     return mean_log_bessel_iv(z, v=v)


# mean_log_mod_bessel_1st.defjvps(  # FIXME check this
#     lambda g1, ans, v, z: lax.full_like(g1, 0),
#     lambda g2, ans, v, z: grad_mean_log_bessel_ive(z, v=v) * g2)


def skellam_nll(data, mu1, mu2, mods=[]):
    if 'tfp' in mods:
        z = 2 * ag_np.sqrt(mu1 * mu2)
        log_mod_bessel1 = tfp.math.log_bessel_ive(data, z) + ag_np.abs(z)
        log_mod_bessel1_mean = ag_np.mean(log_mod_bessel1)
    # elif 'tfp2' in mods:
    #     z = 2 * ag_np.sqrt(mu1 * mu2)
    #     log_mod_bessel1 = mean_log_mod_bessel_1st(data, z)
    #     log_mod_bessel1_mean = ag_np.mean(log_mod_bessel1)
    else:
        z = 2 * ag_np.sqrt(mu1 * mu2)
        log_mod_bessel1 = log_modified_bessel_1st_kind(data, z)
        # log_mod_bessel1 = log_modified_bessel_1st_kind(data, relu_min(z, 350))

        # if isinstance(log_mod_bessel1, np.ndarray):
        #     print(log_mod_bessel1.mean(), flush=True)

        # isinf = log_mod_bessel1 == np.inf
        # if isinf.sum() > 0:
        #     print(isinf.sum())
        #     print(ag_np.sqrt(mu1 * mu2)[isinf].min(), ag_np.sqrt(mu1 * mu2)[~isinf].max())
        #     print("\n* * * * * * INF IN BESSEL * * * * * *\n")
        #     exit(1)
        #     # 360.69380971291554 353.3475608389218
        log_mod_bessel1_mean = ag_np.mean(log_mod_bessel1)

    obj = ag_np.mean(mu1) + ag_np.mean(mu2) - log_mod_bessel1_mean
    if np.any(data != 0):
        obj = obj - ag_np.mean(data / 2 * (ag_np.log(mu1) - ag_np.log(mu2)))
    return obj


def negbinom_nll(data, n, p, mean=True):
    if mean:
        tmp1 = ag_np.mean(gammaln(data + n))
        tmp2 = - ag_np.mean(gammaln(n))
        tmp3 = ag_np.mean(data * ag_np.log(1 - p))
        tmp4 = ag_np.mean(n * ag_np.log(p))
    else:
        tmp1 = ag_np.sum(gammaln(data + n)) / data.shape[0]
        tmp2 = - ag_np.sum(gammaln(n))
        tmp3 = ag_np.sum(data * ag_np.log(1 - p)) / data.shape[0]
        tmp4 = ag_np.sum(n * ag_np.log(p))
    log_likelihood = tmp1 + tmp2 + tmp3 + tmp4
    return -log_likelihood


def poisson_nll(data, lambda_pois, mask=None, num_fullres_per_lowres_bins=1, mean=True):
    if mean:
        fxn = ag_np.mean
    else:
        fxn = ag_np.sum

    lambda_pois = ag_np.asarray(lambda_pois)
    data = ag_np.asarray(data)

    obj = fxn(lambda_pois * num_fullres_per_lowres_bins)
    non0 = (data != 0)
    if np.any(non0):
        if data.ndim > lambda_pois.ndim:
            data = _masksum(data, mask=mask, axis=0)
            # non0 = (data != 0)
        # obj = obj - fxn(data[non0] * ag_np.log(lambda_pois[non0])) # TODO implement
        obj = obj - fxn(data * ag_np.log(lambda_pois))
    return obj
