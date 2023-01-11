import sys
import numpy as np

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

from .utils_poisson import _setup_jax
_setup_jax()
from jax import custom_jvp, lax
import jax.numpy as ag_np
from jax.scipy.special import gammaln
from jax.scipy.stats.nbinom import logpmf as logpmf_negbinom

from tensorflow_probability.substrates import jax as tfp
from typing import Any
from scipy.special import iv, ivp
from .polynomial import _polyval


# coefs_stirling = np.flip(np.array([
#     8.11614167470508450300E-4, -5.95061904284301438324E-4,
#     7.93650340457716943945E-4, -2.77777777730099687205E-3,
#     8.33333333333331927722E-2]))  # TODO what happened here?
coefs_stirling = np.array([
    8.11614167470508450300E-4, -5.95061904284301438324E-4,
    7.93650340457716943945E-4, -2.77777777730099687205E-3,
    8.33333333333331927722E-2])


def _stirling(z):
    """Computes B_N(z)
    """
    z_sq_inv = 1 / (z * z)
    return _polyval(z_sq_inv, c=coefs_stirling) / z


def _masksum(x, mask=None, axis=None):
    """Sum of masked array (for jax)"""
    if mask is None:
        return ag_np.sum(x, axis=axis)
    elif axis is None:
        return ag_np.sum(x[mask])
    else:
        return ag_np.sum(ag_np.where(mask, x, 0), axis=axis)


def gamma_poisson_nll(theta, k, data, bias_per_bin=None, mask=None,
                      data_per_bin=None, mods=[]):
    """TODO"""

    if data_per_bin is None:
        if mask is not None:
            data_per_bin = mask.sum(axis=0).reshape(1, -1)
        elif data is not None:
            data_per_bin = data.shape[0]
        else:
            raise ValueError("Must input data_per_bin")
    if np.all(bias_per_bin == 1):
        bias_per_bin = None

    if bias_per_bin is None:
        log1p_theta = ag_np.log1p(theta)
    else:
        log1p_theta = ag_np.log1p(theta * bias_per_bin)
    if bias_per_bin is None:
        tmp1 = ag_np.mean(-k * log1p_theta)
    else:
        tmp1 = ag_np.mean(_masksum(-k * log1p_theta, mask=mask, axis=0) / data_per_bin)

    if data is None or data.sum() == 0:
        log_likelihood = tmp1
    else:
        k_plus_1 = k + 1
        tmp2 = -ag_np.mean(_stirling(k_plus_1))
        tmp3 = ag_np.mean(_masksum(
            _stirling(data + k_plus_1), mask=mask, axis=0) / data_per_bin)
        if bias_per_bin is None:
            tmp4 = ag_np.mean(_masksum(data, mask=mask, axis=0) / data_per_bin * (
                ag_np.log(theta) + ag_np.log1p(k) - log1p_theta - 1))
        else:
            tmp4a = ag_np.mean(_masksum(data, mask=mask, axis=0) / data_per_bin * (
                ag_np.log(theta) + ag_np.log1p(k) - 1))
            tmp4b = -ag_np.mean(
                _masksum(data * log1p_theta, mask=mask, axis=0) / data_per_bin)
            tmp4 = tmp4a + tmp4b
        tmp5 = ag_np.mean(_masksum((data + k + 0.5) * ag_np.log1p(
            data / k_plus_1), mask=mask, axis=0) / data_per_bin)
        tmp6 = -ag_np.mean(
            _masksum(ag_np.log1p(data / k), mask=mask, axis=0) / data_per_bin)
        log_likelihood = (tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6)

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


# def invgamma_nll(data, a, b):
#     tmp1 = -(a * ag_np.log(b)).mean()
#     tmp2 = gammaln(a).mean()
#     tmp3 = ((a + 1) * ag_np.log(data)).mean()
#     tmp4 = (b / data).mean()
#     obj = tmp1 + tmp2 + tmp3 + tmp4

#     # if type(obj).__name__ in ('ndarray', 'float', 'DeviceArray'):
#     #     from scipy.stats import invgamma
#     #     test_obj = - invgamma.logpdf(data._value, a=a, scale=b).mean()
#     #     if not ag_np.isclose(obj, test_obj):
#     #         print("ugh invgamma is wrong"); exit(1)
#     return obj


# def gengamma_nll(data, a, d, p):
#     tmp1 = - ag_np.log(p).mean()
#     tmp2 = (d * ag_np.log(a)).mean()
#     tmp3 = - (d * ag_np.log(data)).mean()
#     tmp4 = ag_np.power(data / a, p).mean()
#     tmp5 = gammaln(d / p).mean()
#     obj = tmp1 + tmp2 + tmp3 + tmp4
#     return obj


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


def negbinom_nll(data, n, p, mean=True, use_scipy=False, num_stable=True):
    if num_stable:
        k = n
        theta = (1 - p) / p

        # tmp = ' obj' if type(k).__name__ in ('DeviceArray', 'ndarray') else 'grad'  # TODO
        # print(f"{tmp}: gamma_poisson_nll begin", flush=True)  # TODO
        nll = gamma_poisson_nll(theta=theta, k=k, data=data)
        # print(f"{tmp}: gamma_poisson_nll done", flush=True)  # TODO

        if not mean:
            nll = nll * n.size
        return nll
    if use_scipy:
        log_factorial_data = gammaln(data + 1).mean()
        if mean:
            log_likelihood = logpmf_negbinom(data, n=n, p=p).mean()
            log_likelihood = log_likelihood + log_factorial_data
        else:
            log_likelihood = logpmf_negbinom(
                data, n=n, p=p).sum() / data.shape[0]
            log_likelihood = log_likelihood + log_factorial_data * n.size
        return -log_likelihood
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


def poisson_nll(data, lambda_pois, mask=None, data_per_bin=None):
    if data_per_bin is None:
        if mask is not None:
            data_per_bin = mask.sum(axis=0).reshape(1, -1)
        elif data.ndim == 2:
            data_per_bin = data.shape[0]
        else:
            data_per_bin = 1

    obj = ag_np.mean(lambda_pois * data_per_bin)

    if data.sum() == 0:
        return obj

    if data.ndim > lambda_pois.ndim:
        data = _masksum(data, mask=mask, axis=0)
    obj = obj - ag_np.mean(data * ag_np.log(lambda_pois))
    return obj
