import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import numpy as np

from .utils_poisson import _setup_jax
_setup_jax()
import jax.numpy as jnp
from .polynomial import _polyval


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
        return jnp.sum(x, axis=axis)
    elif axis is None:
        return jnp.sum(x[mask])
    else:
        return jnp.sum(jnp.where(mask, x, 0), axis=axis)


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
    # if bias_per_bin is not None and np.all(bias_per_bin == 1):
    #     bias_per_bin = None

    if bias_per_bin is None:
        log1p_theta = jnp.log1p(theta)
        tmp1 = jnp.mean(-k * log1p_theta)
    else:
        log1p_theta = jnp.log1p(theta * bias_per_bin)
        tmp1 = jnp.mean(
            _masksum(-k * log1p_theta, mask=mask, axis=0) / data_per_bin)

    if data is None or data.sum() == 0:
        log_likelihood = tmp1
    else:
        k_plus_1 = k + 1
        tmp2 = -jnp.mean(_stirling(k_plus_1))
        tmp3 = jnp.mean(_masksum(
            _stirling(data + k_plus_1), mask=mask, axis=0) / data_per_bin)
        if bias_per_bin is None:
            tmp4 = jnp.mean(_masksum(data, mask=mask, axis=0) / data_per_bin * (
                jnp.log(theta) + jnp.log1p(k) - log1p_theta - 1))
        else:
            tmp4a = jnp.mean(_masksum(data, mask=mask, axis=0) / data_per_bin * (
                jnp.log(theta) + jnp.log1p(k) - 1))
            tmp4b = -jnp.mean(
                _masksum(data * log1p_theta, mask=mask, axis=0) / data_per_bin)
            tmp4 = tmp4a + tmp4b
        tmp5 = jnp.mean(_masksum((data + k + 0.5) * jnp.log1p(
            data / k_plus_1), mask=mask, axis=0) / data_per_bin)
        tmp6 = -jnp.mean(
            _masksum(jnp.log1p(data / k), mask=mask, axis=0) / data_per_bin)
        log_likelihood = (tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6)

    return -log_likelihood


def poisson_nll(data, lambda_pois, mask=None, data_per_bin=None):
    """TODO"""
    if data_per_bin is None:
        if mask is not None:
            data_per_bin = mask.sum(axis=0).reshape(1, -1)
        elif data is None:
            raise ValueError("Must input data_per_bin")
        elif data.ndim == 2:
            data_per_bin = data.shape[0]
        else:
            data_per_bin = 1

    obj = jnp.mean(lambda_pois * data_per_bin)

    if data is None or data.sum() == 0:
        return obj

    if data.ndim > lambda_pois.ndim:
        data = _masksum(data, mask=mask, axis=0)
    obj = obj - jnp.mean(data * jnp.log(lambda_pois))
    return obj
