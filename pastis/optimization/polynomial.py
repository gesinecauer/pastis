import sys
import numpy as np

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

from .utils_poisson import _setup_jax
_setup_jax()
import jax.numpy as jnp
from .utils_poisson import relu_min, relu_max


def _polyval(x, c, unroll=128):
    return jnp.polyval(c, x, unroll=unroll)


def _polygrid2d(c, *args):
    """Analagous to np.polynomial.polynomial.polygrid2d (which is not
    differentiable with jax)
    Note: jax.numpy.polyval uses flipped coefs"""

    for xi in args:
        if isinstance(xi, (tuple, list)):
            xi = jnp.asarray(xi)
        if isinstance(xi, jnp.ndarray):
            c = c.reshape(c.shape + (1,) * xi.ndim)
        c = _polyval(xi, c)
    return c


def _approx_ln_f_mean(epsilon_over_dis, alpha, inferring_alpha=False):
    """TODO"""
    if alpha == -3 and not inferring_alpha:
        ln_f_mean = _polyval(epsilon_over_dis, c=coefs_mean_alpha_minus3)
    else:
        ln_f_mean = _polygrid2d(coefs_mean, epsilon_over_dis, alpha)
    return ln_f_mean


def _approx_ln_f_var(epsilon_over_dis, alpha, inferring_alpha=False):
    """TODO"""
    if alpha == -3 and not inferring_alpha:
        ln_f_var = _polyval(epsilon_over_dis, c=coefs_var_alpha_minus3)
    else:
        ln_f_var = _polygrid2d(coefs_var, epsilon_over_dis, alpha)
    return ln_f_var


def _approx_ln_f(dis, epsilon, alpha, inferring_alpha=False,
                 min_epsilon_over_dis=1e-3, max_epsilon_over_dis=25, mods=[]):
    """TODO"""

    epsilon_over_dis = epsilon / dis
    # TODO temp, verify jax_min, jax_max
    epsilon_over_dis = relu_max(epsilon_over_dis, min_epsilon_over_dis)
    epsilon_over_dis = relu_min(epsilon_over_dis, max_epsilon_over_dis)

    epsilon_over_dis = jnp.log(epsilon_over_dis)

    ln_f_mean = _approx_ln_f_mean(
        epsilon_over_dis, alpha=alpha, inferring_alpha=inferring_alpha)
    ln_f_var = _approx_ln_f_var(
        epsilon_over_dis, alpha=alpha, inferring_alpha=inferring_alpha)

    return ln_f_mean, ln_f_var


# COEFS FOR LOGGED EPS, AT MIN_DIS=0.50 (Flipped coefs - for numpy.polyval or jax.numpy.polyval)
coefs_mean_alpha_minus3 = np.array([
    1.5596073433917893e-06, 3.2620631488453274e-05, 0.00016463647909964506,
    -0.0007106011663149201, -0.007093818737731484, 0.0008450452323096411,
    0.09866938076438023, 0.04593886763105599, -0.8280920323069436,
    -1.4597817710384067, -0.4186793292698835])
coefs_var_alpha_minus3 = np.array([
    6.785535786260004e-07, 2.5465203313866906e-05, 0.0002693630516037337,
    0.0002922335763135698, -0.008895686550510727, -0.026232395913149353,
    0.11427103692416216, 0.31926085489852724, -1.2231132550595873,
    -1.7139620396967072, 0.7021763995509929])
coefs_mean = coefs_var = None
