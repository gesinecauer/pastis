import sys
import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.special import gammaln as gammaln
from scipy.stats import nbinom, poisson


pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from pastis.optimization.utils_poisson import _setup_jax
    _setup_jax(traceback=True, debug_nan_inf=True)

    from pastis.optimization import likelihoods
    from pastis.optimization.polynomial import _polyval


coefs_stirling = np.array([
    8.11614167470508450300E-4, -5.95061904284301438324E-4,
    7.93650340457716943945E-4, -2.77777777730099687205E-3,
    8.33333333333331927722E-2])


def stirling_with_np_polyval(z):
    z_sq_inv = 1 / (z * z)
    return np.polyval(coefs_stirling, x=z_sq_inv) / z


def gammaln_via_stirling_approx(x, stirling_fxn):
    LS2PI = np.log(2 * np.pi) / 2
    return (x - 0.5) * np.log(x) - x + LS2PI + stirling_fxn(x)


@pytest.mark.parametrize("x", [1.8, 3.5, 6.23, 10.4, 81.3, 140.5])
def test_polyval(x):
    correct_polyval = np.polyval(coefs_stirling, x=x)
    jax_polyval = _polyval(x, c=coefs_stirling)
    assert_allclose(correct_polyval, jax_polyval)


@pytest.mark.parametrize("x", [1.8, 3.5, 6.23, 10.4, 81.3, 140.5])
def test_stirling_approx(x):
    correct_stirling = stirling_with_np_polyval(x)
    jax_stirling = likelihoods._stirling(x)
    assert_allclose(correct_stirling, jax_stirling)


@pytest.mark.parametrize("x", [1.8, 3.5, 6.23, 10.4, 81.3, 140.5])
def test_gammaln_numpy(x):
    correct_gammaln = gammaln(x)
    np_gammaln = gammaln_via_stirling_approx(
        x, stirling_fxn=stirling_with_np_polyval)
    assert_allclose(correct_gammaln, np_gammaln, rtol=1e-4)


@pytest.mark.parametrize("x", [1.8, 3.5, 6.23, 10.4, 81.3, 140.5])
def test_gammaln_jax(x):
    correct_gammaln = gammaln(x)
    jax_gammaln = gammaln_via_stirling_approx(
        x, stirling_fxn=likelihoods._stirling)
    assert_allclose(correct_gammaln, jax_gammaln, rtol=1e-4)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_gamma_poisson_nll(seed):
    nbins = 100
    data_per_bin = 50

    random_state = np.random.RandomState(seed=seed)
    n = random_state.uniform(low=0.01, high=100, size=nbins)
    p = random_state.uniform(low=0.01, high=1, size=nbins)
    data = random_state.negative_binomial(n=n, p=p, size=(data_per_bin, nbins))

    k = n
    theta = (1 - p) / p
    nll = likelihoods.gamma_poisson_nll(theta=theta, k=k, data=data)._value

    log_factorial_data = gammaln(data + 1).mean()
    nll_scipy = -nbinom.logpmf(data, n=n, p=p).mean() - log_factorial_data

    tmp1 = np.mean(gammaln(data + n))
    tmp2 = - np.mean(gammaln(n))
    tmp3 = np.mean(data * np.log(1 - p))
    tmp4 = np.mean(n * np.log(p))
    nll_correct = - (tmp1 + tmp2 + tmp3 + tmp4)

    assert_allclose(nll, nll_scipy)
    assert_allclose(nll, nll_correct)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_poisson_nll(seed):
    nbins = 100

    random_state = np.random.RandomState(seed=seed)
    mu = random_state.uniform(low=0, high=100, size=nbins)
    data = random_state.poisson(mu, size=nbins)

    nll = likelihoods.poisson_nll(data=data, lambda_pois=mu)

    log_factorial_data = gammaln(data + 1).mean()
    nll_scipy = -poisson.logpmf(data, mu=mu).mean() - log_factorial_data

    assert_allclose(nll, nll_scipy)
