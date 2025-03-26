import sys
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from utils import get_counts_haploid, get_counts_diploid

    from pastis.optimization.utils_poisson import _setup_jax
    _setup_jax(traceback=True, debug_nan_inf=True)

    from pastis.optimization import estimate_alpha_beta
    from pastis.optimization.counts import _format_counts


def test_estimate_alpha_beta_haploid():
    lengths = np.array([20])
    ploidy = 1
    seed = 0
    alpha_true, beta_true = -3., 2.

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.uniform(size=(lengths.sum() * ploidy, 3))
    counts, beta_true = get_counts_haploid(
        struct_true, lengths=lengths, alpha=alpha_true, beta=beta_true,
        struct_nan=None, random_state=random_state, use_poisson=False,
        bias=None)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta_true)

    alpha, obj, converged, _, conv_desc = estimate_alpha_beta.estimate_alpha(
        X=struct_true, counts=counts, alpha_init=alpha_true, lengths=lengths,
        ploidy=ploidy)

    beta = estimate_alpha_beta._estimate_beta(
        X=struct_true, counts=counts, alpha=alpha, lengths=lengths,
        ploidy=ploidy)[0]._value

    assert converged
    assert obj < (-1e4 / sum([c.nbins for c in counts]))
    assert_array_almost_equal(alpha_true, alpha, decimal=3)
    assert_array_almost_equal(beta_true, beta, decimal=3)


def test_estimate_alpha_beta_haploid_biased():
    lengths = np.array([20])
    ploidy = 1
    seed = 0
    alpha_true, beta_true = -3., 3.

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.uniform(size=(lengths.sum() * ploidy, 3))
    bias = 0.1 + random_state.uniform(size=lengths.sum())
    counts, beta_true = get_counts_haploid(
        struct_true, lengths=lengths, alpha=alpha_true, beta=beta_true,
        struct_nan=None, random_state=random_state, use_poisson=False,
        bias=bias)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta_true,
        bias=bias)

    alpha, obj, converged, _, conv_desc = estimate_alpha_beta.estimate_alpha(
        X=struct_true, counts=counts, alpha_init=alpha_true, lengths=lengths,
        ploidy=ploidy, bias=bias)

    beta = estimate_alpha_beta._estimate_beta(
        X=struct_true, counts=counts, alpha=alpha, lengths=lengths,
        ploidy=ploidy, bias=bias)[0]._value

    assert converged
    assert obj < (-1e3 / sum([c.nbins for c in counts]))
    assert_array_almost_equal(alpha_true, alpha, decimal=5)
    assert_array_almost_equal(beta_true, beta, decimal=1)


@pytest.mark.parametrize("ambiguity", ["ua", "ambig", "pa"])
def test_estimate_alpha_beta_diploid(ambiguity):
    lengths = np.array([20])
    ploidy = 2
    seed = 0
    alpha_true, beta_ambig = -3., 2.

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.uniform(size=(lengths.sum() * ploidy, 3))
    counts, beta_true = get_counts_diploid(
        struct_true, lengths=lengths, alpha=alpha_true,
        beta_ambig=beta_ambig, ambiguity=ambiguity, struct_nan=None,
        random_state=random_state, use_poisson=False, bias=None)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta_true)

    alpha, obj, converged, _, conv_desc = estimate_alpha_beta.estimate_alpha(
        X=struct_true, counts=counts, alpha_init=alpha_true, lengths=lengths,
        ploidy=ploidy)

    beta = estimate_alpha_beta._estimate_beta(
        X=struct_true, counts=counts, alpha=alpha, lengths=lengths,
        ploidy=ploidy)[0]._value

    assert converged
    assert obj < (-1e4 / sum([c.nbins for c in counts]))
    assert_array_almost_equal(alpha_true, alpha, decimal=5)
    assert_array_almost_equal(beta_true, beta, decimal=3)


@pytest.mark.parametrize("ambiguity", ["ua", "ambig", "pa"])
def test_estimate_alpha_beta_diploid_biased(ambiguity):
    lengths = np.array([20])
    ploidy = 2
    seed = 0
    alpha_true, beta_ambig = -3., 2.

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.uniform(size=(lengths.sum() * ploidy, 3))
    bias = 0.1 + random_state.uniform(size=lengths.sum())
    counts, beta_true = get_counts_diploid(
        struct_true, lengths=lengths, alpha=alpha_true,
        beta_ambig=beta_ambig, ambiguity=ambiguity, struct_nan=None,
        random_state=random_state, use_poisson=False, bias=bias)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta_true,
        bias=bias)

    alpha, obj, converged, _, conv_desc = estimate_alpha_beta.estimate_alpha(
        X=struct_true, counts=counts, alpha_init=alpha_true, lengths=lengths,
        ploidy=ploidy, bias=bias)

    beta = estimate_alpha_beta._estimate_beta(
        X=struct_true, counts=counts, alpha=alpha, lengths=lengths,
        ploidy=ploidy, bias=bias)[0]._value

    assert converged
    assert obj < (-1e4 / sum([c.nbins for c in counts]))
    assert_array_almost_equal(alpha_true, alpha, decimal=5)
    assert_array_almost_equal(beta_true, beta, decimal=1)


def test_estimate_alpha_beta_diploid_combo():
    lengths = np.array([20])
    ploidy = 2
    seed = 0
    alpha_true, beta_ambig_true = -3., 4.
    nreads_ratios = {'ambig': 0.2, 'pa': 0.7, 'ua': 0.1}
    bias = None

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.uniform(size=(lengths.sum() * ploidy, 3))

    counts = []
    beta_true = []
    nreads_ratios = {k: (v / sum(
        nreads_ratios.values())) for k, v in nreads_ratios.items()}
    for ambiguity, nreads_ratio in nreads_ratios.items():
        counts_tmp, beta_tmp = get_counts_diploid(
            struct_true, lengths=lengths, alpha=alpha_true,
            beta_ambig=beta_ambig_true * nreads_ratio, ambiguity=ambiguity,
            struct_nan=None, random_state=random_state, use_poisson=False,
            bias=bias)
        counts.append(counts_tmp)
        beta_true.append(beta_tmp)
    beta_true = np.array(beta_true)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta_true)

    alpha, obj, converged, _, conv_desc = estimate_alpha_beta.estimate_alpha(
        X=struct_true, counts=counts, alpha_init=alpha_true, lengths=lengths,
        ploidy=ploidy, bias=bias)

    beta = estimate_alpha_beta._estimate_beta(
        X=struct_true, counts=counts, alpha=alpha, lengths=lengths,
        ploidy=ploidy, bias=bias)._value

    assert converged
    assert obj < (-1e4 / sum([c.nbins for c in counts]))
    assert_array_almost_equal(alpha_true, alpha, decimal=3)
    assert_array_almost_equal(beta_true, beta, decimal=3)
