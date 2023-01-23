import sys
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from utils import get_counts

    from pastis.optimization.utils_poisson import _setup_jax
    _setup_jax(debug_nan_inf=True)

    from pastis.optimization import estimate_alpha_beta
    from pastis.optimization.counts import _format_counts


def test_estimate_alpha_beta_haploid():
    lengths = np.array([20])
    ploidy = 1
    seed = 0
    alpha_correct, beta_correct = -3., 2.

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.rand(lengths.sum() * ploidy, 3)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha_correct,
        beta=beta_correct, ambiguity="ua", struct_nan=None,
        random_state=random_state, use_poisson=False, bias=None)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta_correct)

    alpha, obj, converged, _, conv_desc = estimate_alpha_beta.estimate_alpha(
        X=struct_true, counts=counts, alpha_init=alpha_correct, lengths=lengths,
        ploidy=ploidy)

    beta = list(estimate_alpha_beta._estimate_beta(
        X=struct_true, counts=counts, alpha=alpha, lengths=lengths,
        ploidy=ploidy, verbose=False).values())[0]

    assert converged
    assert obj < (-1e4 / sum([c.nbins for c in counts]))
    assert_array_almost_equal(alpha_correct, alpha, decimal=3)
    assert_array_almost_equal(beta_correct, beta, decimal=3)


def test_estimate_alpha_beta_haploid_biased():
    lengths = np.array([20])
    ploidy = 1
    seed = 0
    alpha_correct, beta_correct = -3., 3.

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.rand(lengths.sum() * ploidy, 3)
    bias = 0.1 + random_state.rand(lengths.sum())
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha_correct,
        beta=beta_correct, ambiguity="ua", struct_nan=None,
        random_state=random_state, use_poisson=False, bias=bias)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta_correct,
        bias=bias)

    alpha, obj, converged, _, conv_desc = estimate_alpha_beta.estimate_alpha(
        X=struct_true, counts=counts, alpha_init=alpha_correct, lengths=lengths,
        ploidy=ploidy, bias=bias)

    beta = list(estimate_alpha_beta._estimate_beta(
        X=struct_true, counts=counts, alpha=alpha, lengths=lengths,
        ploidy=ploidy, bias=bias, verbose=False).values())[0]

    assert converged
    assert obj < (-1e3 / sum([c.nbins for c in counts]))
    assert_array_almost_equal(alpha_correct, alpha, decimal=5)
    assert_array_almost_equal(beta_correct, beta, decimal=1)


@pytest.mark.parametrize("ambiguity", ["ua", "ambig", "pa"])
def test_estimate_alpha_beta_diploid(ambiguity):
    lengths = np.array([20])
    ploidy = 2
    seed = 0
    alpha_correct, beta_correct = -3., 2.

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.rand(lengths.sum() * ploidy, 3)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha_correct,
        beta=beta_correct, ambiguity=ambiguity, struct_nan=None,
        random_state=random_state, use_poisson=False, bias=None)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta_correct)

    alpha, obj, converged, _, conv_desc = estimate_alpha_beta.estimate_alpha(
        X=struct_true, counts=counts, alpha_init=alpha_correct, lengths=lengths,
        ploidy=ploidy)

    beta = list(estimate_alpha_beta._estimate_beta(
        X=struct_true, counts=counts, alpha=alpha, lengths=lengths,
        ploidy=ploidy, verbose=False).values())[0]

    assert converged
    assert obj < (-1e4 / sum([c.nbins for c in counts]))
    assert_array_almost_equal(alpha_correct, alpha, decimal=5)
    assert_array_almost_equal(beta_correct, beta, decimal=3)


@pytest.mark.parametrize("ambiguity", ["ua", "ambig", "pa"])
def test_estimate_alpha_beta_diploid_biased(ambiguity):
    lengths = np.array([20])
    ploidy = 2
    seed = 0
    alpha_correct, beta_correct = -3., 2.

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.rand(lengths.sum() * ploidy, 3)
    bias = 0.1 + random_state.rand(lengths.sum())
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha_correct,
        beta=beta_correct, ambiguity=ambiguity, struct_nan=None,
        random_state=random_state, use_poisson=False, bias=bias)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta_correct,
        bias=bias)

    alpha, obj, converged, _, conv_desc = estimate_alpha_beta.estimate_alpha(
        X=struct_true, counts=counts, alpha_init=alpha_correct, lengths=lengths,
        ploidy=ploidy, bias=bias)

    beta = list(estimate_alpha_beta._estimate_beta(
        X=struct_true, counts=counts, alpha=alpha, lengths=lengths,
        ploidy=ploidy, bias=bias, verbose=False).values())[0]

    assert converged
    assert obj < (-1e4 / sum([c.nbins for c in counts]))
    assert_array_almost_equal(alpha_correct, alpha, decimal=5)
    assert_array_almost_equal(beta_correct, beta, decimal=1)


def test_estimate_alpha_beta_diploid_combo():
    lengths = np.array([20])
    ploidy = 2
    seed = 0
    alpha_correct, beta_correct_single = -3., 4.
    ratio_ambig, ratio_pa, ratio_ua = [1 / 3] * 3
    bias = None

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.rand(lengths.sum() * ploidy, 3)
    counts_ambig = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha_correct,
        beta=beta_correct_single * ratio_ambig, ambiguity="ambig", struct_nan=None,
        random_state=random_state, use_poisson=False, bias=None)
    counts_pa = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha_correct,
        beta=beta_correct_single * ratio_pa, ambiguity="pa", struct_nan=None,
        random_state=random_state, use_poisson=False, bias=None)
    counts_ua = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha_correct,
        beta=beta_correct_single * ratio_ua, ambiguity="ua", struct_nan=None,
        random_state=random_state, use_poisson=False, bias=None)
    counts = [counts_ambig, counts_pa, counts_ua]
    beta_correct = np.array(
        [ratio_ambig, ratio_pa, ratio_ua]) * beta_correct_single

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta_correct)

    alpha, obj, converged, _, conv_desc = estimate_alpha_beta.estimate_alpha(
        X=struct_true, counts=counts, alpha_init=alpha_correct, lengths=lengths,
        ploidy=ploidy, bias=bias)

    beta = list(estimate_alpha_beta._estimate_beta(
        X=struct_true, counts=counts, alpha=alpha, lengths=lengths, ploidy=ploidy,
        bias=bias, verbose=False).values())[0]

    assert converged
    assert obj < (-1e4 / sum([c.nbins for c in counts]))
    assert_array_almost_equal(alpha_correct, alpha, decimal=3)
    assert_array_almost_equal(beta_correct, beta, decimal=3)
