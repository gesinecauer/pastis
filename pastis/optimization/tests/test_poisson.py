import sys
import pytest
import numpy as np

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from utils import get_counts, get_struct_randwalk
    from utils import decrease_struct_res_correct

    from pastis.optimization.utils_poisson import _setup_jax
    _setup_jax(debug_nan_inf=True)

    from pastis.optimization import poisson
    from pastis.optimization.counts import _format_counts
    from pastis.optimization.counts import preprocess_counts
    from pastis.optimization.multiscale_optimization import get_multiscale_epsilon_from_struct


def test_poisson_objective_haploid():
    lengths = np.array([20])
    ploidy = 1
    seed = 42
    alpha, beta = -3, 1

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    struct_true = random_state.rand(n * ploidy, 3)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity="ua", struct_nan=None, random_state=random_state,
        use_poisson=False, bias=None)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta)

    obj = poisson.objective(
        X=struct_true, counts=counts, alpha=alpha, lengths=lengths,
        ploidy=ploidy)

    assert obj < (-1e4 / sum([c.nbins for c in counts]))


def test_poisson_objective_haploid_biased():
    lengths = np.array([20])
    ploidy = 1
    seed = 42
    alpha, beta = -3, 1

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    struct_true = random_state.rand(n * ploidy, 3)
    bias = 0.1 + random_state.rand(n)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity="ua", struct_nan=None, random_state=random_state,
        use_poisson=False, bias=bias)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta)

    obj = poisson.objective(
        X=struct_true, counts=counts, alpha=alpha, lengths=lengths,
        ploidy=ploidy, bias=bias)

    assert obj < (-1e3 / sum([c.nbins for c in counts]))


@pytest.mark.parametrize("ambiguity", ["ua", "ambig", "pa"])
def test_poisson_objective_diploid(ambiguity):
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    alpha, beta = -3, 1

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    struct_true = random_state.rand(n * ploidy, 3)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=None, random_state=random_state,
        use_poisson=False, bias=None)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta)

    obj = poisson.objective(
        X=struct_true, counts=counts, alpha=alpha, lengths=lengths,
        ploidy=ploidy)

    assert obj < (-1e4 / sum([c.nbins for c in counts]))


@pytest.mark.parametrize("ambiguity", ["ua", "ambig", "pa"])
def test_poisson_objective_diploid_biased(ambiguity):
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    alpha, beta = -3, 1

    random_state = np.random.RandomState(seed=seed)
    n = lengths.sum()
    struct_true = random_state.rand(n * ploidy, 3)
    bias = 0.1 + random_state.rand(n)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=None, random_state=random_state,
        use_poisson=False, bias=bias)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta)

    obj = poisson.objective(
        X=struct_true, counts=counts, alpha=alpha, lengths=lengths,
        ploidy=ploidy, bias=bias)

    assert obj < (-1e4 / sum([c.nbins for c in counts]))


@pytest.mark.parametrize("ambiguity,multiscale_factor", [
    ('ua', 2), ('ambig', 2), ('pa', 2), ('ua', 4), ('ambig', 4), ('pa', 4),
    ('ua', 8), ('ambig', 8), ('pa', 8)])
def test_objective_multiscale(ambiguity, multiscale_factor):  # TODO also test with bias
    lengths = np.array([20])
    ploidy = 2
    seed = 42
    alpha, beta = -3, 1e3
    true_interhmlg_dis = np.array([15.])

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state,
        true_interhmlg_dis=true_interhmlg_dis)
    struct_true_lowres = decrease_struct_res_correct(
        struct_true, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=None, random_state=random_state,
        use_poisson=False)

    counts, struct_nan, _ = preprocess_counts(
        counts_raw=counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, beta=beta, verbose=False)
    epsilon_true = np.mean(get_multiscale_epsilon_from_struct(
        struct_true, lengths=lengths, multiscale_factor=multiscale_factor,
        verbose=False))

    obj = poisson.objective(
        X=struct_true_lowres, counts=counts, alpha=alpha, lengths=lengths,
        ploidy=ploidy, bias=None, multiscale_factor=multiscale_factor,
        multiscale_reform=True, epsilon=epsilon_true)

    assert obj < (-1e4 / sum([c.nbins for c in counts]))
