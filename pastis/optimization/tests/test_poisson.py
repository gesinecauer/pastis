import sys
import pytest
import numpy as np
from numpy.testing import assert_allclose

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires python3.6 or higher")

if sys.version_info[0] >= 3:
    from utils import get_counts, get_struct_randwalk
    from utils import decrease_struct_res_correct

    from pastis.optimization.utils_poisson import _setup_jax
    _setup_jax(traceback=True, debug_nan_inf=True)

    from pastis.optimization import poisson
    from pastis.optimization.counts import _format_counts
    from pastis.optimization.counts import preprocess_counts
    from pastis.optimization.multiscale_optimization import get_epsilon_from_struct
    from pastis.optimization.multiscale_optimization import decrease_bias_res


def test_poisson_objective_haploid():
    lengths = np.array([40])
    ploidy = 1
    seed = 0
    alpha, beta = -3, 1

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.uniform(size=(lengths.sum() * ploidy, 3))
    bias = None
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity="ua", struct_nan=None, random_state=random_state,
        use_poisson=False, bias=bias)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta, bias=bias)

    obj = poisson.objective(
        X=struct_true, counts=counts, alpha=alpha, lengths=lengths,
        ploidy=ploidy, bias=bias)[0]._value

    assert obj < (-1e4 / sum([c.nbins for c in counts]))


def test_poisson_objective_haploid_biased():
    lengths = np.array([40])
    ploidy = 1
    seed = 0
    alpha, beta = -3, 1

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.uniform(size=(lengths.sum() * ploidy, 3))
    bias = 0.1 + random_state.uniform(size=lengths.sum())
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity="ua", struct_nan=None, random_state=random_state,
        use_poisson=False, bias=bias)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta, bias=bias)

    obj = poisson.objective(
        X=struct_true, counts=counts, alpha=alpha, lengths=lengths,
        ploidy=ploidy, bias=bias)[0]._value

    assert obj < (-1e4 / sum([c.nbins for c in counts]))


@pytest.mark.parametrize("ambiguity", ["ua", "ambig", "pa"])
def test_poisson_objective_diploid(ambiguity):
    lengths = np.array([20])
    ploidy = 2
    seed = 0
    alpha, beta = -3, 1

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.uniform(size=(lengths.sum() * ploidy, 3))
    bias = None
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=None, random_state=random_state,
        use_poisson=False, bias=bias)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta, bias=bias)

    obj = poisson.objective(
        X=struct_true, counts=counts, alpha=alpha, lengths=lengths,
        ploidy=ploidy, bias=bias)[0]._value

    assert obj < (-1e4 / sum([c.nbins for c in counts]))


@pytest.mark.parametrize("ambiguity", ["ua", "ambig", "pa"])
def test_poisson_objective_diploid_biased(ambiguity):
    lengths = np.array([20])
    ploidy = 2
    seed = 0
    alpha, beta = -3, 1

    random_state = np.random.RandomState(seed=seed)
    struct_true = random_state.uniform(size=(lengths.sum() * ploidy, 3))
    bias = 0.1 + random_state.uniform(size=lengths.sum())
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=None, random_state=random_state,
        use_poisson=False, bias=bias)

    counts = _format_counts(
        counts=counts, lengths=lengths, ploidy=ploidy, beta=beta, bias=bias)

    obj = poisson.objective(
        X=struct_true, counts=counts, alpha=alpha, lengths=lengths,
        ploidy=ploidy, bias=bias)[0]._value

    assert obj < (-1e4 / sum([c.nbins for c in counts]))


@pytest.mark.parametrize("ambiguity,multiscale_factor", [
    ('ua', 2), ('ambig', 2), ('pa', 2), ('ua', 4), ('ambig', 4), ('pa', 4),
    ('ua', 8), ('ambig', 8), ('pa', 8)])
def test_objective_multires(ambiguity, multiscale_factor):
    lengths = np.array([40])
    ploidy = 2
    seed = 0
    alpha, beta = -3, 1e3
    true_interhmlg_dis = 15
    multiscale_reform = True

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state,
        true_interhmlg_dis=true_interhmlg_dis)
    bias = None
    struct_true_lowres = decrease_struct_res_correct(
        struct_true, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=None, random_state=random_state,
        use_poisson=False, bias=bias)

    counts, struct_nan, _ = preprocess_counts(
        counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, beta=beta, bias=bias,
        multiscale_reform=multiscale_reform, verbose=False)
    if multiscale_reform:
        epsilon_true, _, _ = get_epsilon_from_struct(
            struct_true, lengths=lengths, ploidy=ploidy,
            multiscale_factor=multiscale_factor, verbose=False)
        X = np.append(struct_true_lowres.ravel(), epsilon_true)
    else:
        X = struct_true_lowres
        bias = decrease_bias_res(
            bias, multiscale_factor=multiscale_factor, lengths=lengths)

    obj = poisson.objective(
        X=X, counts=counts, alpha=alpha, lengths=lengths,
        ploidy=ploidy, multiscale_factor=multiscale_factor,
        multiscale_reform=multiscale_reform, bias=bias)[0]._value

    assert obj < (-1e4 / sum([c.nbins for c in counts]))


@pytest.mark.parametrize("ambiguity,multiscale_factor", [
    ('ua', 2), ('ambig', 2), ('pa', 2), ('ua', 4), ('ambig', 4), ('pa', 4),
    ('ua', 8), ('ambig', 8), ('pa', 8)])
def test_objective_multires_biased(ambiguity, multiscale_factor):
    lengths = np.array([40])
    ploidy = 2
    seed = 0
    alpha, beta = -3, 1e3
    true_interhmlg_dis = 15
    multiscale_reform = True

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state,
        true_interhmlg_dis=true_interhmlg_dis)
    struct_true_lowres = decrease_struct_res_correct(
        struct_true, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)
    bias = 0.1 + random_state.uniform(size=lengths.sum())
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=None, random_state=random_state,
        use_poisson=False, bias=bias)

    counts, struct_nan, _ = preprocess_counts(
        counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, beta=beta, bias=bias,
        multiscale_reform=multiscale_reform, verbose=False)
    if multiscale_reform:
        epsilon_true, _, _ = get_epsilon_from_struct(
            struct_true, lengths=lengths, ploidy=ploidy,
            multiscale_factor=multiscale_factor, verbose=False)
        X = np.append(struct_true_lowres.ravel(), epsilon_true)
    else:
        X = struct_true_lowres
        bias = decrease_bias_res(
            bias, multiscale_factor=multiscale_factor, lengths=lengths)

    obj = poisson.objective(
        X=X, counts=counts, alpha=alpha, lengths=lengths,
        ploidy=ploidy, bias=bias, multiscale_factor=multiscale_factor,
        multiscale_reform=multiscale_reform)[0]._value

    assert obj < (-1e3 / sum([c.nbins for c in counts]))


@pytest.mark.parametrize("ambiguity,multiscale_factor", [
    ('ua', 2), ('ambig', 2), ('pa', 2), ('ua', 4), ('ambig', 4), ('pa', 4),
    ('ua', 8), ('ambig', 8), ('pa', 8)])
def test_objective_multires_bias_approx1(ambiguity, multiscale_factor):
    lengths = np.array([40])
    ploidy = 2
    seed = 0
    alpha, beta = -3, 1e3
    true_interhmlg_dis = 15
    multiscale_reform = True

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
        use_poisson=True, bias=None)  # Counts can be unbiased, doesn't matter

    # Bias should be as close as possible to 1 without being exactly 1
    # (If bias is exactly 1, it gets set to None in the objective)
    bias = np.ones(lengths.sum()) + random_state.choice(
        [-1, 1], size=lengths.sum()) * np.finfo(np.float64).resolution / 8
    assert not np.all(bias == 1)  # Make sure we aren't exactly at 1

    counts, struct_nan, _ = preprocess_counts(
        counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, beta=beta, bias=None,
        multiscale_reform=multiscale_reform, verbose=False)
    if multiscale_reform:
        epsilon_true, _, _ = get_epsilon_from_struct(
            struct_true, lengths=lengths, ploidy=ploidy,
            multiscale_factor=multiscale_factor, verbose=False)
        X = np.append(struct_true_lowres.ravel(), epsilon_true)
    else:
        X = struct_true_lowres
        bias = decrease_bias_res(
            bias, multiscale_factor=multiscale_factor, lengths=lengths)

    obj_unbiased = poisson.objective(
        X=X, counts=counts, alpha=alpha, lengths=lengths,
        ploidy=ploidy, bias=None, multiscale_factor=multiscale_factor,
        multiscale_reform=multiscale_reform)[0]._value
    obj_biased = poisson.objective(
        X=X, counts=counts, alpha=alpha, lengths=lengths,
        ploidy=ploidy, bias=bias, multiscale_factor=multiscale_factor,
        multiscale_reform=multiscale_reform)[0]._value

    print(f"{obj_unbiased=:g}   {obj_biased=:g}")
    assert_allclose(obj_unbiased, obj_biased)


@pytest.mark.parametrize("ambiguity,multiscale_factor", [
    ('ua', 2), ('ambig', 2), ('pa', 2), ('ua', 4), ('ambig', 4), ('pa', 4),
    ('ua', 8), ('ambig', 8), ('pa', 8)])
def test_objective_multires_naive(ambiguity, multiscale_factor):
    lengths = np.array([40])
    ploidy = 2
    seed = 0
    alpha, beta = -3, 1e3
    true_interhmlg_dis = 15
    multiscale_reform = False

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state,
        true_interhmlg_dis=true_interhmlg_dis)
    struct_true_lowres = decrease_struct_res_correct(
        struct_true, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)
    bias = None
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=None, random_state=random_state,
        use_poisson=False, bias=bias)

    counts, struct_nan, _ = preprocess_counts(
        counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, beta=beta, bias=bias,
        multiscale_reform=multiscale_reform, verbose=False)
    if multiscale_reform:
        epsilon_true, _, _ = get_epsilon_from_struct(
            struct_true, lengths=lengths, ploidy=ploidy,
            multiscale_factor=multiscale_factor, verbose=False)
        X = np.append(struct_true_lowres.ravel(), epsilon_true)
    else:
        X = struct_true_lowres
        bias = decrease_bias_res(
            bias, multiscale_factor=multiscale_factor, lengths=lengths)

    obj = poisson.objective(
        X=X, counts=counts, alpha=alpha, lengths=lengths,
        ploidy=ploidy, bias=bias, multiscale_factor=multiscale_factor,
        multiscale_reform=multiscale_reform)[0]._value

    assert obj < (-1e3 / sum([c.nbins for c in counts]))


@pytest.mark.parametrize("ambiguity,multiscale_factor", [
    ('ua', 2), ('ambig', 2), ('pa', 2), ('ua', 4), ('ambig', 4), ('pa', 4),
    ('ua', 8), ('ambig', 8), ('pa', 8)])
def test_objective_multires_naive_biased(ambiguity, multiscale_factor):
    lengths = np.array([40])
    ploidy = 2
    seed = 0
    alpha, beta = -3, 1e3
    true_interhmlg_dis = 15
    multiscale_reform = False

    random_state = np.random.RandomState(seed=seed)
    struct_true = get_struct_randwalk(
        lengths=lengths, ploidy=ploidy, random_state=random_state,
        true_interhmlg_dis=true_interhmlg_dis)
    struct_true_lowres = decrease_struct_res_correct(
        struct_true, multiscale_factor=multiscale_factor, lengths=lengths,
        ploidy=ploidy)
    bias = 0.1 + random_state.uniform(size=lengths.sum())
    counts = get_counts(
        struct_true, ploidy=ploidy, lengths=lengths, alpha=alpha, beta=beta,
        ambiguity=ambiguity, struct_nan=None, random_state=random_state,
        use_poisson=False, bias=bias)

    counts, struct_nan, _ = preprocess_counts(
        counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor, beta=beta, bias=bias,
        multiscale_reform=multiscale_reform, verbose=False)
    if multiscale_reform:
        epsilon_true, _, _ = get_epsilon_from_struct(
            struct_true, lengths=lengths, ploidy=ploidy,
            multiscale_factor=multiscale_factor, verbose=False)
        X = np.append(struct_true_lowres.ravel(), epsilon_true)
    else:
        X = struct_true_lowres
        bias = decrease_bias_res(
            bias, multiscale_factor=multiscale_factor, lengths=lengths)

    obj = poisson.objective(
        X=X, counts=counts, alpha=alpha, lengths=lengths,
        ploidy=ploidy, bias=bias, multiscale_factor=multiscale_factor,
        multiscale_reform=multiscale_reform)[0]._value

    assert obj < (-1e3 / sum([c.nbins for c in counts]))
