import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import numpy as np
import os

from .utils_poisson import _setup_jax
_setup_jax()
import jax.numpy as jnp

from .multiscale_optimization import increase_struct_res, decrease_struct_res
from .multiscale_optimization import decrease_lengths_res
from .multiscale_optimization import decrease_bias_res
from .mds import estimate_X
from .utils_poisson import find_beads_to_remove
from .utils_poisson import _struct_replace_nan, _format_structures


def _initialize_struct_mds(counts, lengths, ploidy, alpha, bias, random_state,
                           multiscale_factor=1, bias_per_hmlg=False,
                           verbose=True):
    """Initialize structure via multi-dimensional scaling of unambig counts.
    """

    if verbose:
        print('INITIALIZATION: multi-dimensional scaling', flush=True)

    if alpha is None:
        raise ValueError("Must supply alpha for MDS initialization.")

    if not isinstance(counts, (list, tuple)):
        counts = [counts]
    ua_counts = [c for c in counts if c.ambiguity == 'ua' and (
        c.multiscale_factor == multiscale_factor)]
    if len(ua_counts) == 1:
        ua_counts = ua_counts[0]
    elif len(ua_counts) == 0:
        raise ValueError("Unambiguous counts needed to initialize via MDS.")
    else:
        raise ValueError(
            "Multiple unambiguous counts matrices detected. Pool data from"
            " unambiguous counts matrices before inference.")

    ua_beta = ua_counts.beta
    if ua_beta is not None:
        ua_beta = ua_beta * multiscale_factor ** 2

    bias_per_hmlg = bias_per_hmlg and ploidy == 2 and len(counts) == 1 and set(
        counts[0].shape) == {lengths.sum() * ploidy}

    bias = decrease_bias_res(
        bias, multiscale_factor=multiscale_factor, lengths=lengths,
        bias_per_hmlg=bias_per_hmlg)
    if bias is not None and not np.all(bias == 1):
        if bias_per_hmlg:
            bias_tiled = bias
        else:
            bias_tiled = np.tile(bias, ploidy)
    else:
        bias_tiled = None

    ua_counts = ua_counts.tocoo()
    if bias is not None:
        ua_counts = ua_counts.astype(float)

    struct = estimate_X(
        ua_counts, alpha=alpha, beta=ua_beta,
        verbose=-1, use_zero_entries=False, precompute_distances='auto',
        bias=bias_tiled, random_state=random_state, type="MDS2", factr=1e12,
        maxiter=10000, ini=None)

    struct = struct.reshape(-1, 3)
    struct_nan = find_beads_to_remove(
        counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor)
    struct[struct_nan] = np.nan

    return struct


def _initialize_struct(counts, lengths, ploidy, alpha, bias, random_state,
                       init='mds', multiscale_factor=1, mixture_coefs=None,
                       struct_true=None, bias_per_hmlg=False, verbose=True):
    """Initialize structure, randomly or via MDS of unambig counts.
    """

    if mixture_coefs is None:
        mixture_coefs = [1]

    if not isinstance(counts, (list, tuple)):
        counts = [counts]
    have_unambig = len([c for c in counts if c.ambiguity == 'ua']) > 0

    lengths_lowres = decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor)

    if init is None:
        init = 'mds'

    # If initializing with true structure
    if isinstance(init, str) and init.lower() == 'true':
        if struct_true is None:
            raise ValueError("Attempting to initialize with true structure but"
                             " true structure was not provided.")
        if verbose:
            print('INITIALIZATION: initializing with true structure',
                  flush=True)
        init = struct_true

    if isinstance(init, (np.ndarray, jnp.ndarray, list)):
        if verbose:
            print('INITIALIZATION: 3D structure', flush=True)
        structures = _format_structures(init, mixture_coefs=mixture_coefs)
    elif not isinstance(init, str):
        raise ValueError(f"Initialization not understood, {type(init)=}."
                         "Options: np.ndarray, 'mds', or 'random'.")
    elif (init.lower() == "mds") and have_unambig:
        struct = _initialize_struct_mds(
            counts=counts, lengths=lengths, ploidy=ploidy, alpha=alpha,
            bias=bias, random_state=random_state,
            multiscale_factor=multiscale_factor, bias_per_hmlg=bias_per_hmlg,
            verbose=verbose)
        structures = [struct] * len(mixture_coefs)
    elif init.lower() in ("random", "rand", "mds"):
        if verbose:
            print('INITIALIZATION: random points', flush=True)
        struct = random_state.uniform(
            low=-1, high=1, size=(lengths_lowres.sum() * ploidy, 3))
        structures = [struct] * len(mixture_coefs)
    elif os.path.isfile(init):
        if verbose:
            print(f'INITIALIZATION: 3D structure, {init}', flush=True)
        structures = _format_structures(
            np.loadtxt(init), mixture_coefs=mixture_coefs)
    else:
        raise ValueError(f"Initialization not understood, {init=}."
                         "Options: np.ndarray, 'mds', or 'random'.")

    # Figure out what resolution structures are currently at
    struct_nbeads = set([s.shape[0] for s in structures])
    if len(struct_nbeads) > 1:
        raise ValueError("Initial structures have differing numbers of beads.")
    else:
        struct_nbeads = struct_nbeads.pop()
    multiscale_factor_current = 1
    while ploidy * decrease_lengths_res(
            lengths, multiscale_factor_current * 2).sum() >= struct_nbeads:
        multiscale_factor_current *= 2
    if np.log2(multiscale_factor_current) != int(
            np.log2(multiscale_factor_current)):
        raise ValueError(
            f"Initial structures contain {struct_nbeads} beads, they can't be"
            f" resized into {lengths_lowres.sum() * ploidy}-bead structures.")
    lengths_current = decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor_current)

    # Check format of structures, replace NaN
    structures = _format_structures(
        structures, lengths=lengths_current, ploidy=ploidy,
        mixture_coefs=mixture_coefs)
    structures = [_struct_replace_nan(
        struct, lengths=lengths_current, ploidy=ploidy,
        random_state=random_state) for struct in structures]

    # If needed, change resolution of structures to the current resolution
    for i in range(len(structures)):
        if multiscale_factor_current > multiscale_factor:
            resize_factor = int(multiscale_factor_current / multiscale_factor)
            if verbose:
                print('INITIALIZATION: increasing resolution of structure by'
                      f' a factor of {resize_factor}', flush=True)
            structures[i] = increase_struct_res(
                structures[i], multiscale_factor=resize_factor,
                lengths=lengths_lowres, ploidy=ploidy,
                random_state=random_state)

        elif multiscale_factor_current < multiscale_factor:
            resize_factor = int(multiscale_factor / multiscale_factor_current)
            if verbose:
                print('INITIALIZATION: decreasing resolution of structure by'
                      f' a factor of {resize_factor}', flush=True)
            structures[i] = decrease_struct_res(
                structures[i], multiscale_factor=resize_factor, lengths=lengths,
                ploidy=ploidy)

    # Check format of structures one more time, just for fun...
    structures = _format_structures(
        structures, lengths=lengths_lowres, ploidy=ploidy,
        mixture_coefs=mixture_coefs)

    return np.concatenate(structures)


def initialize(counts, lengths, init, ploidy, random_state=None, alpha=-3.,
               bias=None, multiscale_factor=1, reorienter=None,
               mixture_coefs=None, struct_true=None, bias_per_hmlg=False,
               verbose=False, mods=[]):
    """Initialize optimization.

    Create initialization for optimization. Structures can be initialized
    randomly, or via MDS2.

    Parameters
    ----------
    counts : list of pastis.counts.CountsMatrix instances
        Preprocessed counts data.
    lengths : array_like of int
        Number of beads per homolog of each chromosome.
    random_state : int or RandomState instance
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    init : str or array_like of float
        If array of float, this will be used for initialization. Structures
        will be re-sized to the appropriate resolution, and NaN beads will be
        linearly interpolated. If str, indicates the method of initalization:
        random ("random" or "rand") or MDS2 ("mds").
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    alpha : float, optional
        Biophysical parameter of the transfer function used in converting
        counts to wish distances.
    bias : array_like of float, optional
        Biases computed by ICE normalization.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.

    Returns
    -------
    array of float
        Initialization for inference.
    """

    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    if reorienter is not None and reorienter.reorient:
        if isinstance(init, np.ndarray):
            print('INITIALIZATION: inputted translation coordinates'
                  ' and/or rotation quaternions', flush=True)
            init_reorient = init.flatten()
            reorienter.check_X(init_reorient)
        else:
            print('INITIALIZATION: random', flush=True)
            lengths_lowres = decrease_lengths_res(
                lengths, multiscale_factor=multiscale_factor)
            init_reorient = []
            if reorienter.translate:
                init_reorient.append(random_state.uniform(
                    low=-1, high=1, size=lengths_lowres.size * 3 * (
                        1 + np.invert(reorienter.fix_homo))))
            if reorienter.rotate:
                init_reorient.append(random_state.uniform(
                    size=lengths_lowres.shape[0] * 4 * (
                        1 + np.invert(reorienter.fix_homo))))
            init_reorient = np.concatenate(init_reorient)
        return init_reorient
    else:
        struct_init = _initialize_struct(
            counts=counts, lengths=lengths, ploidy=ploidy, alpha=alpha,
            bias=bias, random_state=random_state, init=init,
            multiscale_factor=multiscale_factor, mixture_coefs=mixture_coefs,
            struct_true=struct_true, bias_per_hmlg=bias_per_hmlg,
            verbose=verbose)
        return struct_init
