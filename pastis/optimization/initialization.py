import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import numpy as np
import os

from .multiscale_optimization import increase_struct_res, decrease_struct_res
from .multiscale_optimization import decrease_lengths_res
from .multiscale_optimization import decrease_bias_res
from .mds import estimate_X
from .utils_poisson import find_beads_to_remove
from .utils_poisson import _struct_replace_nan, _format_structures


def _initialize_struct_mds(counts, lengths, ploidy, alpha, bias, random_state,
                           multiscale_factor=1, verbose=True):
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

    bias = decrease_bias_res(
        bias, multiscale_factor=multiscale_factor, lengths=lengths)
    if bias is not None and not np.all(bias == 1):
        bias_tiled = np.tile(bias, ploidy)
    else:
        bias_tiled = None

    ua_counts = ua_counts.tocoo()
    if bias is not None:
        ua_counts = ua_counts.astype(float)

    struct = estimate_X(
        ua_counts, alpha=alpha, beta=ua_beta,
        verbose=False, use_zero_entries=False, precompute_distances='auto',
        bias=bias_tiled, random_state=random_state, type="MDS2", factr=1e12,
        maxiter=10000, ini=None)

    struct = struct.reshape(-1, 3)
    struct_nan = find_beads_to_remove(
        counts, lengths=lengths, ploidy=ploidy,
        multiscale_factor=multiscale_factor)
    struct[struct_nan] = np.nan

    return struct


def _initialize_struct(counts, lengths, ploidy, alpha, bias, random_state,
                       init='mds', multiscale_factor=1,
                       mixture_coefs=None, verbose=True):
    """Initialize structure, randomly or via MDS of unambig counts.
    """

    if mixture_coefs is None:
        mixture_coefs = [1.]

    if not isinstance(counts, (list, tuple)):
        counts = [counts]
    have_unambig = len([c for c in counts if c.ambiguity == 'ua']) > 0

    lengths_lowres = decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor)

    if init is None:
        init = 'mds'

    if isinstance(init, np.ndarray) or isinstance(init, list):
        if verbose:
            print('INITIALIZATION: 3D structure', flush=True)
        structures = _format_structures(init, mixture_coefs=mixture_coefs)
    elif isinstance(init, str) and (init.lower() == "mds") and have_unambig:
        struct = _initialize_struct_mds(
            counts=counts, lengths=lengths, ploidy=ploidy, alpha=alpha,
            bias=bias, random_state=random_state,
            multiscale_factor=multiscale_factor, verbose=verbose)
        structures = [struct] * len(mixture_coefs)
    elif isinstance(init, str) and (init.lower() in ("random", "rand", "mds")):
        if verbose:
            print('INITIALIZATION: random points', flush=True)
        structures = [random_state.uniform(
            low=-1, high=1, size=(int(
                lengths_lowres.sum() * ploidy), 3)) for coef in mixture_coefs]
    elif isinstance(init, str) and os.path.exists(init):
        if verbose:
            print('INITIALIZATION: 3D structure, %s' % init, flush=True)
        structures = _format_structures(
            np.loadtxt(init), mixture_coefs=mixture_coefs)
    else:
        raise ValueError("Initialization method not understood.")

    struct_length = set([s.shape[0] for s in structures])
    if len(struct_length) > 1:
        raise ValueError("Initial structures are of different shapes")
    else:
        struct_length = struct_length.pop()
    # multiscale_factor_init = int(np.ceil(
    #     lengths.sum() * ploidy / struct_length)) # TODO remove junk
    # multiscale_factor_init = int(np.ceil(
    #     np.tile(lengths, ploidy) / struct_length).sum())
    multiscale_factor_init = 1
    while ploidy * decrease_lengths_res(
            lengths, multiscale_factor_init * 2).sum() >= struct_length:
        multiscale_factor_init *= 2
    lengths_init = decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor_init)

    structures = _format_structures(
        structures, lengths=lengths_init, ploidy=ploidy,
        mixture_coefs=mixture_coefs)
    structures = [_struct_replace_nan(
        struct, lengths=lengths_init, ploidy=ploidy,
        random_state=random_state) for struct in structures]

    for i in range(len(structures)):
        if struct_length < int(lengths_lowres.sum() * ploidy):
            resize_factor = int(np.ceil(
                lengths_lowres.sum() * ploidy / struct_length))
            if verbose:
                print('INITIALIZATION: increasing resolution of structure by'
                      ' %d' % resize_factor, flush=True)
            structures[i] = increase_struct_res(
                structures[i], multiscale_factor=resize_factor,
                lengths=lengths_lowres, ploidy=ploidy,
                random_state=random_state)

        elif struct_length > lengths_lowres.sum() * ploidy:
            resize_factor = int(np.ceil(
                struct_length / (lengths_lowres.sum() * ploidy)))
            if verbose:
                print('INITIALIZATION: decreasing resolution of structure by'
                      ' %d' % resize_factor, flush=True)
            structures[i] = decrease_struct_res(
                structures[i], multiscale_factor=resize_factor, lengths=lengths,
                ploidy=ploidy)

    structures = _format_structures(
        structures, lengths=lengths_lowres, ploidy=ploidy,
        mixture_coefs=mixture_coefs)

    return np.concatenate(structures)


def initialize(counts, lengths, init, ploidy, random_state=None, alpha=-3.,
               bias=None, multiscale_factor=1, reorienter=None,
               mixture_coefs=None, verbose=False, mods=[]):
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

    lengths_lowres = decrease_lengths_res(
        lengths, multiscale_factor=multiscale_factor)

    if reorienter is not None and reorienter.reorient:
        if isinstance(init, np.ndarray):
            print('INITIALIZATION: inputted translation coordinates'
                  ' and/or rotation quaternions', flush=True)
            init_reorient = init.flatten()
            reorienter.check_X(init_reorient)
        else:
            print('INITIALIZATION: random', flush=True)
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
            verbose=verbose)
        return struct_init
