import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

import numpy as np
from timeit import default_timer as timer
from datetime import timedelta
import os

from .utils_poisson import _setup_jax
_setup_jax()
import jax.numpy as jnp

from .utils_poisson import find_beads_to_remove
from .multiscale_optimization import decrease_lengths_res, decrease_struct_res
from .multiscale_optimization import get_epsilon_from_struct


class Callback(object):
    """An object that adds functionality during optimization.

    A callback is a function or group of functions that can be executed during
    optimization. A callback can be called at three stages -- the
    beginning of optimization, at the end of each iteration, and at
    the end of optimization. Users can define any functions that they wish in
    the corresponding functions.Compute objective constraints.

    Parameters
    ----------
    lengths : array_like of int
        Number of beads per homolog of each chromosome at full resolution.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    counts : list of CountsMatrix subclass instances
        Preprocessed counts data.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
    history : dict of list, optional
        Previously generated history logs, to be added to during this
        optimization.
    frequency : int or dict, optional
        Frequency of iterations at which operations are performed. Each key
        indicates an operation, options: "history" (logs history), "print"
        (prints time and current objective value), "save" (saves current
        structure to file).
    directory : str, optional
        Directory in which to save structures generated during optimization.
    seed : int, optional
        Random seed used when generating the starting point in the
        optimization.
    struct_true : array of float, optional
        True structure, to be used by `analysis_function`.
    alpha_true : array of float, optional
        True alpha, to be used by `analysis_function`.
    verbose : bool, optional
        Verbosity.

    Attributes
    ----------
    lengths : array of int
        Number of beads per homolog of each chromosome at full the current
        resolution (indicated by `multiscale_factor`). TODO update
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    multiscale_factor : int
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
    struct_nan : array of int
        Beads that should be removed (set to NaN) in the structure.
    frequency : dict
        Frequency of iterations at which operations are performed. Each key
        indicates an operation, options: "history" (logs history), "print"
        (prints time and current objective value), "save" (saves current
        structure to file).
    directory : str
        Directory in which to save structures generated during optimization.
    seed : int, optional
        Random seed used when generating the starting point in the
        optimization.
    struct_true : array of float or None
        True structure, to be used by `analysis_function`.
    alpha_true : array of float or None
        True alpha, to be used by `analysis_function`.
    verbose : bool
        Verbosity.
    history : dict of list
        History logs generated during optimization. By default, history includes
        the following information and keys, respectively: iteration ("iter"),
        alpha ("alpha"), multiscale_factor ("multiscale_factor"),
        current iteration of alpha/structure optimization ("alpha_loop"),
        relative orientation of each chromosome ("opt_type").
    opt_type : {"structure", "alpha", "chrom_reorient", None}
        Type of optimization being performed. Options: 3D chromatin structure
        ("structure"), alpha ("alpha"), relative orientation of each chromosome
        ("chrom_reorient").
    alpha_loop : int or None
        Current iteration of alpha/structure optimization.
    iter : int
        Current iter (iteration) of optimization.
    time : str
        Time since optimization began.
    structures : list of array of float or None
        Current 3D chromatin structure(s).
    alpha : float or None
        Current biophysical parameter of the transfer function used in
        converting counts to wish distances.
    Xi : float or array of float or None
        Current values being optimized (structure, alpha, or chromosome
        orientation).
    time_start : timeit.default_timer instance
        Time at which optimization began.
    """

    def __init__(self, lengths, ploidy, counts=None, bias=None, beta_init=None,
                 multiscale_factor=1, multiscale_reform=False,
                 history=None, analysis_function=None, frequency=None,
                 on_optimization_begin=None, directory=None, seed=None, struct_true=None,
                 alpha_true=None, constraints=None, simple_diploid=False, mixture_coefs=None,
                 verbose=False, mods=[]):
        self.ploidy = ploidy
        self.multiscale_factor = multiscale_factor
        self.lengths = np.array(lengths, copy=False, ndmin=1, dtype=int).ravel()
        self.lengths_lowres = decrease_lengths_res(
            lengths, multiscale_factor=multiscale_factor)
        self.bias = bias
        self.beta_init = beta_init
        self.counts = counts
        if counts is None:
            self.struct_nan = np.array([])
        else:
            self.struct_nan = find_beads_to_remove(
                counts, lengths=lengths, ploidy=ploidy,
                multiscale_factor=multiscale_factor)
        self.constraints = constraints
        self.multiscale_reform = multiscale_reform
        self.mixture_coefs = mixture_coefs
        self.mods = mods

        if frequency is None or isinstance(frequency, int):
            self.frequency = {
                'print': frequency, 'history': frequency, 'save': frequency}
            if frequency is None:
                self.frequency['print'] = 100
        else:
            if not isinstance(frequency, dict) or any(
                    [k not in ('print', 'history', 'save') for k in frequency.keys()]):
                raise ValueError("Callback frequency must be None, int, or dict"
                                 " with keys = (print, history, save).")
            self.frequency = {'print': None, 'history': None, 'save': None}
            for k, v in frequency.items():
                self.frequency[k] = v

        self.on_optimization_begin_ = on_optimization_begin
        self.analysis_function = analysis_function

        self.directory = '' if directory is None else directory
        self.seed = seed
        self.verbose = verbose

        self.epsilon_true = None
        if struct_true is not None:
            struct_true = struct_true.reshape(-1, 3)
            if multiscale_factor != 1 and multiscale_reform:
                self.epsilon_true = get_epsilon_from_struct(
                    struct_true, lengths=lengths,
                    ploidy=(2 if simple_diploid else ploidy),
                    multiscale_factor=multiscale_factor, verbose=False)
                if verbose:
                    print(f"True epsilon ({multiscale_factor}x):"
                          f" {self.epsilon_true:.3g}", flush=True)
            if simple_diploid:
                struct_true = np.nanmean(  # FIXME is this even correct?
                    [struct_true[:int(struct_true.shape[0] / 2)],
                     struct_true[int(struct_true.shape[0] / 2):]], axis=0)
            if struct_true.shape[0] > self.lengths_lowres.sum() * ploidy:
                struct_true = decrease_struct_res(
                    struct_true, multiscale_factor=multiscale_factor,
                    lengths=lengths, ploidy=ploidy)
            self.struct_true = struct_true
            if isinstance(self.struct_true, jnp.ndarray):
                self.struct_true = self.struct_true._value
        else:
            self.struct_true = None
        self.alpha_true = alpha_true

        if history is None:
            self.history = {}
        elif isinstance(history, dict) and all(
                [isinstance(v, list) for v in history.values()]):
            self.history = history
        else:
            raise ValueError("History must be dictionary of lists")

        self.opt_type = None
        self.alpha_loop = None
        self.iter = -1
        self.seconds = 0
        self.time = '0:00:00.0'
        self.time_start = None
        self.optimization_complete = False

        self.structures_ = None
        self.alpha_ = None
        self.epsilon_ = None
        self.X_ = None

    def on_optimization_begin(self, opt_type=None, alpha_loop=None):
        """Functionality to add to the beginning of optimization.

        This method will be called at the beginning of the optimization
        procedure.

        Parameters
        ----------
        opt_type : {"structure", "alpha", "chrom_reorient", None}
            Type of optimization being performed. Options: 3D chromatin
            structure ("structure"), alpha ("alpha"), relative orientation of
            each chromosome ("chrom_reorient").
        alpha_loop : int
            Current iteration of alpha/structure optimization.
        """

        if opt_type is None:
            self.opt_type = 'structure'
        else:
            self.opt_type = opt_type
        self.alpha_loop = alpha_loop
        self.iter = -1
        self.seconds = 0
        self.time = '0:00:00.0'
        self.optimization_complete = False

        self.structures_ = None
        self.alpha_ = None
        self.epsilon_ = None
        self.X_ = None

        if self.on_optimization_begin_ is not None:
            res = self.on_optimization_begin_(self, **kwargs)
        else:
            res = None
        self.time_start = timer()
        return res

    def on_iter_end(self, obj_logs, structures, alpha, Xi, epsilon=None):
        """Functionality to add to the end of each iter.

        This method will be called at the end of each iter during the
        iterative optimization procedure.

        Parameters
        ----------
        obj_logs : dict of float
            Current objective function. Each component of the objective is
            indicated by a separate dictionary item.
        structures : list of array of float
            Current 3D chromatin structure(s).
        alpha : float
            Current biophysical parameter of the transfer function used in
            converting counts to wish distances.
        Xi : float or array of float
            Current values being optimized (structure, alpha, or chromosome
            orientation).
        """

        self.iter += 1
        self.seconds = round(timer() - self.time_start, 1)
        current_time = str(timedelta(seconds=self.seconds)).split('.')
        if len(current_time) == 1:
            self.time = current_time[0] + '.0'
        else:
            self.time = current_time[
                0] + str(float('0.' + current_time[1])).lstrip('0')

        self.obj = obj_logs
        if self.obj is not None:
            for k, v in self.obj.items():
                if isinstance(v, jnp.ndarray):
                    self.obj[k] = v._value

        if isinstance(Xi, jnp.ndarray):
            Xi = Xi._value
        self.X_ = Xi

        self.structures_ = []
        for struct in structures:
            if isinstance(struct, jnp.ndarray):
                struct = struct._value
            self.structures_.append(struct)

        if isinstance(alpha, jnp.ndarray):
            alpha = alpha._value
        if isinstance(alpha, np.ndarray):
            if alpha.size > 1:
                raise ValueError("Alpha must be a float.")
            alpha = float(alpha)
        self.alpha_ = alpha

        if isinstance(epsilon, jnp.ndarray):
            epsilon = epsilon._value
        if isinstance(epsilon, np.ndarray):
            if epsilon.size > 1:
                raise ValueError("Alpha must be a float.")
            epsilon = float(epsilon)
        self.epsilon_ = epsilon

        self._print()
        self._log_history()
        self._save_X()

    def on_optimization_end(self):
        """Functionality to add to the end of optimization.

        This method will be called at the end of the optimization procedure."""

        self.optimization_complete = True
        self._print()
        self._log_history()

    def _check_frequency(self, frequency):
        if frequency is not None and frequency > 0:
            if not self.optimization_complete and self.iter % frequency == 0:
                return True
            elif self.optimization_complete and self.iter % frequency != 0:
                return True
        return False

    def _print(self):
        """Prints loss every given number of iters."""

        if not self._check_frequency(self.frequency['print']):
            return

        info_dict = {'At iterate': ' ' * (6 - len(str(self.iter))) + str(
            self.iter), 'f= ': '%.6g' % self.obj['obj'],
            'time= ': self.time}
        if self.epsilon_ is not None:
            if np.array(self.epsilon_).size == 1:
                info_dict['epsilon= '] = f"{self.epsilon_:.3g}"
            elif self.epsilon_.size == self.X_.shape[0]:
                info_dict['epsilon= '] = f"{self.epsilon_.mean():.3g}"
        print('\t\t'.join([f'{k}{v}' for k, v in info_dict.items()]) + '\n',
              flush=True)

    def _save_X(self):
        """This will save the model to disk every given number of iters."""

        if not self._check_frequency(self.frequency['save']):
            return

        X = self.X_
        if isinstance(X, list):
            X = np.concatenate(X)

        if self.seed is None:
            seed_str = ''
        else:
            seed_str = f'.{self.seed:03d}'
        filename = os.path.join(
            self.directory,
            f"{self.opt_type}_inferred{seed_str}.iter{self.iter:07d}.coords")
        if self.verbose:
            print(f"[{self.iter}] Saving model checkpoint to {filename}",
                  flush=True)
        np.savetxt(filename, X)

    def _log_history(self):
        """Keeps a log of the loss and other values."""

        if not self._check_frequency(self.frequency['history']):
            return

        to_log = [('iter', self.iter), ('alpha', self.alpha_),
                  ('alpha_loop', self.alpha_loop),
                  ('opt_type', self.opt_type),
                  ('multiscale_factor', self.multiscale_factor),
                  ('seconds', self.seconds),
                  ('epsilon', self.epsilon_)]
        to_log.extend(list(self.obj.items()))

        if self.analysis_function is not None:
            to_log.extend(self.analysis_function(self).items())

        for k, v in to_log:
            if k in self.history:
                self.history[k].append(v)
            else:
                self.history[k] = [v]
