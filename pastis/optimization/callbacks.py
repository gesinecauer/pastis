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
    log : dict of list, optional
        Previously generated logs, to be added to during this
        optimization.
    XXX : int or dict, optional
        Frequency of iterations at which operations are performed. Each key
        indicates an operation, options: "log" (adds to logs), "print"
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
    XXX : dict
        Frequency of iterations at which operations are performed. Each key
        indicates an operation, options: "log" (adds to logs), "print"
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
    log : dict of list
        Logs generated during optimization. By default, log includes
        the following information and keys, respectively: iteration ("iter"),
        alpha ("alpha"), multiscale_factor ("multiscale_factor"),
        current iteration of alpha/structure optimization ("alpha_loop"),
        relative orientation of each chromosome ("inferring").
    inferring : {"structure", "alpha", "chrom_reorient", None}
        Type of optimization being performed. Options: 3D chromatin structure
        ("structure"), alpha ("alpha"), relative orientation of each chromosome
        ("chrom_reorient").
    alpha_loop : int or None
        Current iteration of alpha/structure optimization.
    iter : int
        Current iter (iteration) of optimization.
    time : str
        Time since optimization began.
    structures_ : list of array of float or None
        Current 3D chromatin structure(s).
    alpha_ : float or None
        Current biophysical parameter of the transfer function used in
        converting counts to wish distances.
    X_ : float or array of float or None
        Current values being optimized (structure, alpha, or chromosome
        orientation).
    time_start : timeit.default_timer instance
        Time at which optimization began.
    """

    def __init__(self, lengths, ploidy, counts=None, bias=None, beta_init=None,
                 multiscale_factor=1, multiscale_reform=False,
                 log=None, analysis_function=None, print_freq=100,
                 log_freq=1000, save_freq=None, directory=None, seed=None,
                 on_optimization_begin=None, struct_true=None,
                 alpha_true=None, constraints=None, fullres_struct_nan=None,
                 reorienter=None, mixture_coefs=None, verbose=False, mods=[]):
        self.ploidy = ploidy
        self.multiscale_factor = multiscale_factor
        self.lengths = np.array(lengths, copy=None, ndmin=1, dtype=int).ravel()
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
        self.reorienter = reorienter
        self.mods = mods

        self.on_optimization_begin_ = on_optimization_begin
        self.analysis_function = analysis_function

        self.print_freq = None if (not print_freq) else print_freq
        self.log_freq = None if (not log_freq) else log_freq
        self.save_freq = None if (not save_freq) else save_freq
        self.directory = '' if directory is None else directory
        self.seed = seed
        self.verbose = verbose

        self.epsilon_true = None
        if struct_true is not None:
            self.struct_true = [x.reshape(-1, 3) for x in struct_true]
            struct_true_fullres = [x.copy() for x in self.struct_true]

            if multiscale_factor != 1:
                self.epsilon_true = [get_epsilon_from_struct(
                    x, lengths=lengths, ploidy=ploidy,
                    multiscale_factor=multiscale_factor,
                    verbose=False)[0] for x in struct_true_fullres]
                if verbose and multiscale_reform:
                    print(f"True epsilon ({multiscale_factor}x):"
                          f" {np.mean(self.epsilon_true):.3g}", flush=True)

            for i in range(len(self.struct_true)):
                self.struct_true[i] = decrease_struct_res(
                    struct_true_fullres[i], multiscale_factor=multiscale_factor,
                    lengths=lengths, ploidy=ploidy,
                    fullres_struct_nan=fullres_struct_nan)
                if multiscale_factor > 1:
                    struct_nan_mask = np.isnan(self.struct_true[i][:, 0])
                    self.struct_true[i][struct_nan_mask] = decrease_struct_res(
                        struct_true_fullres[i], multiscale_factor=multiscale_factor,
                        lengths=lengths, ploidy=ploidy)[struct_nan_mask]
                if isinstance(self.struct_true[i], jnp.ndarray):
                    self.struct_true[i] = self.struct_true[i]._value
        else:
            self.struct_true = None
        self.alpha_true = alpha_true

        if log is None:
            self.log = {}
        elif isinstance(log, dict) and all(
                [isinstance(v, list) for v in log.values()]):
            self.log = log
        else:
            raise ValueError("log must be dictionary of lists")

        self.inferring = None
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

    def on_optimization_begin(self, inferring=None, alpha_loop=None):
        """Functionality to add to the beginning of optimization.

        This method will be called at the beginning of the optimization
        procedure.

        Parameters
        ----------
        inferring : {"structure", "alpha", "chrom_reorient", None}
            Type of optimization being performed. Options: 3D chromatin
            structure ("structure"), alpha ("alpha"), relative orientation of
            each chromosome ("chrom_reorient").
        alpha_loop : int
            Current iteration of alpha/structure optimization.
        """

        if inferring is None:
            inferring = 'structure'
        if self.reorienter is not None and self.reorienter.reorient:
            inferring = f'{inferring}.chrom_reorient'
        self.inferring = inferring

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
            res = self.on_optimization_begin_(self)
        else:
            res = None
        self.time_start = timer()
        return res

    def on_iter_end(self, obj_logs, structures, alpha, X, epsilon=None):
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
        X : float or array of float
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

        if 'exit_after_i5' in self.mods and self.iter >= 5:
            exit(0)

        self.obj = obj_logs
        if self.obj is not None:
            for k, v in self.obj.items():
                if isinstance(v, jnp.ndarray):
                    self.obj[k] = v._value

        if isinstance(X, jnp.ndarray):
            X = X._value
        self.X_ = X

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
                raise ValueError("Epsilon must be a float.")
            epsilon = float(epsilon)
        self.epsilon_ = epsilon

        self._print()
        self._add_to_log()
        self._save_X()

    def on_optimization_end(self):
        """Functionality to add to the end of optimization.

        This method will be called at the end of the optimization procedure."""

        self.optimization_complete = True
        self._print()
        self._add_to_log()

    def _check_frequency(self, frequency):
        if frequency is not None and frequency > 0:
            if not self.optimization_complete and self.iter % frequency == 0:
                return True
            elif self.optimization_complete and self.iter % frequency != 0:
                return True
        return False

    def _print(self):
        """Prints loss every given number of iters."""

        if not self._check_frequency(self.print_freq):
            return

        info_dict = {'At iterate': ' ' * (6 - len(str(self.iter))) + str(
            self.iter), 'f= ': '%.6g' % self.obj['obj'],  # TODO fix this, it's junk
            'time= ': self.time}
        if self.epsilon_ is not None:
            info_dict['epsilon= '] = f"{self.epsilon_:.3g}"
        if self.iter == 0:
            print(flush=True)
        print('\t\t'.join([f'{k}{v}' for k, v in info_dict.items()]) + '\n',
              flush=True)

    def _save_X(self):
        """This will save the model to disk every given number of iters."""

        if not self._check_frequency(self.save_freq):
            return

        X = self.X_
        if isinstance(X, list):
            X = np.concatenate(X)

        if self.seed is None:
            seed_str = ''
        else:
            seed_str = f'.{self.seed:03d}'
        filename = os.path.join(
            self.directory, "in_progress",
            f"{self.inferring}_inferred{seed_str}.iter{self.iter}.coords")
        os.path.makedirs(self.directory, exist_ok=True)
        if self.verbose:
            print(f"[{self.iter}] Saving model checkpoint to {filename}",
                  flush=True)
        np.savetxt(filename, X)

    def _add_to_log(self):
        """Keeps a log of the loss and other values."""

        if not self._check_frequency(self.log_freq):
            return

        to_log = [('iter', self.iter), ('alpha', self.alpha_),
                  ('alpha_loop', self.alpha_loop),
                  ('inferring', self.inferring),
                  ('multiscale_factor', self.multiscale_factor),
                  ('seconds', self.seconds),
                  ('epsilon', self.epsilon_)]
        to_log.extend(list(self.obj.items()))

        if self.analysis_function is not None:
            to_log.extend(self.analysis_function(self).items())

        for k, v in to_log:
            if k in self.log:
                self.log[k].append(v)
            else:
                self.log[k] = [v]

        to_log = dict(to_log)
        for k in self.log.keys():
            if k not in to_log.keys():
                self.log[k].append(None)
