from __future__ import division, print_function, absolute_import
import numpy as np
from scipy import sparse
from scipy import optimize
from sklearn.utils import check_random_state
from . import poisson_model
from . import mds
from topsy.metrics.generate_metrics import simulated_vs_inferred

# Set up for callback function
n_iter = 0
iter_obj = iter_grad = np.nan
iter_details = {'iter': [], 'obj': [], 'grad': [], 'RMSD_whole':[], 'distanceError_whole':[],
                'RMSD_chr':[], 'distanceError_chr':[]}

def poisson_obj(X, counts, alpha=-3., beta=1., bias=None,
                use_zero_counts=False, cst=0):
    if bias is None:
        bias = np.ones((counts.shape[0], 1))

    if sparse.issparse(counts):
        return _poisson_obj_sparse(
            X, counts, beta=beta, bias=bias,
            alpha=alpha,
            use_zero_counts=use_zero_counts, cst=cst)
    else:
        raise NotImplementedError(
            "Poisson model is not implemented for dense")


def _poisson_obj_sparse(X, counts, alpha=-3., beta=1., bias=None,
                        use_zero_counts=False, cst=0):

    if bias is None:
        bias = np.ones((X.shape[0], ))
    bias = bias.flatten()
    dis = np.sqrt(((X[counts.row] - X[counts.col])**2).sum(axis=1))
    fdis = bias[counts.row] * bias[counts.col] * beta * dis ** alpha

    obj = fdis.sum() - (counts.data * np.log(fdis)).sum()
    if np.isnan(obj):
        raise ValueError("Objective function is nan")

    global iter_obj
    iter_obj = obj

    return obj


def poisson_gradient(X, counts, alpha=-3, beta=1, bias=None,
                     use_zero_counts=False):
    if bias is None:
        bias = np.ones((counts.shape[0], 1))

    if sparse.issparse(counts):
        return _poisson_gradient_sparse(
            X, counts, beta=beta, bias=bias,
            alpha=alpha)
    else:
        raise NotImplementedError(
            "Poisson model is not implemented for dense")


def _poisson_gradient_sparse(X, counts, alpha=-3, beta=1, bias=None):
    if bias is None:
        bias = np.ones((counts.shape[0], 1))

    bias = bias.flatten()
    dis = np.sqrt(((X[counts.row] - X[counts.col])**2).sum(axis=1))
    fdis = bias[counts.row] * bias[counts.col] * beta * dis ** alpha

    diff = X[counts.row] - X[counts.col]

    grad = - ((counts.data / fdis - 1) * fdis * alpha /
              (dis ** 2))[:, np.newaxis] * diff

    grad_ = np.zeros(X.shape)

    for i in range(X.shape[0]):
        grad_[i] += grad[counts.row == i].sum(axis=0)
        grad_[i] -= grad[counts.col == i].sum(axis=0)

    global iter_grad
    iter_grad = grad_.max()

    return grad_


def eval_f(x, user_data=None):
    n, counts, alpha, beta, bias, use_zero_counts = user_data
    x = x.reshape((n, 3))
    obj = poisson_obj(x, counts, alpha=alpha, beta=beta, bias=bias)
    x = x.flatten()
    return obj


def eval_grad_f(x, user_data=None):
    n, counts, alpha, beta, bias, use_zero_counts = user_data
    x = x.reshape((n, 3))
    grad = poisson_gradient(x, counts, alpha=alpha,
                            beta=beta, bias=bias)
    x = x.flatten()
    return grad.flatten()


def estimate_X(counts, alpha=-3., beta=1.,
               ini=None, bias=None,
               random_state=None, maxiter=10000, verbose=0,
               X_true = None, lengths = None,
               use_callback=False):
    """
    Estimate the parameters of g

    Parameters
    ----------
    counts : sparse scipy matrix (n, n)

    alpha : float, optional, default: -3
        counts-to-distances mapping coefficient

    beta : float, optional, default: 1
        counts-to-distnances scaling coefficient

    init : ndarray (n, 3), optional, default: None
        initialization point

    bias : ndarray (n, 1), optional, default: None
        bias vector. If None, no bias will be applied to the model

    random_state : {RandomState, int, None}, optional, default: None
        random state object, or seed, or None.

    maxiter : int, optional, default: 10000
        Maximum number of iteration

    verbose : int, optional, default: 0
        verbosity

    Returns
    ------
    X : 3D structure

    """
    n = counts.shape[0]

    if not sparse.isspmatrix_coo(counts):
        counts = sparse.coo_matrix(counts)

    counts.setdiag(0)
    counts.eliminate_zeros()

    random_state = check_random_state(random_state)
    if ini is None:
        ini = 1 - 2 * random_state.rand(n * 3)
    else:
        ini = np.array(ini)

    data = (n, counts, alpha, beta, bias,
            False)

    def callback_fxn(Xi):
        global n_iter
        global iter_details
        global iter_obj
        global iter_grad
        iter_details['iter'].append(n_iter)
        iter_details['obj'].append(iter_obj)
        iter_details['grad'].append(iter_grad)
        if X_true is not None:
            metrics = simulated_vs_inferred(X_true, Xi.reshape(-1, 3), lengths, verbose=0)
            iter_details['RMSD_whole'].append(metrics['RMSD']['Whole structure'])
            iter_details['distanceError_whole'].append(metrics['distanceError']['Whole structure'])
            iter_details['RMSD_chr'].append(metrics['RMSD']['Intra-chromosomal'])
            iter_details['distanceError_chr'].append(metrics['distanceError']['Intra-chromosomal'])
            if len(np.unique(mapping)) != len(mapping):
                iter_details['RMSD_homo'].append(metrics['RMSD']['Homologous pairs'])
                iter_details['distanceError_homo'].append(metrics['distanceError']['Homologous pairs'])
        n_iter += 1
    if not use_callback:
        callback_fxn = None

    # Generate metrics for ini
    callback_fxn(ini)

    results = optimize.fmin_l_bfgs_b(
        eval_f,
        ini.flatten(),
        eval_grad_f,
        (data, ),
        iprint=verbose,
        maxiter=maxiter,
        callback=callback_fxn)
    results = results[0].reshape(-1, 3)

    global iter_details
    iter_details = {k: v for k, v in iter_details.items() if len(v) != 0}

    return results, iter_details


class PM1(object):
    """
    """
    def __init__(self, alpha=-3., beta=1.,
                 max_iter=5000, random_state=None, n_init=1, n_jobs=1,
                 init="MDS2", verbose=False, bias=None):
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.random_state = check_random_state(random_state)
        self.n_init = n_init
        self.bias = bias
        self.n_jobs = n_jobs
        self.init = init
        self.verbose = verbose

    def fit(self, counts, lengths=None, X_true=None, use_callback=False):
        """

        """
        if not sparse.isspmatrix_coo(counts):
            counts = sparse.coo_matrix(counts)
        if not sparse.issparse(counts):
            counts[np.isnan(counts)] = 0
        if self.init == "MDS2":
            if self.verbose:
                print("Initialing with MDS2")
            X = mds.estimate_X(counts, alpha=self.alpha,
                               beta=self.beta,
                               bias=self.bias,
                               random_state=self.random_state,
                               maxiter=self.max_iter,
                               verbose=self.verbose)
        else:
            X = self.init
        X, iter_details = estimate_X(counts,
                       alpha=self.alpha,
                       beta=self.beta,
                       ini=X,
                       bias=self.bias,
                       verbose=self.verbose,
                       random_state=self.random_state,
                       maxiter=self.max_iter,
                       lengths=lengths,
                       X_true=X_true,
                       use_callback=use_callback)
        if use_callback:
            self.iter_details_ = iter_details
        return X


class PM2(object):
    """
    """
    def __init__(self, alpha=-3., beta=1.,
                 max_iter=5000, max_iter_outer_loop=5,
                 random_state=None, n_init=1, n_jobs=1,
                 bias=None,
                 init="MDS2", verbose=False):
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.random_state = check_random_state(random_state)
        self.n_init = n_init
        self.n_jobs = n_jobs
        self.init = init
        self.max_iter_outer_loop = max_iter_outer_loop
        self.verbose = verbose
        self.bias = bias

    def fit(self, counts, lengths=None, X_true=None, use_callback=False):
        """

        """

        if not sparse.isspmatrix_coo(counts):
            counts = sparse.coo_matrix(counts)
        counts.setdiag(0)
        counts.eliminate_zeros()

        if self.init == "MDS2":
            X = mds.estimate_X(counts, alpha=self.alpha,
                               beta=self.beta,
                               ini="random",
                               verbose=self.verbose,
                               bias=self.bias,
                               random_state=self.random_state,
                               maxiter=self.max_iter)
        elif self.init == "random":
            X = self.init
        else:
            raise ValueError("Unknown initialization strategy")

        self.alpha_ = self.alpha
        self.beta_ = self.beta
        for it in range(self.max_iter_outer_loop):
            self.alpha_, self.beta_ = poisson_model.estimate_alpha_beta(
                counts,
                X, bias=self.bias, ini=[self.alpha_, self.beta_],
                verbose=self.verbose,
                random_state=self.random_state)
            print(self.alpha_, self.beta_)
            X_, iter_details = estimate_X(counts,
                            alpha=self.alpha_,
                            beta=self.beta_,
                            ini=X,
                            verbose=self.verbose,
                            bias=self.bias,
                            random_state=self.random_state,
                            maxiter=self.max_iter,
                            lengths=lengths,
                            X_true=X_true,
                            use_callback=use_callback)
        if use_callback:
            self.iter_details_ = iter_details
        return X_
