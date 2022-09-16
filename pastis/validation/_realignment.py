import numpy as np
from scipy import linalg
from sklearn.metrics import euclidean_distances


def realignment_error(X, Y, error_type):
    if error_type.lower() == 'rmsd':
        return np.sqrt(((Y - X) ** 2.).sum() / len(Y))
    elif error_type.lower() == 'disterror':
        return np.sqrt((
            (euclidean_distances(Y) - euclidean_distances(X)) ** 2.).sum())
    else:
        raise ValueError(f"The error_type is '{error_type}', it must be"
                         " 'rmsd' or 'disterror'.")


def realign_structures(X, Y, rescale_X_to_Y=False, copy=True, error_type='rmsd',
                       verbose=False):
    """
    Realigns X to Y, checks mirror image, optionally rescales

    Parameters
    ----------
    X : ndarray (n, 3)
        First 3D structure

    Y : ndarray (n, 3)
        Second 3D structure

    rescale_X_to_Y : boolean, optional, default: False
        Whether to rescale X to Y or not.

    copy : boolean, optional, default: True
        Whether to copy the data or not

    verbose : boolean, optional, default: False
        The level of verbosity

    Returns
    -------
    X : ndarray (n, 3)
        Realigned 3D structure of X
    error : float
        The error between structures
    """
    if copy:
        X = X.copy()
        Y = Y.copy()

    mask = np.invert(np.isnan(X[:, 0]) | np.isnan(Y[:, 0]))

    if rescale_X_to_Y:
        # Realign structures without rescaling
        X -= X[mask].mean(axis=0)
        Y -= Y[mask].mean(axis=0)
        X = realign_structures(
            X, Y, rescale_X_to_Y=False, error_type=error_type,
            verbose=verbose)[0]

        # Rescale X
        if error_type.lower() == 'rmsd':
            rescale_factor = (Y[mask] * X[mask]).sum() / (X[mask] ** 2).sum()
        elif error_type.lower() == 'disterror':
            dis_X = euclidean_distances(X[mask])
            dis_Y = euclidean_distances(Y[mask])
            rescale_factor = (dis_Y * dis_X).sum() / (dis_X ** 2).sum()
        X *= rescale_factor

    # Center structures
    X -= X[mask].mean(axis=0)
    Y -= Y[mask].mean(axis=0)

    K = np.dot(Y[mask].T, X[mask])
    U, L, V = linalg.svd(K)
    V = V.T

    R = np.dot(V, U.T)
    if linalg.det(R) < 0:
        if verbose:
            print("Reflexion found", flush=True)
        V[:, -1] *= -1
        R = np.dot(V, U.T)
    X_fit = np.dot(X, R)

    error = realignment_error(Y[mask], X_fit[mask], error_type)

    # Check at the mirror
    X_mirror = X.copy()
    X_mirror[:, 0] = - X[:, 0]

    K = np.dot(Y[mask].T, X_mirror[mask])
    U, L, V = linalg.svd(K)
    V = V.T
    if linalg.det(V) < 0:
        V[:, -1] *= -1

    R_mirror = np.dot(V, U.T)
    X_mirror_fit = np.dot(X_mirror, R_mirror)
    error_mirror = realignment_error(Y[mask], X_mirror_fit[mask], error_type)

    if error <= error_mirror:
        best_X_fit = X_fit
        best_error = error
    else:
        if verbose:
            print("Reflexion is better", flush=True)
        best_X_fit = X_mirror_fit
        best_error = error_mirror

    return best_X_fit, best_error


def find_rotation(X, Y, copy=True):
    """
    Finds the rotation matrice C such that \|x - Q.T Y\| is minimum.

    Parameters
    ----------
    X : ndarray (n, 3)
        First 3D structure

    Y : ndarray (n, 3)
        Second 3D structure

    copy : boolean, optional, default: True
        Whether to copy the data or not

    Returns
    -------
    Y : ndarray (n, 3)
        Realigned 3D structure
    """
    if copy:
        Y = Y.copy()
        X = X.copy()
    mask = np.invert(np.isnan(X[:, 0]) | np.isnan(Y[:, 0]))
    K = np.dot(X[mask].T, Y[mask])
    U, L, V = np.linalg.svd(K)
    V = V.T

    t = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, np.linalg.det(np.dot(V, U.T))]])
    R = np.dot(V, np.dot(t, U.T))
    Y_fit = np.dot(Y, R)
    X_mean = X[mask].mean()
    Y_fit -= Y_fit[mask].mean() - X_mean
    error = ((X[mask] - Y_fit[mask]) ** 2).sum()

    # Check at the mirror
    Y_mirror = Y.copy()
    Y_mirror[:, 0] = - Y[:, 0]

    K = np.dot(X[mask].T, Y_mirror[mask])
    U, L, V = np.linalg.svd(K)
    V = V.T

    t = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, np.linalg.det(np.dot(V, U.T))]])
    R_ = np.dot(V, np.dot(t, U.T))
    Y_mirror_fit = np.dot(Y_mirror, R_)
    Y_mirror_fit -= Y_mirror[mask].mean() - X_mean
    error_mirror = ((X[mask] - Y_mirror_fit[mask]) ** 2).sum()
    return R


def distance_between_structures(X, Y):
    """
    Computes the distances per loci between structures

    Parameters
    ----------
    X : ndarray (n, l)
        First 3D structure

    Y : ndarray (n, l)
        Second 3D structure

    Returns
    -------
    distances : (n, )
        Distances between the 2 structures
    """
    if X.shape != Y.shape:
        raise ValueError("Shapes of the two matrices need to be the same")

    return np.sqrt(((X - Y) ** 2).sum(axis=1))
