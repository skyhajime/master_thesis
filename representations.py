import numpy as np
import bases
import eigensolvers
from functools import partial
from scipy.linalg import schur


def _get_GA(X, basis):
    assert X.ndim == 2
    M, _ = X.shape
    G = sum(np.conjugate(basis(X[m])).T @ basis(X[m]) for m in range(M-1)) / (M-1)
    A = sum(np.conjugate(basis(X[m])).T @ basis(X[m+1]) for m in range(M-1)) / (M-1)
    return G, A

def EDMD_matrix_representation(X, basis=partial(bases.fourier_basis, degree=5)):
    """approximate matrix representation of Koopman operator

    Args:
        X (ndarray): shape (M, nx)
        degree (int): degree of basis function
    """
    G, A = _get_GA(X, basis)
    K = np.linalg.lstsq(G, A, rcond=1e-9)[0]
    D, V = eigensolvers.arnoldi_iteration(K)
    return K, V, D


def mpEDMD_matrix_representation(X, basis=partial(bases.fourier_basis, degree=5)):
    """approximate matrix representation of Koopman operator

    Args:
        X (ndarray): shape (M, nx)
        degree (int): degree of basis function
    """
    G, A = _get_GA(X, basis)

    DG, VG = np.linalg.eig(G)
    Gsqrt = VG @ np.diag(np.sqrt(DG)) @ np.conjugate(VG)
    Gsqrt_inv = VG @ np.diag(np.sqrt(1.0/DG)) @ np.conjugate(VG)
    U,S,V = np.linalg.svd(Gsqrt_inv @ np.conjugate(A) @ Gsqrt_inv)
    _mpD, _mpV = schur(V @ np.conjugate(U), output='complex')
    mpK = Gsqrt_inv @ V @ np.conjugate(U) @ Gsqrt
    mpD = np.diag(_mpD)
    mpV = Gsqrt_inv @ _mpV

    return mpK, mpV, mpD
