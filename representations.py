import numpy as np
import bases
import eigensolvers
from functools import partial
from scipy.linalg import schur, svd


def _get_GA(X, Y, basis):
    assert X.ndim == 2
    assert Y.ndim == 2
    assert X.shape[0] == Y.shape[0]
    M, _ = X.shape
    G = sum(np.conjugate(basis(X[m])).T @ basis(X[m]) for m in range(M)) / M
    A = sum(np.conjugate(basis(X[m])).T @ basis(Y[m]) for m in range(M)) / M
    return G, A

def EDMD_matrix_representation(X, Y=None, basis=partial(bases.fourier_basis, degree=5)):
    """approximate matrix representation of Koopman operator

    Args:
        X (ndarray): shape (M, nx)
        degree (int): degree of basis function
    """
    if Y is None:
        Y = X[1:]
        X = X[:-1]
    G, A = _get_GA(X, Y, basis)
    K = np.linalg.lstsq(G, A, rcond=1e-9)[0]
    D, V = np.linalg.eig(K)
    # D, V = schur(K, output='complex')
    # D = np.diag(D)
    return K, V, D


def mpEDMD_matrix_representation(X, Y=None, basis=partial(bases.fourier_basis, degree=5)):
    """approximate matrix representation of Koopman operator
    original code written in Matlab: https://github.com/MColbrook/Measure-preserving-Extended-Dynamic-Mode-Decomposition/blob/main/mpEDMD.m
    see more the paper https://epubs.siam.org/doi/abs/10.1137/22M1521407?journalCode=sjnaam

    Args:
        X (ndarray): shape (M, nx)
        degree (int): degree of basis function
    """
    if Y is None:
        Y = X[1:]
        X = X[:-1]
    G, A = _get_GA(X, Y, basis)
    G = (G + G.conj().T)/2  # make sure G is Hermitian

    DG, VG = np.linalg.eig(G)  # See https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
    VG_inv = np.linalg.inv(VG)
    # VG_inv = VG.conj().T  # showed worse performance
    Gsqrt = VG @ np.diag(np.sqrt(DG)) @ VG_inv  # G^{1/2}
    Gsqrt_inv = VG @ np.diag(np.sqrt(1.0/DG)) @ VG_inv  # G^{-1/2}

    U,S,Vh = np.linalg.svd(Gsqrt_inv @ A.conj().T @ Gsqrt_inv)
    _mpD, _mpV = schur(Vh.conj().T @ U.conj().T, output='complex')  # use schur to make sure eigenvalues on unit circle
    
    mpV = Gsqrt_inv @ _mpV
    mpK = Gsqrt_inv @ Vh.conj().T @ U.conj().T @ Gsqrt
    mpD = np.diag(_mpD)

    return mpK, mpV, mpD
