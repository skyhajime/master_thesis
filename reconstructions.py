import numpy as np

def get_koopman_modes(X, W, basis):
    assert X.ndim == 2
    M, N = X.shape
    Phi = np.array([np.squeeze(basis(X[m])) for m in range(M)])  # should be (M, r) where r is number of dictionaries
    # Phi_inv = np.linalg.pinv(Phi)  # (M, r)
    # KM = W @ Phi_inv @ X  # should be (N, r) where N is dimention of input state
    KM = W @ np.linalg.lstsq(Phi, X, rcond=1e-9)[0]
    return KM 

def get_data_point(L, Ph, KM, k=0):
    """reconstruct one data point

    Args:
        L (np array): eigenvalues, (r, 1)
        Ph (np array): eigenfunction values for initial data point, (r, 1)
        KM (np array): koopman modes, (N, r)
        k (int, optional): index of the data point. Defaults to 0.
    """
    return KM@(L**k * Ph)  # should be (N, 1)