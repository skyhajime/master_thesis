import numpy as np

def get_eigenfunctions_from_matrix(basis, V):
    """reconstruct eigenfunction from eigenvectors

    Args:
        basis (function): basis function which returns values with shape (1, r)
        V (2d-ndarray): eigenvectors with shape (r, r)

    Returns:
        function: vector-valued eigenfunction reconstruction with shape (1, r) where i-th function is \sum_{k=1}^r (v_{k,i} basis(x)_{i})
    """
    return lambda x: basis(x)@V

def get_eigenfunctions_with_exponential(basis, v_0, powers):
    """reconstruct eigenfunction from one eigenvector and their exponential

    Args:
        basis (function): basis function which returns values with shape (1, r)
        v_0 (1d-ndarray): eigenvectors with shape (r,) and the norm 1
        powers (list): power values with shape (r,)

    Returns:
        function: vector-valued eigenfunction reconstruction with shape (1, r) where i-th function is (\sum_{k=1}^r (v_0)_i basis(x)_{i})^p_i
    """
    v_0 /= np.linalg.norm(v_0)
    phi = lambda x: np.squeeze(np.inner(basis(x), v_0*np.sqrt(2*np.pi)))
    return lambda x: np.array([phi(x)**p for p in powers])
