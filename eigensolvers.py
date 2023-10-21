import numpy as np
from scipy.linalg import schur

import const as C
from otrhogonalizations import powered_gram_schmidt, vector_projection, powered_vector


def power_iteration(A, num_iterations: int):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_iterations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k

def inverse_iteration(Matrix, mu, b, n=1000):
    k = 0
    m = Matrix.shape[1]
    try:
        inv = np.linalg.inv(Matrix - mu * np.eye(m))
    except np.linalg.LinAlgError:
        inv = np.linalg.inv(Matrix - (mu + C.INV_EPS) * np.eye(m))
    b = b / np.linalg.norm(b)
    for _ in range(n):
        k += 1
        b_ = inv @ b
        norm = np.linalg.norm(b_)
        if norm < C.INV_EPS:
            break
        else:
            b = b_ / norm
    return b, k

def _arnoldi_iteration(A, b, n: int):
    """Compute a basis of the (n + 1)-Krylov subspace of the matrix A.

    This is the space spanned by the vectors {b, Ab, ..., A^n b}.

    Parameters
    ----------
    A : array_like
        An m × m array.
    b : array_like
        Initial vector (length m).
    n : int
        One less than the dimension of the Krylov subspace, or equivalently the *degree* of the Krylov space. Must be >= 1.
    
    Returns
    -------
    Q : numpy.array
        An m x (n + 1) array, where the columns are an orthonormal basis of the Krylov subspace.
    h : numpy.array
        An (n + 1) x n array. A on basis Q. It is upper Hessenberg.
    """
    h = np.zeros((n + 1, n), dtype = 'complex_')
    Q = np.zeros((A.shape[0], n + 1), dtype = 'complex_')
    # Normalize the input vector
    Q[:, 0] = b / np.linalg.norm(b, 2)  # Use it as the first Krylov vector
    for k in range(1, n + 1):
        v = np.dot(A, Q[:, k - 1])  # Generate a new candidate vector
        for j in range(k):  # Subtract the projections on previous vectors
            h[j, k - 1] = np.dot(np.conjugate(Q[:, j]), v)
            v = v - h[j, k - 1] * Q[:, j]
        h[k, k - 1] = np.linalg.norm(v, 2)
        if h[k, k - 1] > C.ARNOLDI_EPS:  # Add the produced vector to the list, unless
            Q[:, k] = v / h[k, k - 1]
        else:  # If that happens, stop iterating.
            return Q, h, k
    return Q, h, n

def arnoldi_iteration(Matrix):
    n = Matrix.shape[0]
    b = np.random.rand(n)
    Q, H, k = _arnoldi_iteration(Matrix, b, n)
    Q = Q[:k,:k]
    Q = Q / np.linalg.norm(Q, axis=1)[:,np.newaxis]  # renormalize so that it remains orthogonal
    L, W = np.linalg.eig(H[:k,:k])
    V = Q @ W
    return L, V

def QR_algorithm(Matrix):
    eigenvalues, _ = schur(Matrix, output='complex')
    return np.diag(eigenvalues)

def power_orthogonal_inverse_iteration(Matrix, eigenvalues, basis, input_size):
    n = 0
    V = np.random.rand(Matrix.shape[0], Matrix.shape[1]) + 1j * np.random.rand(Matrix.shape[0], Matrix.shape[1])
    V = V.astype('complex128')
    V = V/np.linalg.norm(V, axis=1)
    for k in range(Matrix.shape[1]):
        curr_eig_angle = np.angle(eigenvalues[k])
        power_list = [np.angle(eig)/curr_eig_angle % (2*np.pi) for eig in eigenvalues[:k]]
        V, _n = powered_gram_schmidt(V, k, basis, power_list, input_size)
        n += _n
        V[:,k], _n = inverse_iteration(Matrix, eigenvalues[k], V[:,k])
        n += _n
    return V, n

def power_orthogonal_QR_algorithm(Matrix, basis, input_size):
    eigenvalues = QR_algorithm(Matrix)
    V, n = power_orthogonal_inverse_iteration(Matrix, eigenvalues, basis, input_size)
    return eigenvalues, V, n

def QR_algorithm_with_inverse_iteration(Matrix):
    n = 0
    eigenvalues, _ = schur(Matrix, output='complex')
    eigenvalues = np.diag(eigenvalues)
    V = np.random.rand(Matrix.shape[0], Matrix.shape[1]) + 1j * np.random.rand(Matrix.shape[0], Matrix.shape[1])
    V = V.astype('complex128')
    V = V/np.linalg.norm(V, axis=1)
    for k in range(Matrix.shape[1]):
        # for j in range(k):
        #     V[:,k] -= vector_projection(V[:,k], V[:,j])
        #     V[:,k] /= np.linalg.norm(V[:,k])
        V[:,k], _n = inverse_iteration(Matrix, eigenvalues[k], V[:,k])
        n += _n
    # assert np.allclose(Matrix, V@np.diag(eigenvalues)@V.T.conj())
    return eigenvalues, V*np.sqrt(2*np.pi), n

def QR_algorithm_with_power_vector(Matrix, basis, n_dim):
    n = 0
    eigenvalues, V = schur(Matrix, output='complex')
    eigenvalues = np.diag(eigenvalues)

    # init for power vectors
    ev_angles = np.angle(eigenvalues)
    num_inv = 1
    for i in range(num_inv):
        if i == 0:
            continue
        else:
            V[:,i], _ = inverse_iteration(Matrix, ev_angles[i], V[:,i])
            V[:,i] -= vector_projection(u=V[:,0], v=V[:,i])
            V[:,i] /= np.linalg.norm(V[:,i])
    # N = V.shape[1]
    # crand = np.random.rand(N) + 1j*np.random.rand(N)
    # V[:,l], _n = inverse_iteration(Matrix, ev_angles[l], V[:,l])
    # V[:,l] -= vector_projection(u=V[:,k], v=V[:,l])
    # V[:,l] /= np.linalg.norm(V[:,l])
    # assert np.allclose(Matrix@V[:,k], eigenvalues[k]*V[:,k])
    # print(Matrix@V[:,l] - eigenvalues[l]*V[:,l])
    # assert np.allclose(Matrix@V[:,l], eigenvalues[l]*V[:,l])
    # V[:,k] /= np.linalg.norm(V[:,k])
    for i, ev_angle in enumerate(ev_angles):
        if i < num_inv:
            continue
        v = V[:,i]
        v.fill(0)
        for j in range(num_inv):
            pj = ev_angle / ev_angles[j]
            vj = powered_vector(V[:,j], basis, pj, n_dim)
            v += vj[0]
        V[:,i] = v/np.linalg.norm(v)
        # pk = ev_angle / ev_angles[k]
        # pl = ev_angle / ev_angles[l]
        # assert np.isclose(eigenvalues[i], eigenvalues[k]**pk)
        # assert np.isclose(eigenvalues[i], eigenvalues[l]**pl)
        # vk = powered_vector(V[:,k], basis, pk, n_dim)
        # vl = powered_vector(V[:,l], basis, pk, n_dim)
        # v = vk + vl
        # V[:,i] = vk/np.linalg.norm(vk)
        # V[:,i] -= vector_projection(u=V[:,k], v=V[:,i])
        # for j in range(i):
        #     V[:,i] -= vector_projection(u=V[:,j], v=V[:,i])
        #     V[:,i] /= np.linalg.norm(V[:,i])
        # V[:,i] /= np.linalg.norm(V[:,i])
    V *= np.sqrt(2*np.pi)
    n += len(ev_angles)
    return eigenvalues, V, n
