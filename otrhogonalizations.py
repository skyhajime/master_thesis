import numpy as np
import functionals
import const as C


def vector_projection(u, v):
    # project v onto u
    if np.linalg.norm(u) > C.INV_EPS:
        return (np.inner(v, u.conj())/np.linalg.norm(u)) * u
    else:
        return v

def gram_schmidt(V, k):
    v = V[k]
    for j in range(k):  # Subtract the projections onto previous vectors
        v -= vector_projection(V[j], v)
    return v

def powered_vector(v, basis, p, input_size):
    varphi = lambda x: (np.inner(v, basis(x)))**p
    p_vec = functionals.functional_inner_product(varphi, basis, -np.pi, np.pi, input_size)
    return p_vec

def powered_gram_schmidt(V, k, basis, power_list, input_size):
    n = 0
    v = V[:,k]
    for idx, p in enumerate(power_list):   # Subtract the projections onto previous vectors and its power
        n += 1
        vp = powered_vector(V[:,idx], basis, p, input_size)
        v -= vector_projection(vp, v)
        if np.linalg.norm(v) < C.POWER_EPS:
            v.fill(0)
            break
        else:
            v /= np.linalg.norm(v)
    return V, n

    # v = V[:,k]
    # for j in range(k):
    #     for p in power_list:   # Subtract the projections onto previous vectors and its power
    #         if j == 0:
    #             n += 1
    #         vp = powered_vector(V[:,j], basis, p)
    #         v -= vector_projection(vp, v)
    #         if np.linalg.norm(v) < C.POWER_EPS:
    #             v.fill(0)
    #             break
    #         else:
    #             v /= np.linalg.norm(v)
    #     if np.linalg.norm(v) < C.POWER_EPS:
    #         break
    # return V, n
