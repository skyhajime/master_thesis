import numpy as np
from itertools import combinations, product


def fourier_basis(x, degree):
    if type(x) in (int, float):
        return np.expand_dims(np.array([np.exp(1j*k*x) for k in range(-degree, degree+1, 1)]), axis=0)/np.sqrt(2*np.pi)
    else:
        ks = [list(range(-degree, degree+1, 1)) for _ in range(x.size)]
        return np.expand_dims(np.array([np.exp(1j*sum(k*e for (k,e) in zip(p, x)))
                                        for p in product(*ks)]), axis=0)/np.sqrt(2*np.pi)

def polynomial_basis(x, degree):
    if type(x) in (int, float):
        return np.expand_dims(np.array([x**k for k in range(degree)]), axis=0)
    elif x.size == 1:
        x = np.squeeze(x)
        return np.expand_dims(np.array([x**k for k in range(degree)]), axis=0)
    else:
        return np.expand_dims(np.array([1] + [e1**k * e2**j for k in range(1, degree) for j in range(1, degree) for e1, e2 in combinations(x, 2)]), axis=0)
