import numpy as np

def functional_inner_product(f1, f2, d_min, d_max, input_size=1, n=1000):
    step_size = (d_max - d_min) / n
    domain = np.linspace([d_min for _ in range(input_size)], [d_max for _ in range(input_size)], num=n)
    values = sum(f1(e) * np.conj(f2(e)) for e in domain)
    return values * step_size

def functional_inner_product_by_Birkhoff(f1, f2, X):
    # X (np.array): state evolution, x[m+1] = T(x[m]), x[m] \in M
    f1_vec = np.apply_along_axis(f1, 1, X)
    f2_vec = np.apply_along_axis(f2, 1, X)
    return np.inner(f1_vec, f2_vec)