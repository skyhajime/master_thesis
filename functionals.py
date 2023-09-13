import numpy as np

def functional_inner_product(f1, f2, d_min, d_max, input_size=1, n=5):
    step_size = (d_max - d_min) // n
    domain = np.linspace([d_min for _ in range(input_size)], [d_max for _ in range(input_size)], num=n)
    values = sum(f1(e) * np.conj(f2(e)) for e in domain)
    return values * step_size
