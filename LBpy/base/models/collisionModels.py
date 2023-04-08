import numba


@numba.njit
def BGK(f, f_new, f_eq, omega):
    for k in range(f.shape[0]):
        f[k] = (1 - omega) * f_new[k] + omega * f_eq[k]
