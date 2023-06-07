import numba


@numba.njit
def BGK(f, f_new, f_eq, omega, source):
    for k in range(f.shape[0]):
        f[k] = (1 - omega) * f_new[k] + omega * f_eq[k] +\
            source[k]
