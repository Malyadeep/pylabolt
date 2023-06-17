import numba


@numba.njit
def BGK(f, f_new, f_eq, tau_1, source):
    for k in range(f.shape[0]):
        f[k] = (1 - tau_1) * f_new[k] + tau_1 * f_eq[k] +\
            source[k]
