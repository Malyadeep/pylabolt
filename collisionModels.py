import numba


@numba.njit
def BGK(f, f_new, f_eq, preFactor):
    for k in range(f.shape[0]):
        f[k] = f_new[k] + preFactor * \
            (f_eq[k] - f[k])
