import numba


@numba.njit
def BGK(f, f_new, f_eq, nodeType, preFactor):
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            if nodeType[i, j] != 's':
                for k in range(f.shape[2]):
                    f[i, j, k] = f_new[i, j, k] + preFactor * \
                        (f_eq[i, j, k] - f[i, j, k])
