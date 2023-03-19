import numba


@numba.njit
def stream(f, f_new, nodeType, dtdx, c):
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            if nodeType[i, j] != 's':
                for k in range(1, f.shape[2]):
                    i_old = (i - int(c[k, 0]*dtdx)
                             + f.shape[0]) % f.shape[0]
                    j_old = (j - int(c[k, 1]*dtdx)
                             + f.shape[1]) % f.shape[1]
                    f_new[i, j, k] = f[i_old, j_old, k]
