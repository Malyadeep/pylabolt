import numba


@numba.njit
def BGK(f, f_new, f_eq, preFactor, forcingPreFactor, source, force=False):
    for k in range(f.shape[0]):
        f[k] = (1 - preFactor[k]) * f_new[k] + preFactor[k] * f_eq[k] +\
            forcingPreFactor[k, k] * source[k]


@numba.njit
def MRT(f, f_new, f_eq, preFactor, forcingPreFactor, source, force=False):
    for k in range(f.shape[0]):
        f_sum = 0
        source_sum = 0
        for col in range(f.shape[0]):
            if force is True:
                source_sum += forcingPreFactor[k, col] * source[col]
            f_sum += preFactor[k, col] * (f_new[col] - f_eq[col])
        f[k] = f_new[k] - f_sum + source_sum
