from numba import cuda


@cuda.jit(device=True)
def BGK(f, f_new, f_eq, preFactor, forcingPreFactor, source, force):
    for k in range(f.shape[0]):
        f[k] = (1 - preFactor[k, k]) * f_new[k] + preFactor[k, k] * f_eq[k] +\
            forcingPreFactor[k, k] * source[k]


@cuda.jit(device=True)
def MRT(f, f_new, f_eq, preFactor, forcingPreFactor, source, force):
    for k in range(f.shape[0]):
        f_sum = 0
        source_sum = 0
        for col in range(f.shape[0]):
            if force is True:
                source_sum += forcingPreFactor[k, col] * source[col]
            f_sum += preFactor[k, col] * (f_new[col] - f_eq[col])
        f[k] = f_new[k] - f_sum + source_sum
