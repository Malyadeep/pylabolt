import numba


@numba.njit
def BGK(pop, pop_new, pop_eq, preFactor, forcingPreFactor, source,
        force=False):
    for k in range(pop.shape[0]):
        pop[k] = (1 - preFactor[k, k]) * pop_new[k] + preFactor[k, k] * \
            pop_eq[k] + forcingPreFactor[k, k] * source[k]


@numba.njit
def MRT(pop, pop_new, pop_eq, preFactor, forcingPreFactor, source,
        force=False):
    for k in range(pop.shape[0]):
        f_sum = 0
        source_sum = 0
        for col in range(pop.shape[0]):
            if force is True:
                source_sum += forcingPreFactor[k, col] * source[col]
            f_sum += preFactor[k, col] * (pop_new[col] - pop_eq[col])
        pop[k] = pop_new[k] - f_sum + source_sum
