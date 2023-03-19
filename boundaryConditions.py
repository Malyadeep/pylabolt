import numba


@numba.njit
def fixedVector(f, f_new, value, ind):
    print('\n')
    print('fixedVector BC invoked')
    print(value)
    print(ind)
    print('\n')
    pass


@numba.njit
def fixedScalar(f, f_new, value, ind):
    print('\n')
    print('fixedScalar BC invoked')
    print(value)
    print(ind)
    print('\n')
    pass


@numba.njit
def bounceBack(f, f_new, value, ind):
    print('\n')
    print('bounceBack BC invoked')
    print(value)
    print(ind)
    print('\n')
    pass


@numba.njit
def periodic(f, f_new, value, ind):
    print('\n')
    print('Periodic BC invoked')
    print(value)
    print(ind)
    print('\n')
    pass
