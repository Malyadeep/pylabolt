import numba


@numba.njit
def fixedVector(elements, value, ind, initial):
    print('\n')
    print('fixedVector BC invoked')
    print(value)
    print(ind)
    print('\n')
    pass


@numba.njit
def fixedScalar(elements, value, ind, initial):
    print('\n')
    print('fixedScalar BC invoked')
    print(value)
    print(ind)
    print('\n')
    pass


@numba.njit
def bounceBack(elements, value, ind, initial):
    print('\n')
    print('bounceBack BC invoked')
    print(value)
    print(ind)
    print('\n')
    pass


@numba.njit
def periodic(elements, value, ind, initial):
    pass
