<<<<<<< HEAD
from numba import cuda


@cuda.jit(device=True)
=======
import numba


@numba.njit
>>>>>>> 763b1fb88aebd7534ec6dac5ae28097d575b24a7
def secondOrder(f_eq, u, rho, cs_2, cs_4, c, w):
    u_2 = u[0] * u[0] + u[1] * u[1]
    for k in range(f_eq.shape[0]):
        cu = c[k, 0] * u[0] + c[k, 1] * u[1]
        f_eq[k] = w[k] * rho * (1 + cs_2 * cu +
                                0.5 * cs_4 * cu * cu -
                                0.5 * cs_2 * u_2)


<<<<<<< HEAD
@cuda.jit(device=True)
def firstOrder(f_eq, u, rho, cs_2, cs_4, c, w):
=======
@numba.njit
def firstOrder(f_eq, u, rho, cs_2, c, w):
>>>>>>> 763b1fb88aebd7534ec6dac5ae28097d575b24a7
    for k in range(f_eq.shape[0]):
        cu = c[k, 0] * u[0] + c[k, 1] * u[1]
        f_eq[k] = w[k] * rho * (1 + cu * cs_2)
