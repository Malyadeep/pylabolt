import numba


@numba.njit
def secondOrder(f_eq, u, rho, cs_2, cs_4, c, w):
    u_2 = u[0] * u[0] + u[1] * u[1]
    for k in range(f_eq.shape[0]):
        cu = c[k, 0] * u[0] + c[k, 1] * u[1]
        f_eq[k] = w[k] * rho * (1 + cs_2 * cu +
                                0.5 * cs_4 * cu * cu -
                                0.5 * cs_2 * u_2)


@numba.njit
def stokesLinear(f_eq, u, rho, rho_0, cs_2, c, w):
    for k in range(f_eq.shape[0]):
        cu = c[k, 0] * u[0] + c[k, 1] * u[1]
        f_eq[k] = w[k] * (rho + rho_0 * cu * cs_2)


@numba.njit
def incompressible(f_eq, u, rho, rho_0, cs_2, cs_4, c, w):
    u_2 = u[0] * u[0] + u[1] * u[1]
    for k in range(f_eq.shape[0]):
        cu = c[k, 0] * u[0] + c[k, 1] * u[1]
        f_eq[k] = w[k] * rho + w[k] * rho_0 * (cs_2 * cu +
                                               0.5 * cs_4 * cu * cu -
                                               0.5 * cs_2 * u_2)
