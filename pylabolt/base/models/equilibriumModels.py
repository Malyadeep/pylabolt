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
def linear(f_eq, u, rho, rho_0, cs_2, c, w):
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


@numba.njit
def oseen(f_eq, u, rho, rho_0, U_0, cs_2, cs_4, c, w):
    uU = u[0] * U_0[0] + u[1] * U_0[1]
    UU = U_0[0] * U_0[0] + U_0[1] * U_0[1]
    for k in range(f_eq.shape[0]):
        cu = c[k, 0] * u[0] + c[k, 1] * u[1]
        cU = c[k, 0] * U_0[0] + c[k, 1] * U_0[1]
        f_eq[k] = w[k] * rho + w[k] * rho_0 * \
            (cs_2 * cu + (2 * cu * cU - cU * cU) * 0.5 * cs_4
             - 0.5 * cs_2 * (2 * uU - UU))
