import numba


@numba.njit
def secondOrder(f_eq, u, p, cs_2, cs_4, c, w):
    u_2 = u[0] * u[0] + u[1] * u[1]
    for k in range(f_eq.shape[0]):
        cu = c[k, 0] * u[0] + c[k, 1] * u[1]
        f_eq[k] = w[k] * (p + cs_2 * cu + 0.5 * cs_4 * cu * cu -
                          0.5 * cs_2 * u_2)


@numba.njit
def zuHe(f_eq, u, p, cs_2, cs_4, c, w, phiWeight):
    u_2 = u[0] * u[0] + u[1] * u[1]
    # f_eq[0] = phiWeight[0] - (1 - w[0]) * p - 0.5 * w[0] * u_2 * cs_2
    f_eq[0] = - (1 - w[0]) * p - 0.5 * w[0] * u_2 * cs_2
    # print(f_eq[0])
    for k in range(1, f_eq.shape[0]):
        cu = c[k, 0] * u[0] + c[k, 1] * u[1]
        # f_eq[k] = phiWeight[k] + w[k] * (p + cs_2 * cu + 0.5 * cs_4 * cu * cu -
        #                                  0.5 * cs_2 * u_2)
        f_eq[k] = w[k] * (p + cs_2 * cu + 0.5 * cs_4 * cu * cu -
                          0.5 * cs_2 * u_2)
    # print(f_eq)


@numba.njit
def segregationEquilibrium(h_eq, f_eq, u, phi, cs_2, cs_4, c, w):
    for k in range(f_eq.shape[0]):
        h_eq[k] = phi * f_eq[k]
