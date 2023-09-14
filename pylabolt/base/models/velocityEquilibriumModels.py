import numba


@numba.njit
def secondOrder(f_eq, u, p, cs_2, cs_4, c, w):
    u_2 = u[0] * u[0] + u[1] * u[1]
    for k in range(f_eq.shape[0]):
        cu = c[k, 0] * u[0] + c[k, 1] * u[1]
        f_eq[k] = w[k] * (p + cs_2 * cu + 0.5 * cs_4 * cu * cu -
                          0.5 * cs_2 * u_2)
