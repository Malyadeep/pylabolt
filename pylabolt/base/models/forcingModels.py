import numba


@numba.njit
def Guo_force(u, source, F, c, w, noOfDirections, cs_2, cs_4, tau):
    for k in range(noOfDirections):
        cu = c[k, 0] * u[0] + c[k, 1] * u[1]
        preFactor_0 = (c[k, 0] - u[0]) * cs_2 + cu * u[0] * cs_4
        preFactor_1 = (c[k, 1] - u[1]) * cs_2 + cu * u[1] * cs_4
        source[k] = (1 - 0.5 / tau) * w[k] * (preFactor_0 * F[0] +
                                              preFactor_1 * F[1])


@numba.njit
def Guo_vel(u, rho, F, A):
    u += F * A / (rho + 1e-9)
