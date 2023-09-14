import numba


@numba.njit
def Guo_force(u, source, force, c, w, noOfDirections, cs_2,
              cs_4):
    for k in range(noOfDirections):
        cu = c[k, 0] * u[0] + c[k, 1] * u[1]
        preFactor_0 = (c[k, 0] - u[0]) * cs_2 + cu * c[k, 0] * cs_4
        preFactor_1 = (c[k, 1] - u[1]) * cs_2 + cu * c[k, 1] * cs_4
        source[k] = w[k] * (preFactor_0 * force[0] + preFactor_1 * force[1])


@numba.njit
def Guo_force_linear(u, source, force, c, w, noOfDirections):
    for k in range(noOfDirections):
        source[k] = w[k] * (c[k, 0] * force[0] + c[k, 1] * force[1])


@numba.njit
def Guo_vel(u, rho, force, forceCoeff):
    u[0] = u[0] + force[0] * forceCoeff / (rho)
    u[1] = u[1] + force[1] * forceCoeff / (rho)
