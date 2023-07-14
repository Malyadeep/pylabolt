from numba import cuda


@cuda.jit(device=True)
def Guo_force(u, source, F, c, w, noOfDirections, cs_2,
              cs_4):
    for k in range(noOfDirections):
        cu = c[k, 0] * u[0] + c[k, 1] * u[1]
        preFactor_0 = (c[k, 0] - u[0]) * cs_2 + cu * c[k, 0] * cs_4
        preFactor_1 = (c[k, 1] - u[1]) * cs_2 + cu * c[k, 1] * cs_4
        source[k] = w[k] * (preFactor_0 * F[0] + preFactor_1 * F[1])


@cuda.jit(device=True)
def Guo_vel(u, rho, F, A):
    u[0] = u[0] + F[0] * A / (rho)
    u[1] = u[1] + F[1] * A / (rho)
