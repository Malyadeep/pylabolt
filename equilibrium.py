import numba


@numba.njit
def secondOrder(f, Nx, Ny, w, c, u, v, rho, cs):
    cs_2 = 1/(cs*cs)
    cs_4 = cs_2/(cs*cs)
    for j in range(Ny):
        for i in range(Nx):
            t2 = u[i, j]*u[i, j] + v[i, j]*v[i, j]
            for k in range(w.shape[0]):
                t1 = c[0, k]*u[i, j] + c[1, k]*v[i, j]
                f[i, j, k] = w[k]*rho[i, j]*(1 + t1*cs_2 + 0.5*t1*t1*cs_4 -
                                             0.5*t2*cs_2)


@numba.njit
def firstOrder(f, Nx, Ny, w, c, u, v, rho, cs):
    cs_2 = 1/(cs*cs)
    for j in range(Ny):
        for i in range(Nx):
            for k in range(w.shape[0]):
                t1 = c[0, k]*u[i, j] + c[1, k]*v[i, j]
                f[i, j, k] = w[k]*rho[i, j]*(1 + t1*cs_2)


if __name__ == '__main__':
    print('equilibrium distribution function')
