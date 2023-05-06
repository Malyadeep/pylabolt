import numpy as np
from numba import cuda
from pylabolt.base.cuda.models.equilibriumModels_cuda import (firstOrder,
                                                              secondOrder)
from pylabolt.base.cuda.models.collisionModels_cuda import BGK


@cuda.jit
def equilibriumRelaxation_cuda(Nx, Ny, f_eq, f, f_new, u, rho, solid,
                               preFactor, cs_2, cs_4, c, w, equilibriumType,
                               collisionType):
    ind = cuda.grid(1)
    if ind < Nx * Ny:
        if solid[ind] != 1:
            if equilibriumType == 1:
                firstOrder(f_eq[ind, :], u[ind, :],
                           rho[ind], cs_2, cs_4, c, w)
            elif equilibriumType == 2:
                secondOrder(f_eq[ind, :], u[ind, :],
                            rho[ind], cs_2, cs_4, c, w)

            if collisionType == 1:
                BGK(f[ind, :], f_new[ind, :],
                    f_eq[ind, :], preFactor)
    else:
        return


@cuda.reduce
def residueReduce(a, b):
    return a + b


def computeResiduals_cuda(u_sq, u_err_sq, rho_sq, rho_err_sq):
    resU = np.sqrt(residueReduce(u_err_sq[:, 0])/(residueReduce(u_sq[:, 0])
                   + 1e-9))
    resV = np.sqrt(residueReduce(u_err_sq[:, 1])/(residueReduce(u_sq[:, 1])
                   + 1e-9))
    resRho = np.sqrt(residueReduce(rho_err_sq)/(residueReduce(rho_sq)
                     + 1e-9))
    # resU = np.sqrt(np.sum(u_err_sq[:, 0])/(np.sum(u_sq[:, 0])
    #                + 1e-9))
    # resV = np.sqrt(np.sum(u_err_sq[:, 1])/(np.sum(u_sq[:, 1])
    #                + 1e-9))
    # resRho = np.sqrt(np.sum(rho_err_sq)/(np.sum(rho_sq)
    #                  + 1e-9))
    return resU, resV, resRho


@cuda.jit
def computeFields_cuda(Nx, Ny, f_new, u, rho, solid, c,
                       noOfDirections, u_old, rho_old, u_sq,
                       u_err_sq, rho_sq, rho_err_sq):
    ind = cuda.grid(1)
    if ind < Nx * Ny:
        if solid[ind] != 1:
            rhoSum = 0.
            uSum = 0.
            vSum = 0.
            for k in range(noOfDirections):
                rhoSum += f_new[ind, k]
                uSum += c[k, 0] * f_new[ind, k]
                vSum += c[k, 1] * f_new[ind, k]
            rho[ind] = rhoSum
            u[ind, 0] = uSum/(rho[ind] + 1e-9)
            u[ind, 1] = vSum/(rho[ind] + 1e-9)

            u_err_sq[ind, 0] = (u[ind, 0] - u_old[ind, 0]) * \
                (u[ind, 0] - u_old[ind, 0])
            u_sq[ind, 0] = u_old[ind, 0] * u_old[ind, 0]
            u_err_sq[ind, 1] = (u[ind, 1] - u_old[ind, 1]) * \
                (u[ind, 1] - u_old[ind, 1])
            u_sq[ind, 1] = u_old[ind, 1] * u_old[ind, 1]
            rho_err_sq[ind] = (rho[ind] - rho_old[ind]) * \
                (rho[ind] - rho_old[ind])
            rho_sq[ind] = rho_old[ind] * rho_old[ind]
            u_old[ind, 0] = u[ind, 0]
            u_old[ind, 1] = u[ind, 1]
            rho_old[ind] = rho[ind]
    else:
        return


@cuda.jit
def stream_cuda(Nx, Ny, f, f_new, c, noOfDirections, invList, solid):
    ind = cuda.grid(1)
    if ind < Nx * Ny:
        i, j = np.int32(ind / Ny), np.int32(ind % Ny)
        for k in range(noOfDirections):
            i_old = (i - np.int32(c[k, 0])
                     + Nx) % Nx
            j_old = (j - np.int32(c[k, 1])
                     + Ny) % Ny
            if solid[i_old * Ny + j_old] != 1:
                f_new[ind, k] = f[i_old * Ny + j_old, k]
            elif solid[i_old * Ny + j_old] == 1:
                f_new[ind, k] = f[ind, invList[k]]
    else:
        return
