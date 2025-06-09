import numpy as np
from numba import cuda
from pylabolt.base.models.equilibriumModels_cuda import (stokesLinear,
                                                         secondOrder,
                                                         incompressible,
                                                         oseen)
from pylabolt.base.models.collisionModels_cuda import (BGK, MRT)
from pylabolt.base.models.forcingModels_cuda import (Guo_force, Guo_vel)
import pylabolt.parallel.cudaReduce as cudaReduce


@cuda.jit
def equilibriumRelaxation_cuda(Nx, Ny, f_eq, f, f_new, u, rho, solid,
                               preFactor, rho_0, U_0, cs_2, cs_4, c, w,
                               source, noOfDirections, F, equilibriumType,
                               collisionType, forcingType, forcingPreFactor):
    ind = cuda.grid(1)
    if ind < Nx * Ny:
        if solid[ind, 0] != 1:
            if equilibriumType == 1:
                stokesLinear(f_eq[ind, :], u[ind, :],
                             rho[ind], rho_0, cs_2, c, w)
            elif equilibriumType == 2:
                secondOrder(f_eq[ind, :], u[ind, :],
                            rho[ind], cs_2, cs_4, c, w)
            elif equilibriumType == 3:
                incompressible(f_eq[ind, :], u[ind, :],
                               rho[ind], rho_0, cs_2, cs_4, c, w)
            elif equilibriumType == 4:
                oseen(f_eq[ind, :], u[ind, :],
                      rho[ind], rho_0, U_0, cs_2, cs_4, c, w)

            if forcingType == 1:
                force = True
                Guo_force(u[ind, :], source[ind, :], F, c, w,
                          noOfDirections, cs_2, cs_4)

            if collisionType == 1:
                BGK(f[ind, :], f_new[ind, :],
                    f_eq[ind, :], preFactor, forcingPreFactor,
                    source[ind, :], force)
            elif collisionType == 2:
                MRT(f[ind, :], f_new[ind, :],
                    f_eq[ind, :], preFactor, forcingPreFactor,
                    source[ind, :], force)
    else:
        return


def computeResiduals_cuda(residues, noOfResidues, blocks,
                          n_threads, blockSize):
    cudaReduce.arrayShape = 6
    cudaReduce.cudaSum(blocks, n_threads, blockSize, residues)
    resU_num = residues[0, 0]
    resU_den = residues[0, 1]
    resV_num = residues[0, 2]
    resV_den = residues[0, 3]
    resRho_num = residues[0, 4]
    resRho_den = residues[0, 5]
    if np.isclose(resU_den, 0, rtol=1e-10):
        resU_den += 1e-10
    if np.isclose(resV_den, 0, rtol=1e-10):
        resV_den += 1e-10
    if np.isclose(resRho_den, 0, rtol=1e-10):
        resRho_den += 1e-10
    resU = np.sqrt(resU_num/resU_den)
    resV = np.sqrt(resV_num/resV_den)
    resRho = np.sqrt(resRho_num/resRho_den)
    return resU, resV, resRho


@cuda.jit
def computeFields_cuda(Nx, Ny, f_new, u, rho, solid, c,
                       noOfDirections, forcingType, F, A, u_old,
                       rho_old, residues, noOfResidues):
    ind = cuda.grid(1)
    if ind < Nx * Ny:
        if solid[ind, 0] != 1:
            rhoSum = 0.
            uSum = 0.
            vSum = 0.
            for k in range(noOfDirections):
                rhoSum += f_new[ind, k]
                uSum += c[k, 0] * f_new[ind, k]
                vSum += c[k, 1] * f_new[ind, k]
            rho[ind] = rhoSum
            u[ind, 0] = uSum/(rho[ind])
            u[ind, 1] = vSum/(rho[ind])

            if forcingType == 1:
                Guo_vel(u[ind, :], rho[ind], F, A)

        residues[ind, 0] = (u[ind, 0] - u_old[ind, 0]) * \
            (u[ind, 0] - u_old[ind, 0])
        residues[ind, 1] = u_old[ind, 0] * u_old[ind, 0]
        residues[ind, 2] = (u[ind, 1] - u_old[ind, 1]) * \
            (u[ind, 1] - u_old[ind, 1])
        residues[ind, 3] = u_old[ind, 1] * u_old[ind, 1]
        residues[ind, 4] = (rho[ind] - rho_old[ind]) * \
            (rho[ind] - rho_old[ind])
        residues[ind, 5] = rho_old[ind] * rho_old[ind]
        u_old[ind, 0] = u[ind, 0]
        u_old[ind, 1] = u[ind, 1]
        rho_old[ind] = rho[ind]
    else:
        return


@cuda.jit
def stream_cuda(Nx, Ny, f, f_new, solid, rho, u, c, w,
                noOfDirections, cs_2, invList):
    ind = cuda.grid(1)
    if ind < Nx * Ny:
        if solid[ind, 0] != 1:
            i, j = np.int64(ind / Ny), np.int64(ind % Ny)
            for k in range(noOfDirections):
                i_old = (i - np.int64(c[k, 0]) + Nx) % Nx
                j_old = (j - np.int64(c[k, 1]) + Ny) % Ny
                if solid[i_old * Ny + j_old, 0] != 1:
                    f_new[ind, k] = f[i_old * Ny + j_old, k]
                elif solid[i_old * Ny + j_old, 0] == 1:
                    ind_old = i_old * Ny + j_old
                    preFactor = 2 * w[invList[k]] * rho[ind] * \
                        (c[invList[k], 0] * u[ind_old, 0] +
                         c[invList[k], 1] * u[ind_old, 1]) * cs_2
                    f_new[ind, k] = f[ind, invList[k]] - preFactor
    else:
        return
