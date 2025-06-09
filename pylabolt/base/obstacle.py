import os
import sys
import numba
import numpy as np
from numba import (prange, cuda)


from pylabolt.base import obstacleTypes
from pylabolt.parallel.MPI_comm import (proc_boundary,
                                        proc_boundaryGradTerms)


@numba.njit
def findNb(solid, fluidBoundary, solidBoundary, boundaryNode,
           procBoundary, Nx, Ny, c, noOfDirections,
           obsNo, nx, ny, nProc_x, nProc_y, size=1):
    fluidNbNodes = []
    solidNbNodes = []
    for i in range(Nx):
        for j in range(Ny):
            ind = i * Ny + j
            if (solid[ind, 0] != 1 and boundaryNode[ind] != 1 and
                    procBoundary[ind] != 1):
                for k in range(noOfDirections):
                    i_nb = i + int(c[k, 0])
                    j_nb = j + int(c[k, 1])
                    if size == 1 and boundaryNode[ind] == 2:
                        if i + int(2 * c[k, 0]) < 0 or i + int(2 * c[k, 0]) >= Nx:
                            i_nb = (i_nb + int(2 * c[k, 0]) + Nx) % Nx
                        else:
                            i_nb = (i_nb + Nx) % Nx
                        if j + int(2 * c[k, 1]) < 0 or j + int(2 * c[k, 1]) >= Ny:
                            j_nb = (j_nb + int(2 * c[k, 1]) + Ny) % Ny
                        else:
                            j_nb = (j_nb + Ny) % Ny
                    elif size > 1 and boundaryNode[ind] == 2:
                        if (i + int(2 * c[k, 0]) == 0 and nx == 0 or
                                i + int(2 * c[k, 0]) == Nx - 1 and
                                nx == nProc_x - 1):
                            i_nb = i_nb + int(c[k, 0])
                        if (j + int(2 * c[k, 1]) == 0 and ny == 0 or
                                j + int(2 * c[k, 1]) == Ny - 1 and
                                ny == nProc_y - 1):
                            j_nb = j_nb + int(c[k, 1])
                    ind_nb = int(i_nb * Ny + j_nb)
                    if (solid[ind_nb, 0] == 1 and solid[ind_nb, 1] == obsNo
                            and boundaryNode[ind_nb] != 1):
                        fluidBoundary[ind] = 1
                        fluidNbNodes.append(ind)
                        break
            elif solid[ind, 0] != 0 and boundaryNode[ind] != 1:
                for k in range(noOfDirections):
                    i_nb = i + int(c[k, 0])
                    j_nb = j + int(c[k, 1])
                    if size == 1 and boundaryNode[ind] == 2:
                        if i + int(2 * c[k, 0]) < 0 or i + int(2 * c[k, 0]) >= Nx:
                            i_nb = (i_nb + int(2 * c[k, 0]) + Nx) % Nx
                        else:
                            i_nb = (i_nb + Nx) % Nx
                        if j + int(2 * c[k, 1]) < 0 or j + int(2 * c[k, 1]) >= Ny:
                            j_nb = (j_nb + int(2 * c[k, 1]) + Ny) % Ny
                        else:
                            j_nb = (j_nb + Ny) % Ny
                    elif size > 1 and boundaryNode[ind] == 2:
                        if (i + int(2 * c[k, 0]) == 0 and nx == 0 or
                                i + int(2 * c[k, 0]) == Nx - 1 and
                                nx == nProc_x - 1):
                            i_nb = i_nb + int(c[k, 0])
                        if (j + int(2 * c[k, 1]) == 0 and ny == 0 or
                                j + int(2 * c[k, 1]) == Ny - 1 and
                                ny == nProc_y - 1):
                            j_nb = j_nb + int(c[k, 1])
                    ind_nb = int(i_nb * Ny + j_nb)
                    if (solid[ind_nb, 0] == 0 and solid[ind, 1] == obsNo
                            and boundaryNode[ind_nb] != 1):
                        solidBoundary[ind] = 1
                        solidNbNodes.append(ind)
                        break
            else:
                continue
    return np.array(fluidNbNodes, dtype=np.int64), \
        np.array(solidNbNodes, dtype=np.int64)


def reInitSolidFluidBoundaries(Nx, Ny, fluidBoundary, solidBoundary,
                               fluidBoundary_old, solidBoundary_old):
    for ind in prange(Nx * Ny):
        fluidBoundary_old[ind] = fluidBoundary[ind]
        solidBoundary_old[ind] = solidBoundary[ind]
        fluidBoundary[ind] = 0
        solidBoundary[ind] = 0


def setVelocityObs(Nx, Ny, solid, u, boundaryNode, obsU):
    for ind in prange(Nx * Ny):
        if solid[ind, 0] == 1 and boundaryNode[ind] != 1:
            u[ind, 0] = obsU[0]
            u[ind, 1] = obsU[1]


@cuda.jit
def setObsVelocity_cuda(obsNodes, momentOfInertia, omega, omega_old,
                        torque, u, obsOrigin, Nx, Ny, obsNo):
    ind = cuda.grid(1)
    if ind < Nx * Ny:
        for itr in prange(obsNodes.shape[0]):
            if ind == obsNodes[itr]:
                omega[itr] = omega_old[itr] + torque / momentOfInertia
                i, j = int(ind / Ny), int(ind % Ny)
                x = i - obsOrigin[0]
                y = j - obsOrigin[1]
                theta = np.arctan2(y, x)
                r = (x**2 + y**2)**(0.5)
                u[ind, 0] = - r * omega[obsNo] * np.sin(theta)
                u[ind, 1] = r * omega[obsNo] * np.cos(theta)
                omega_old[obsNo] = omega[obsNo]
            else:
                continue
    else:
        return


def translateSolid(obsMass, obsU, obsOrigin, Nx_global, Ny_global, obsNo,
                   x_periodic, y_periodic):
    # if np.abs(obsU[obsNo, 0]) < 1e-13:
    #     latticeTime_x = np.int64(1)
    # else:
    #     latticeTime_x = np.int64(np.abs(1 / obsU[obsNo, 0]))
    # if np.abs(obsU[obsNo, 1]) < 1e-13:
    #     latticeTime_y = np.int64(1)
    # else:
    #     latticeTime_y = np.int64(np.abs(1 / obsU[obsNo, 1]))
    if x_periodic is True:
        obsOrigin[obsNo, 0] = (obsOrigin[obsNo, 0] + obsU[obsNo, 0] - 1
                               + Nx_global - 2) % (Nx_global - 2) + 1
    else:
        obsOrigin[obsNo, 0] = obsOrigin[obsNo, 0] + obsU[obsNo, 0]
    if y_periodic is True:
        obsOrigin[obsNo, 1] = (obsOrigin[obsNo, 1] + obsU[obsNo, 1] - 1
                               + Ny_global - 2) % (Ny_global - 2) + 1
    else:
        obsOrigin[obsNo, 1] = obsOrigin[obsNo, 1] + obsU[obsNo, 1]
    # print(obsOrigin[obsNo])


def setObsVelocity(solid, procBoundary, boundaryNode, momentOfInertia,
                   obsMass, obsOmega, obsOmega_old, obsU, obsU_old, force,
                   torque, gravity, u, obsOrigin, Nx, Ny, Nx_global, Ny_global,
                   nx, ny, nProc_x, nProc_y, N_local, size, x_periodic,
                   y_periodic, reconstructSolidFunc, reconstructSolidArgs,
                   externalForce, externalTorque, translation, rotation,
                   obsNo):
    if rotation is True:
        obsOmega[obsNo] = obsOmega_old[obsNo] + torque / momentOfInertia
    if translation is True:
        obsU[obsNo, 0] = obsU_old[obsNo, 0] + force[0] / obsMass + gravity[0]
        obsU[obsNo, 1] = obsU_old[obsNo, 1] + force[1] / obsMass + gravity[1]
    # obsU[obsNo, 0] = 0
    # obsU[obsNo, 1] = 0
    for ind in prange(Nx * Ny):
        if (procBoundary[ind] != 1 and boundaryNode[ind] != 1 and
                solid[ind, 0] == 1 and solid[ind, 1] == obsNo):
            i, j = int(ind / Ny), int(ind % Ny)
            if size > 1:
                i_local, j_local = i, j
                Nx_local, Ny_local = Nx - 2, Ny - 2
                if nx == nProc_x - 1:
                    Nx_local = N_local[nx - 1, ny, 0]
                if ny == nProc_y - 1:
                    Ny_local = N_local[nx, ny - 1, 1]
                i_global = nx * Nx_local + i_local - 1
                j_global = ny * Ny_local + j_local - 1
            else:
                i_global, j_global = i, j
            insideSolid, x, y, leastDist = \
                reconstructSolidFunc(i_global, j_global, Nx_global, Ny_global,
                                     obsOrigin[obsNo], x_periodic, y_periodic,
                                     *reconstructSolidArgs)
            theta = np.arctan2(y, x)
            r = np.sqrt(leastDist)
            # if i_global == 41 and j_global == 51:
            #     print(x, y, r, insideSolid, force)
            u[ind, 0] = 0
            u[ind, 1] = 0
            if translation is True:
                u[ind, 0] += obsU[obsNo, 0]
                u[ind, 1] += obsU[obsNo, 1]
            if rotation is True:
                u[ind, 0] += (- r * obsOmega[obsNo] * np.sin(theta))
                u[ind, 1] += r * obsOmega[obsNo] * np.cos(theta)
    obsOmega_old[obsNo] = obsOmega[obsNo]
    obsU_old[obsNo, 0] = obsU[obsNo, 0]
    obsU_old[obsNo, 1] = obsU[obsNo, 1]
    if translation is True:
        if x_periodic is True:
            obsOrigin[obsNo, 0] = (obsOrigin[obsNo, 0] + obsU[obsNo, 0] - 1
                                   + Nx_global - 2) % (Nx_global - 2) + 1
        else:
            obsOrigin[obsNo, 0] = obsOrigin[obsNo, 0] + obsU[obsNo, 0]
        if y_periodic is True:
            obsOrigin[obsNo, 1] = (obsOrigin[obsNo, 1] + obsU[obsNo, 1] - 1
                                   + Ny_global - 2) % (Ny_global - 2) + 1
        else:
            obsOrigin[obsNo, 1] = obsOrigin[obsNo, 1] + obsU[obsNo, 1]


def computeSolidMass(mass, solid, boundaryNode, procBoundary, Nx, Ny):
    temp = 0.
    itr = 0
    for ind in prange(Nx * Ny):
        if (boundaryNode[ind] != 1 and procBoundary[ind] == 0):
            temp += solid[ind, 0]
            if solid[ind, 0]:
                itr += 1
    mass[0] = temp
    return mass


@numba.njit
def createPropertiesFluid(ind, boundaryNode, solid_old, rho, c, w,
                          noOfDirections, Nx, Ny, nx, ny, nProc_x,
                          nProc_y, size):
    rhoSum = 0
    denominator = 0
    i, j = int(ind / Ny), int(ind % Ny)
    for k in range(1, noOfDirections):
        i_nb = i + int(c[k, 0])
        j_nb = j + int(c[k, 1])
        if size == 1 and boundaryNode == 2:
            if (i + int(2 * c[k, 0]) < 0 or i +
                    int(2 * c[k, 0]) >= Nx):
                i_nb = (i_nb + int(2 * c[k, 0]) + Nx) % Nx
            else:
                i_nb = (i_nb + Nx) % Nx
            if (j + int(2 * c[k, 1]) < 0 or j +
                    int(2 * c[k, 1]) >= Ny):
                j_nb = (j_nb + int(2 * c[k, 1]) + Ny) % Ny
            else:
                j_nb = (j_nb + Ny) % Ny
        elif size > 1 and boundaryNode == 2:
            if (i + int(2 * c[k, 0]) == 0 and nx == 0 or
                    i + int(2 * c[k, 0]) == Nx - 1 and
                    nx == nProc_x - 1):
                i_nb = i_nb + int(c[k, 0])
            if (j + int(2 * c[k, 1]) == 0 and ny == 0 or
                    j + int(2 * c[k, 1]) == Ny - 1 and
                    ny == nProc_y - 1):
                j_nb = j_nb + int(c[k, 1])
        ind_nb = int(i_nb * Ny + j_nb)
        if solid_old[ind_nb, 0] != 1:
            rhoSum += w[k] * rho[ind_nb]
            denominator += w[k]
    rho[ind] = rhoSum / (denominator + 1e-17)


@numba.njit
def createPropertiesPhase(ind, boundaryNode, solid_old, rho, p, phi, normalPhi,
                          massAdded, c, w, noOfDirections, rho_l, rho_g, phi_g,
                          Nx, Ny, nx, ny, nProc_x, nProc_y, size):
    pSum, phiSum = 0, 0
    normalPhiSum_x, normalPhiSum_y = 0, 0
    denominator, massAddedSum = 0, 0
    node = False
    i, j = int(ind / Ny), int(ind % Ny)
    # if i == 55 and j == 45:
    #     node = True
    for k in range(1, noOfDirections):
        i_nb = i + int(c[k, 0])
        j_nb = j + int(c[k, 1])
        if size == 1 and boundaryNode == 2:
            if (i + int(2 * c[k, 0]) < 0 or i +
                    int(2 * c[k, 0]) >= Nx):
                i_nb = (i_nb + int(2 * c[k, 0]) + Nx) % Nx
            else:
                i_nb = (i_nb + Nx) % Nx
            if (j + int(2 * c[k, 1]) < 0 or j +
                    int(2 * c[k, 1]) >= Ny):
                j_nb = (j_nb + int(2 * c[k, 1]) + Ny) % Ny
            else:
                j_nb = (j_nb + Ny) % Ny
        elif size > 1 and boundaryNode == 2:
            if (i + int(2 * c[k, 0]) == 0 and nx == 0 or
                    i + int(2 * c[k, 0]) == Nx - 1 and
                    nx == nProc_x - 1):
                i_nb = i_nb + int(c[k, 0])
            if (j + int(2 * c[k, 1]) == 0 and ny == 0 or
                    j + int(2 * c[k, 1]) == Ny - 1 and
                    ny == nProc_y - 1):
                j_nb = j_nb + int(c[k, 1])
        ind_nb = int(i_nb * Ny + j_nb)
        if solid_old[ind_nb, 0] != 1:
            # if node is True:
            #     print(k, i_nb, j_nb, phi[ind_nb], massAdded[ind_nb])
            phiSum += w[k] * phi[ind_nb]
            # normalPhiSum_x += w[k] * normalPhi[ind_nb, 0]
            # normalPhiSum_y += w[k] * normalPhi[ind_nb, 1]
            pSum += w[k] * p[ind_nb]
            massAddedSum += w[k] * massAdded[ind_nb]
            denominator += w[k]
    phi[ind] = phiSum / (denominator + 1e-17)
    # normalPhi[ind, 0] = normalPhiSum_x / (denominator + 1e-17)    # testing pinning
    # normalPhi[ind, 1] = normalPhiSum_y / (denominator + 1e-17)    # testing pinning
    # if i == 55 and j == 45:
    #     print(phi[ind], massAddedSum)
    # if massAddedSum < 0:
    #     massAdded[ind] = - massAddedSum
    # if massAddedSum >= 0:
    #     massAdded[ind] = massAddedSum
    # phi[ind] = -massAddedSum / (denominator + 1e-20)
    # massAdded[ind] = massAddedSum
    # if i == 55 and j == 45:
    #     print(phi[ind], massAdded[ind])
    # phi[ind] -= massAddedSum
    # if i == 55 and j == 45:
    #     print(phi[ind])
    p[ind] = pSum / (denominator + 1e-17)
    rho[ind] = rho_g + (phi[ind] - phi_g) * (rho_l - rho_g)


def initializeObsModifiableFields(solid_old, solid, phi_old, phi,
                                  forceCreation, torqueCreation, forceDes,
                                  torqueDes, Nx, Ny, phase=False):
    for ind in prange(Nx * Ny):
        solid_old[ind, 0] = solid[ind, 0]
        solid_old[ind, 1] = solid[ind, 1]
        forceCreation[ind, 0] = 0
        forceCreation[ind, 1] = 0
        forceDes[ind, 0] = 0
        forceDes[ind, 1] = 0
        torqueCreation[ind] = 0
        torqueDes[ind] = 0
        if phase is True:
            phi_old[ind] = phi[ind]


def moveObstacle(solid_old, solid, procBoundary, boundaryNode, rhoSolid, rho,
                 p, phi, phi_old, normalPhi, massAdded, deltaM, u, rho_l,
                 rho_g, phi_g, f_new, h_new, obsOrigin, obsVel, obsOmega,
                 forceCreation, torqueCreation, forceDes, torqueDes, c, w,
                 noOfDirections, equilibriumFuncFluid, equilibriumArgsFluid,
                 equilibriumFuncPhase, equilibriumArgsPhase, Nx, Ny, Nx_global,
                 Ny_global, nx, ny, nProc_x, nProc_y, N_local, size,
                 x_periodic, y_periodic, calculated, reconstructSolidFunc,
                 reconstructSolidArgs, phase, translation, rotation,
                 segregation, obsNo):
    created = 0
    destroyed = 0
    # if obsNo == 0:
    #     for ind in prange(Nx * Ny):
    #         solid_old[ind, 0] = solid[ind, 0]
    #         solid_old[ind, 1] = solid[ind, 1]
    for ind in prange(Nx * Ny):
        if procBoundary[ind] != 1 and boundaryNode[ind] != 1:
            i, j = int(ind / Ny), int(ind % Ny)
            if size > 1:
                i_local, j_local = i, j
                Nx_local, Ny_local = Nx - 2, Ny - 2
                if nx == nProc_x - 1:
                    Nx_local = N_local[nx - 1, ny, 0]
                if ny == nProc_y - 1:
                    Ny_local = N_local[nx, ny - 1, 1]
                i_global = nx * Nx_local + i_local - 1
                j_global = ny * Ny_local + j_local - 1
            else:
                i_global, j_global = i, j
            insideSolid, x, y, leastDist = \
                reconstructSolidFunc(i_global, j_global, Nx_global, Ny_global,
                                     obsOrigin, x_periodic, y_periodic,
                                     *reconstructSolidArgs)
            # if ind == int(101 * 203 + 81):
            #     print(insideSolid)
            if insideSolid:
                if solid_old[ind, 0] == 1 and solid_old[ind, 1] == obsNo:
                    # Culprit in pinning
                    # if phase is True:
                    #     phi[ind] = 0
                    pass
                elif solid_old[ind, 0] == 0:
                    solid[ind, 0] = 1
                    solid[ind, 1] = obsNo
                    forceDes[ind, 0] = rho[ind] * u[ind, 0]
                    forceDes[ind, 1] = rho[ind] * u[ind, 1]
                    torqueDes[ind] = (x * u[ind, 1] - y * u[ind, 0])\
                        * rho[ind]
                    rho[ind] = rhoSolid
                    if phase is True:
                        deltaM[ind] = phi[ind] - massAdded[ind]
                        # if i_global == 151 and (j_global == 90 or j_global == 89):
                        #     print(phi[ind], massAdded[ind], deltaM[ind])
                        p[ind] = 0
                        # phi[ind] = 0  # Pinning culprit
                        massAdded[ind] = 0
                    destroyed += 1
                    if calculated is False:
                        u[ind, 0] = 0
                        u[ind, 1] = 0
                        if translation is True:
                            u[ind, 0] += obsVel[0]
                            u[ind, 1] += obsVel[1]
                        if rotation is True:
                            theta = np.arctan2(y, x)
                            r = np.sqrt(leastDist)
                            u[ind, 0] = obsVel[0] - r * obsOmega *\
                                np.sin(theta)
                            u[ind, 1] = obsVel[1] + r * obsOmega *\
                                np.cos(theta)
            else:
                if solid_old[ind, 0] == 0:
                    pass
                elif solid_old[ind, 0] == 1 and solid_old[ind, 1] == obsNo:
                    if phase is False:
                        createPropertiesFluid(ind, boundaryNode[ind],
                                              solid_old, rho, c, w,
                                              noOfDirections, Nx, Ny,
                                              nx, ny, nProc_x, nProc_y, size)
                        equilibriumFuncFluid(f_new[ind], u[ind], rho[ind],
                                             *equilibriumArgsFluid)
                    else:
                        createPropertiesPhase(ind, boundaryNode[ind],
                                              solid_old, rho, p, phi,
                                              normalPhi, massAdded, c, w,
                                              noOfDirections, rho_l, rho_g,
                                              phi_g, Nx, Ny, nx, ny, nProc_x,
                                              nProc_y, size)
                        equilibriumFuncFluid(f_new[ind], u[ind], p[ind],
                                             *equilibriumArgsFluid)
                        if segregation is False:
                            equilibriumFuncPhase(h_new[ind], u[ind], phi[ind],
                                                 *equilibriumArgsPhase)
                        # else:
                        #     equilibriumFuncPhase(h_new[ind], f_new[ind],
                        #                          phi[ind],
                        #                          *equilibriumArgsPhase)
                    solid[ind, 0] = 0
                    solid[ind, 1] = 0
                    forceCreation[ind, 0] = rho[ind] * u[ind, 0]
                    forceCreation[ind, 1] = rho[ind] * u[ind, 1]
                    torqueCreation[ind] = (x * u[ind, 1] - y * u[ind, 0]) *\
                        rho[ind]
                    created += 1


def updateGhostNodesHysteresis(phi, phi_old, solid, solid_old, procBoundary,
                               boundaryNode, c, w, noOfDirections, size, nx,
                               ny, nProc_x, nProc_y, Nx, Ny, obsNo):
    for ind in prange(Nx * Ny):
        if (procBoundary[ind] != 1 and boundaryNode[ind] != 1 and
                solid[ind, 0] == 1 and solid_old[ind, 0] == 1):
            i, j = int(ind / Ny), int(ind % Ny)
            phiGhostSum = 0
            denominator = 0
            flag = False
            for k in range(1, noOfDirections):
                i_nb = i + int(c[k, 0])
                j_nb = j + int(c[k, 1])
                if size == 1 and boundaryNode[ind] == 2:
                    if (i + int(2 * c[k, 0]) < 0 or i +
                            int(2 * c[k, 0]) >= Nx):
                        i_nb = (i_nb + int(2 * c[k, 0]) + Nx) % Nx
                    else:
                        i_nb = (i_nb + Nx) % Nx
                    if (j + int(2 * c[k, 1]) < 0 or j +
                            int(2 * c[k, 1]) >= Ny):
                        j_nb = (j_nb + int(2 * c[k, 1]) + Ny) % Ny
                    else:
                        j_nb = (j_nb + Ny) % Ny
                elif size > 1 and boundaryNode[ind] == 2:
                    if (i + int(2 * c[k, 0]) == 0 and nx == 0 or
                            i + int(2 * c[k, 0]) == Nx - 1 and
                            nx == nProc_x - 1):
                        i_nb = i_nb + int(c[k, 0])
                    if (j + int(2 * c[k, 1]) == 0 and ny == 0 or
                            j + int(2 * c[k, 1]) == Ny - 1 and
                            ny == nProc_y - 1):
                        j_nb = j_nb + int(c[k, 1])
                ind_nb = int(i_nb * Ny + j_nb)
                if (solid_old[ind_nb, 0] == 1 and solid_old[ind_nb, 1] ==
                        obsNo and solid[ind_nb, 0] == 0 and
                        solid[ind, 1] == obsNo):
                    flag = True
                    phiGhostSum += w[k] * phi_old[ind_nb]
                    denominator += w[k]
            # if j == 63 and i == 31:
            #     print(flag, solid[ind, 0], solid_old[ind, 0])
            if flag is True:
                if obsNo == 0:
                    phi[ind] = phi_old[int(i * Ny + j - 1)]
                elif obsNo == 1:
                    # phi[ind] = phi_old[int(i * Ny + j + 1)]
                    phi[ind] = phi_old[ind]
                else:
                    phi[ind] = phiGhostSum / (denominator + 1e-17)
                # phi[ind] = phiGhostSum / (denominator + 1e-17)
            # phi[ind] = phiGhostSum / (denominator + 1e-17)
        if (procBoundary[ind] != 1 and boundaryNode[ind] != 1 and
                solid[ind, 0] == 1 and solid_old[ind, 0] == 0):
            i, j = int(ind / Ny), int(ind % Ny)
            phiGhostSum = 0
            denominator = 0
            flag = False
            for k in range(1, noOfDirections):
                i_nb = i + int(c[k, 0])
                j_nb = j + int(c[k, 1])
                if size == 1 and boundaryNode[ind] == 2:
                    if (i + int(2 * c[k, 0]) < 0 or i +
                            int(2 * c[k, 0]) >= Nx):
                        i_nb = (i_nb + int(2 * c[k, 0]) + Nx) % Nx
                    else:
                        i_nb = (i_nb + Nx) % Nx
                    if (j + int(2 * c[k, 1]) < 0 or j +
                            int(2 * c[k, 1]) >= Ny):
                        j_nb = (j_nb + int(2 * c[k, 1]) + Ny) % Ny
                    else:
                        j_nb = (j_nb + Ny) % Ny
                elif size > 1 and boundaryNode[ind] == 2:
                    if (i + int(2 * c[k, 0]) == 0 and nx == 0 or
                            i + int(2 * c[k, 0]) == Nx - 1 and
                            nx == nProc_x - 1):
                        i_nb = i_nb + int(c[k, 0])
                    if (j + int(2 * c[k, 1]) == 0 and ny == 0 or
                            j + int(2 * c[k, 1]) == Ny - 1 and
                            ny == nProc_y - 1):
                        j_nb = j_nb + int(c[k, 1])
                ind_nb = int(i_nb * Ny + j_nb)
                if (solid_old[ind_nb, 0] == 1 and solid_old[ind_nb, 1] ==
                        obsNo and solid[ind_nb, 0] == 1 and
                        solid[ind, 1] == obsNo):
                    flag = True
                    phiGhostSum += w[k] * phi_old[ind_nb]
                    denominator += w[k]
            # if j == 61 and i == 31:
            #     print(flag, solid[ind, 0], solid_old[ind, 0])
            if flag is True:
                phi[ind] = phiGhostSum / (denominator + 1e-17)
            # phi[ind] = phiGhostSum / (denominator + 1e-17)


def normalsCreatedNodes(phi, solid, solid_old, procBoundary,
                        boundaryNode, gradPhi, normalPhi, lapPhi,
                        c, w, cs_2, noOfDirections, size, nx, ny,
                        nProc_x, nProc_y, Nx, Ny, obsNo):
    for ind in prange(Nx * Ny):
        if (procBoundary[ind] != 1 and boundaryNode[ind] != 1 and
                solid[ind, 0] == 0 and solid_old[ind, 0] == 1):
            i, j = int(ind / Ny), int(ind % Ny)
            gradPhiSum_x, gradPhiSum_y = 0., 0.
            lapPhiSum = 0.
            denominator = 0.
            noOfBoundaryNodes = 0
            for k in range(noOfDirections):
                i_nb = i + int(c[k, 0])
                j_nb = j + int(c[k, 1])
                if size == 1 and boundaryNode[ind] == 2:
                    if i + int(2 * c[k, 0]) < 0 or i + int(2 * c[k, 0]) >= Nx:
                        i_nb = (i_nb + int(2 * c[k, 0]) + Nx) % Nx
                    else:
                        i_nb = (i_nb + Nx) % Nx
                    if j + int(2 * c[k, 1]) < 0 or j + int(2 * c[k, 1]) >= Ny:
                        j_nb = (j_nb + int(2 * c[k, 1]) + Ny) % Ny
                    else:
                        j_nb = (j_nb + Ny) % Ny
                elif size > 1 and boundaryNode[ind] == 2:
                    if (i + int(2 * c[k, 0]) == 0 or
                            i + int(2 * c[k, 0]) == Nx - 1):
                        i_nb = i_nb + int(c[k, 0])
                    if (j + int(2 * c[k, 1]) == 0 or
                            j + int(2 * c[k, 1]) == Ny - 1):
                        j_nb = j_nb + int(c[k, 1])
                ind_nb = int(i_nb * Ny + j_nb)
                # gradPhiSum_x += c[k, 0] * w[k] * phi[ind_nb]
                # gradPhiSum_y += c[k, 1] * w[k] * phi[ind_nb]
                # lapPhiSum += w[k] * (phi[ind_nb] - phi[ind])
                if boundaryNode[ind_nb] != 1 or solid[ind_nb, 0] == 1:
                    gradPhiSum_x += c[k, 0] * w[k] * phi[ind_nb]
                    gradPhiSum_y += c[k, 1] * w[k] * phi[ind_nb]
                    lapPhiSum += w[k] * (phi[ind_nb] - phi[ind])
                    denominator += w[k]
                else:
                    noOfBoundaryNodes += 1
            # gradPhi[ind, 0] = cs_2 * gradPhiSum_x
            # gradPhi[ind, 1] = cs_2 * gradPhiSum_y
            # lapPhi[ind] = 2.0 * cs_2 * lapPhiSum
            if noOfBoundaryNodes == 0:
                gradPhi[ind, 0] = cs_2 * gradPhiSum_x
                gradPhi[ind, 1] = cs_2 * gradPhiSum_y
                lapPhi[ind] = 2.0 * cs_2 * lapPhiSum
            else:
                gradPhi[ind, 0] = cs_2 * gradPhiSum_x / (denominator + 1e-17)
                gradPhi[ind, 1] = cs_2 * gradPhiSum_y / (denominator + 1e-17)
                lapPhi[ind] = 2.0 * cs_2 * lapPhiSum / (denominator + 1e-17)
            magGradPhi = np.sqrt(gradPhi[ind, 0] * gradPhi[ind, 0] +
                                 gradPhi[ind, 1] * gradPhi[ind, 1])
            # print(magGradPhi)
            normalPhi[ind, 0] = gradPhi[ind, 0] / (magGradPhi + 1e-17)
            normalPhi[ind, 1] = gradPhi[ind, 1] / (magGradPhi + 1e-17)
        if (procBoundary[ind] != 1 and boundaryNode[ind] != 1 and
                solid[ind, 0] == 1 and solid_old[ind, 0] == 0):
            i, j = int(ind / Ny), int(ind % Ny)
            gradPhiSum_x, gradPhiSum_y = 0., 0.
            lapPhiSum = 0.
            denominator = 0.
            noOfBoundaryNodes = 0
            for k in range(noOfDirections):
                i_nb = i + int(c[k, 0])
                j_nb = j + int(c[k, 1])
                if size == 1 and boundaryNode[ind] == 2:
                    if i + int(2 * c[k, 0]) < 0 or i + int(2 * c[k, 0]) >= Nx:
                        i_nb = (i_nb + int(2 * c[k, 0]) + Nx) % Nx
                    else:
                        i_nb = (i_nb + Nx) % Nx
                    if j + int(2 * c[k, 1]) < 0 or j + int(2 * c[k, 1]) >= Ny:
                        j_nb = (j_nb + int(2 * c[k, 1]) + Ny) % Ny
                    else:
                        j_nb = (j_nb + Ny) % Ny
                elif size > 1 and boundaryNode[ind] == 2:
                    if (i + int(2 * c[k, 0]) == 0 or
                            i + int(2 * c[k, 0]) == Nx - 1):
                        i_nb = i_nb + int(c[k, 0])
                    if (j + int(2 * c[k, 1]) == 0 or
                            j + int(2 * c[k, 1]) == Ny - 1):
                        j_nb = j_nb + int(c[k, 1])
                ind_nb = int(i_nb * Ny + j_nb)
                # gradPhiSum_x += c[k, 0] * w[k] * phi[ind_nb]
                # gradPhiSum_y += c[k, 1] * w[k] * phi[ind_nb]
                # lapPhiSum += w[k] * (phi[ind_nb] - phi[ind])
                if boundaryNode[ind_nb] != 1 or solid[ind_nb, 0] == 1:
                    gradPhiSum_x += c[k, 0] * w[k] * phi[ind_nb]
                    gradPhiSum_y += c[k, 1] * w[k] * phi[ind_nb]
                    lapPhiSum += w[k] * (phi[ind_nb] - phi[ind])
                    denominator += w[k]
                else:
                    noOfBoundaryNodes += 1
            # gradPhi[ind, 0] = cs_2 * gradPhiSum_x
            # gradPhi[ind, 1] = cs_2 * gradPhiSum_y
            # lapPhi[ind] = 2.0 * cs_2 * lapPhiSum
            if noOfBoundaryNodes == 0:
                gradPhi[ind, 0] = cs_2 * gradPhiSum_x
                gradPhi[ind, 1] = cs_2 * gradPhiSum_y
                lapPhi[ind] = 2.0 * cs_2 * lapPhiSum
            else:
                gradPhi[ind, 0] = cs_2 * gradPhiSum_x / (denominator + 1e-17)
                gradPhi[ind, 1] = cs_2 * gradPhiSum_y / (denominator + 1e-17)
                lapPhi[ind] = 2.0 * cs_2 * lapPhiSum / (denominator + 1e-17)
            magGradPhi = np.sqrt(gradPhi[ind, 0] * gradPhi[ind, 0] +
                                 gradPhi[ind, 1] * gradPhi[ind, 1])
            # print(magGradPhi)
            normalPhi[ind, 0] = gradPhi[ind, 0] / (magGradPhi + 1e-17)
            normalPhi[ind, 1] = gradPhi[ind, 1] / (magGradPhi + 1e-17)


def setInnerSolidNodePhi(phi, solid, boundaryNode, procBoundary, c,
                         noOfDirections, Nx, Ny, size, phase):
    if phase is True:
        for ind in prange(Nx * Ny):
            if (procBoundary[ind] == 0 and boundaryNode[ind] != 1 and
                    solid[ind, 0] == 1):
                i, j = int(ind / Ny), int(ind % Ny)
                solidBoundaryFlag = False
                for k in range(noOfDirections):
                    i_nb = i + int(c[k, 0])
                    j_nb = j + int(c[k, 1])
                    if size == 1 and boundaryNode[ind] == 2:
                        if i + int(2 * c[k, 0]) < 0 or i + int(2 * c[k, 0]) >= Nx:
                            i_nb = (i_nb + int(2 * c[k, 0]) + Nx) % Nx
                        else:
                            i_nb = (i_nb + Nx) % Nx
                        if j + int(2 * c[k, 1]) < 0 or j + int(2 * c[k, 1]) >= Ny:
                            j_nb = (j_nb + int(2 * c[k, 1]) + Ny) % Ny
                        else:
                            j_nb = (j_nb + Ny) % Ny
                    elif size > 1 and boundaryNode[ind] == 2:
                        if (i + int(2 * c[k, 0]) == 0 or
                                i + int(2 * c[k, 0]) == Nx - 1):
                            i_nb = i_nb + int(c[k, 0])
                        if (j + int(2 * c[k, 1]) == 0 or
                                j + int(2 * c[k, 1]) == Ny - 1):
                            j_nb = j_nb + int(c[k, 1])
                    ind_nb = int(i_nb * Ny + j_nb)
                    if solid[ind_nb, 0] == 0:
                        solidBoundaryFlag = True
                        break
                if solidBoundaryFlag is False:
                    phi[ind] = 0


class obstacleSetup:
    def __init__(self, size, mesh, precision, phase=False):
        self.obstacles = []
        self.obsType = []
        self.isPeriodic = []
        self.velType = []
        self.obstaclesGlobal = []
        self.noOfObstacles = 0
        self.fluidNbObstacle = []
        self.solidNbObstacle = []
        self.obsNodes = []
        self.modifyObsFunc = []
        self.degreeOfFreedom = []
        self.externalForceType = []
        self.externalForce = []
        self.externalTorqueType = []
        self.externalTorque = []
        self.translation = []
        self.rotation = []
        self.reconstructSolidFunc = []
        self.reconstructSolidArgs = []
        self.obsModifiable = False
        self.momentOfInertia = []
        self.obsDensity = []
        self.obsMass = []
        self.obsOmega_old = []
        self.obsOmega = []
        self.obsU_old = []
        self.obsU = []
        self.obsOrigin = []
        self.radius = []
        self.allStatic = True
        self.solidMass = np.zeros(1, dtype=precision)
        self.displaySolidMass = False
        self.phase = phase
        self.N_local = np.zeros((2, 2, 0), dtype=np.int64)
        if size > 1:
            self.vector_send_topBottom = np.zeros((mesh.Nx + 2, 2),
                                                  dtype=precision)
            self.vector_recv_topBottom = np.zeros((mesh.Nx + 2, 2),
                                                  dtype=precision)
            self.vector_send_leftRight = np.zeros((mesh.Ny + 2, 2),
                                                  dtype=precision)
            self.vector_recv_leftRight = np.zeros((mesh.Ny + 2, 2),
                                                  dtype=precision)
            self.scalar_send_topBottom = np.zeros(mesh.Nx + 2,
                                                  dtype=precision)
            self.scalar_recv_topBottom = np.zeros(mesh.Nx + 2,
                                                  dtype=precision)
            self.scalar_send_leftRight = np.zeros(mesh.Ny + 2,
                                                  dtype=precision)
            self.scalar_recv_leftRight = np.zeros(mesh.Ny + 2,
                                                  dtype=precision)
        self.writeInterval = 1
        self.writeProperties = False

        # device data
        self.obsNodes_device = []
        self.momentOfInertia_device = []
        self.obsOmega_device = []
        self.obsOmega_old_device = []
        self.obsU_device = []
        self.obsU_old_device = []
        self.modifyObsFunc_device = []
        self.modifyObsNo_device = []
        self.obsOrigin_device = []

    def setObstacle(self, mesh, precision, options, initialFields, rank, size,
                    comm):
        workingDir = os.getcwd()
        sys.path.append(workingDir)
        try:
            from simulation import obstacle
            if rank == 0:
                print('Reading Obstacle data...', flush=True)
        except ImportError:
            if rank == 0:
                print('No obstacle defined!', flush=True)
            return np.zeros((mesh.Nx_global * mesh.Ny_global, 2),
                            dtype=np.int32)
        try:
            if rank == 0:
                solid = np.full((mesh.Nx_global * mesh.Ny_global, 2),
                                fill_value=0, dtype=np.int32)
            else:
                return 0
            if len(obstacle.keys()) == 0:
                if rank == 0:
                    print('Reading Obstacle data done!', flush=True)
                return solid
            else:
                self.obstacles = list(obstacle.keys())
                try:
                    self.obstacles.remove('displaySolidMass')
                    self.displaySolidMass = obstacle['displaySolidMass']
                except ValueError:
                    self.displaySolidMass = False
                print("Display solid mass is set as " +
                      str(self.displaySolidMass))
            for obsNo, obsName in enumerate(self.obstacles):
                obstacleType = obstacle[obsName]['type']
                rho_s = obstacle[obsName]['rho_s']
                self.noOfObstacles += 1
                if obstacleType == 'circle':
                    center = obstacle[obsName]['center']
                    radius = obstacle[obsName]['radius']
                    self.radius.append(radius)
                    periodic = obstacle[obsName]['periodic']
                    if periodic is True:
                        self.isPeriodic.append(True)
                    else:
                        self.isPeriodic.append(False)
                    if (isinstance(center, list) and isinstance(radius, float)
                            or isinstance(radius, int)):
                        center = np.array(center, dtype=precision)
                        radius = precision(radius)
                        self.obsType.append('circle')
                    else:
                        if rank == 0:
                            print("ERROR!", flush=True)
                            print("For 'circle' type obstacle center must"
                                  + " be a list and radius must be a float",
                                  flush=True)
                        os._exit(1)
                    static = obstacle[obsName]['static']
                    if static is True:
                        self.modifyObsFunc.append(None)
                        nodes, momentOfInertia, obsMass, obsDensity = \
                            obstacleTypes.circle(center, radius, solid,
                                                 initialFields, rho_s, mesh,
                                                 obsNo, periodic=periodic)
                        self.obsNodes.append(nodes)
                        self.momentOfInertia.append(momentOfInertia)
                        self.obsMass.append(obsMass)
                        self.obsDensity.append(obsDensity)
                        self.obsOrigin.append(center + np.ones(2))
                        self.velType.append(None)
                        self.obsU.append(np.zeros(2, dtype=precision))
                        self.obsOmega.append(precision(0))
                        self.reconstructSolidArgs.append(None)
                        self.reconstructSolidFunc.append(None)
                        self.degreeOfFreedom.append(None)
                        self.externalForceType.append(None)
                        self.externalForce.append(np.array([0, 0],
                                                  dtype=precision))
                        self.externalTorqueType.append(None)
                        self.externalTorque.append(precision(0))
                        self.translation.append(False)
                        self.rotation.append(False)
                    else:
                        self.allStatic = False
                        velDict = obstacle[obsName]['U_def']
                        if velDict['type'] == 'fixedVelocity':
                            velType = 'fixedVelocity'
                            self.velType.append(velType)
                            degreeOfFreedom = velDict['degreeOfFreedom']
                            if degreeOfFreedom == 'translation':
                                self.degreeOfFreedom.append("translation")
                                self.translation.append(True)
                                self.rotation.append(False)
                                self.reconstructSolidFunc.\
                                    append(obstacleTypes.reconstructCircle)
                                self.reconstructSolidArgs.\
                                    append((radius, center))
                                self.modifyObsFunc.append(translateSolid)
                                self.obsModifiable = True
                            elif degreeOfFreedom == 'rotation':
                                self.degreeOfFreedom.append("rotation")
                                self.translation.append(False)
                                self.rotation.append(True)
                                self.reconstructSolidFunc.append(None)
                                self.reconstructSolidArgs.append(None)
                                self.modifyObsFunc.append(None)
                                self.obsModifiable = True
                            elif degreeOfFreedom == 'both':
                                self.degreeOfFreedom.append("both")
                                self.translation.append(True)
                                self.rotation.append(True)
                                self.reconstructSolidFunc.\
                                    append(obstacleTypes.reconstructCircle)
                                self.reconstructSolidArgs.\
                                    append((radius, center))
                                self.modifyObsFunc.append(translateSolid)
                                self.obsModifiable = True
                            else:
                                if rank == 0:
                                    print("ERROR!", flush=True)
                                    print("Invalid degree of freedom!",
                                          flush=True)
                                os._exit(1)
                            velValue = velDict['value']
                            if not isinstance(velValue, list):
                                if rank == 0:
                                    print("ERROR!", flush=True)
                                    print("For 'fixedVelocity' type "
                                          + "velocity, value must be a list",
                                          flush=True)
                                os._exit(1)
                            velValue = np.array(velValue, dtype=precision)
                            self.obsU.append(velValue)
                            velOrigin = velDict['origin']
                            velOmega = velDict['angularVelocity']
                            if (not isinstance(velOrigin, list) and not
                                    isinstance(velOmega, float)):
                                if rank == 0:
                                    print("ERROR!", flush=True)
                                    print("For 'fixedVelocity' type " +
                                          "velocity, origin must be " +
                                          "a list and angular velocity " +
                                          "must be a float",
                                          flush=True)
                                os._exit(1)
                            velOrigin = precision(velOrigin)
                            velOmega = precision(velOmega)
                            self.obsOmega.append(velOmega)
                            self.obsOrigin.append(np.divide(velOrigin,
                                                  mesh.delX) + np.ones(2,
                                                  dtype=np.int64))
                            nodes, momentOfInertia, obsMass, obsDensity = \
                                obstacleTypes.\
                                circle(center, radius, solid, initialFields,
                                       rho_s, mesh, obsNo, velType=velType,
                                       velValue=velValue, velOrigin=velOrigin,
                                       velOmega=velOmega, periodic=periodic)
                            self.obsNodes.append(nodes)
                            self.momentOfInertia.append(momentOfInertia)
                            self.obsMass.append(obsMass)
                            self.obsDensity.append(obsDensity)
                            self.writeProperties = velDict['write']
                            if self.writeProperties is True:
                                self.writeInterval = \
                                    velDict['writeInterval']
                            self.externalForceType.append(None)
                            self.externalForce.append(np.array([0, 0],
                                                      dtype=precision))
                            self.externalTorqueType.append(None)
                            self.externalTorque.append(precision(0))
                        elif (velDict['type'] == 'fixedRotational'):
                            velType = velDict['type']
                            self.velType.append(velType)
                            velOrigin = velDict['origin']
                            velOmega = velDict['angularVelocity']
                            self.reconstructSolidFunc.append(None)
                            self.reconstructSolidArgs.append(None)
                            if (not isinstance(velOrigin, list) and not
                                    isinstance(velOmega, float)):
                                if rank == 0:
                                    print("ERROR!", flush=True)
                                    print("For 'fixedRotational' type " +
                                          "velocity, origin must be " +
                                          "a list and angular velocity " +
                                          "must be a float",
                                          flush=True)
                                os._exit(1)
                            velOrigin = np.array(velOrigin, dtype=precision)
                            velOmega = precision(velOmega)
                            self.obsU.append(np.zeros(2, dtype=precision))
                            self.obsOmega.append(velOmega)
                            self.obsOrigin.append(np.divide(velOrigin,
                                                  mesh.delX) + np.ones(2,
                                                  dtype=np.int64))
                            self.modifyObsFunc.append(None)
                            nodes, momentOfInertia, obsMass, obsDensity = \
                                obstacleTypes.\
                                circle(center, radius, solid, initialFields,
                                       rho_s, mesh, obsNo, velType=velType,
                                       velOrigin=velOrigin,
                                       velOmega=velOmega, periodic=periodic)
                            self.obsModifiable = True
                            self.writeProperties = velDict['write']
                            if self.writeProperties is True:
                                self.writeInterval = \
                                    velDict['writeInterval']
                            self.degreeOfFreedom.append(None)
                            self.externalForceType.append(None)
                            self.externalForce.append(np.array([0, 0],
                                                      dtype=precision))
                            self.externalTorqueType.append(None)
                            self.externalTorque.append(precision(0))
                            self.translation.append(False)
                            self.rotation.append(True)
                        elif velDict['type'] == 'calculated':
                            velType = 'calculated'
                            self.velType.append(velType)
                            self.reconstructSolidFunc.\
                                append(obstacleTypes.reconstructCircle)
                            self.reconstructSolidArgs.append((radius, center))
                            degreeOfFreedom = velDict['degreeOfFreedom']
                            if degreeOfFreedom == 'translation':
                                self.degreeOfFreedom.append("translation")
                                self.translation.append(True)
                                self.rotation.append(False)
                            elif degreeOfFreedom == 'rotation':
                                self.degreeOfFreedom.append("rotation")
                                self.translation.append(False)
                                self.rotation.append(True)
                            elif degreeOfFreedom == 'both':
                                self.degreeOfFreedom.append("both")
                                self.translation.append(True)
                                self.rotation.append(True)
                            else:
                                if rank == 0:
                                    print("ERROR!", flush=True)
                                    print("Invalid degree of freedom!",
                                          flush=True)
                                os._exit(1)
                            velValue = velDict['value']
                            if not isinstance(velValue, list):
                                if rank == 0:
                                    print("ERROR!", flush=True)
                                    print("For 'calculated' type "
                                          + "velocity, value must be a list",
                                          flush=True)
                                os._exit(1)
                            velValue = np.array(velValue, dtype=precision)
                            self.obsU.append(velValue)
                            velOrigin = velDict['origin']
                            velOmega = velDict['angularVelocity']
                            if (not isinstance(velOrigin, list) and not
                                    isinstance(velOmega, float)):
                                if rank == 0:
                                    print("ERROR!", flush=True)
                                    print("For 'calculated' type " +
                                          "velocity, origin must be " +
                                          "a list and angular velocity " +
                                          "must be a float",
                                          flush=True)
                                os._exit(1)
                            velOrigin = precision(velOrigin)
                            velOmega = precision(velOmega)
                            self.obsOmega.append(velOmega)
                            self.obsOrigin.append(np.divide(velOrigin,
                                                  mesh.delX) + np.ones(2,
                                                  dtype=np.int64))
                            nodes, momentOfInertia, obsMass, obsDensity = \
                                obstacleTypes.\
                                circle(center, radius, solid, initialFields,
                                       rho_s, mesh, obsNo, velType=velType,
                                       velValue=velValue, velOrigin=velOrigin,
                                       velOmega=velOmega, periodic=periodic)
                            self.obsNodes.append(nodes)
                            self.momentOfInertia.append(momentOfInertia)
                            self.obsMass.append(obsMass)
                            self.obsDensity.append(obsDensity)
                            self.modifyObsFunc.append(setObsVelocity)
                            self.obsModifiable = True
                            self.writeProperties = velDict['write']
                            if self.writeProperties is True:
                                self.writeInterval = \
                                    velDict['writeInterval']
                            try:
                                externalForceTorqueDict = \
                                    obstacle[obsName]['externalForceTorque']
                                externalForcePresent = True
                            except KeyError:
                                externalForcePresent = False
                                self.externalForceType.append(None)
                                self.externalForce.append(np.array([0, 0],
                                                          dtype=precision))
                                self.externalTorqueType.append(None)
                                self.externalTorque.append(precision(0))
                            if externalForcePresent is True:
                                forceType = \
                                    externalForceTorqueDict['forceType']
                                if forceType == 'constant':
                                    self.externalForceType.append(forceType)
                                else:
                                    if rank == 0:
                                        print("ERROR!", flush=True)
                                        print("Invalid external force type!",
                                              flush=True)
                                    os._exit(1)
                                torqueType = \
                                    externalForceTorqueDict['torqueType']
                                if torqueType == 'constant':
                                    self.externalTorqueType.append(torqueType)
                                else:
                                    if rank == 0:
                                        print("ERROR!", flush=True)
                                        print("Invalid external torque type!",
                                              flush=True)
                                    os._exit(1)
                                forceValue = \
                                    externalForceTorqueDict['forceValue']
                                torqueValue = \
                                    externalForceTorqueDict['torqueValue']
                                if (not isinstance(forceValue, list) and not
                                        isinstance(torqueValue, float)):
                                    if rank == 0:
                                        print("ERROR!", flush=True)
                                        print("External force value " +
                                              "must be a list " +
                                              "and external torque value " +
                                              "must be a float",
                                              flush=True)
                                    os._exit(1)
                                self.externalForce.\
                                    append(np.array(forceValue,
                                           dtype=precision))
                                self.externalTorque.\
                                    append(precision(torqueValue))
                        else:
                            if rank == 0:
                                print("ERROR!", flush=True)
                                print("Unsupported velocity type " +
                                      velDict['type'], flush=True)
                                os._exit(1)
                elif obstacleType == 'rectangle':
                    boundingBox = obstacle[obsName]['boundingBox']
                    periodic = obstacle[obsName]['periodic']
                    self.radius.append(None)
                    if periodic is True:
                        self.isPeriodic.append(True)
                    else:
                        self.isPeriodic.append(False)
                    if isinstance(boundingBox, list):
                        boundingBox = np.array(boundingBox, dtype=np.int64)
                        self.obsType.append('rectangle')
                    else:
                        print("ERROR!", flush=True)
                        print("For 'rectangle' type obstacle bounding box"
                              + "  must be a list", flush=True)
                        os._exit(1)
                    static = obstacle[obsName]['static']
                    if static is True:
                        self.reconstructSolidFunc.\
                            append(None)
                        self.reconstructSolidArgs.append(None)
                        self.modifyObsFunc.append(None)
                        nodes, momentOfInertia, obsMass, obsDensity = \
                            obstacleTypes.rectangle(boundingBox, solid,
                                                    initialFields,
                                                    rho_s, mesh, obsNo,
                                                    periodic=periodic)
                        self.obsNodes.append(nodes)
                        self.momentOfInertia.append(momentOfInertia)
                        self.obsMass.append(obsMass)
                        self.obsDensity.append(obsDensity)
                        self.obsOrigin.append(np.zeros(2, dtype=precision))
                        self.velType.append(None)
                        self.obsU.append(np.zeros(2, dtype=precision))
                        self.obsOmega.append(precision(0))
                        self.degreeOfFreedom.append(None)
                        self.externalForceType.append(None)
                        self.externalForce.append(np.array([0, 0],
                                                  dtype=precision))
                        self.externalTorqueType.append(None)
                        self.externalTorque.append(precision(0))
                        self.translation.append(False)
                        self.rotation.append(False)
                    else:
                        self.allStatic = False
                        velDict = obstacle[obsName]['U_def']
                        velType = velDict['type']
                        if velType == 'fixedVelocity':
                            self.velType.append(velType)
                            self.modifyObsFunc.append(translateSolid)
                            degreeOfFreedom = velDict['degreeOfFreedom']
                            if degreeOfFreedom == 'translation':
                                self.degreeOfFreedom.append("translation")
                                self.translation.append(True)
                                self.rotation.append(False)
                            else:
                                if rank == 0:
                                    print("ERROR!", flush=True)
                                    print("Invalid degree of freedom" +
                                          " for a rectangle!", flush=True)
                                os._exit(1)
                            self.externalForceType.append(None)
                            self.externalForce.append(np.array([0, 0],
                                                      dtype=precision))
                            self.externalTorqueType.append(None)
                            self.externalTorque.append(precision(0))
                        elif velType == 'calculated':
                            self.velType.append(velType)
                            self.modifyObsFunc.append(setObsVelocity)
                            degreeOfFreedom = velDict['degreeOfFreedom']
                            if degreeOfFreedom == 'translation':
                                self.degreeOfFreedom.append("translation")
                                self.translation.append(True)
                                self.rotation.append(False)
                            else:
                                if rank == 0:
                                    print("ERROR!", flush=True)
                                    print("Invalid degree of freedom" +
                                          " for a rectangle!", flush=True)
                                os._exit(1)
                            try:
                                externalForceTorqueDict = \
                                    obstacle[obsName]['externalForceTorque']
                                externalForcePresent = True
                            except KeyError:
                                externalForcePresent = False
                                self.externalForceType.append(None)
                                self.externalForce.append(np.array([0, 0],
                                                          dtype=precision))
                                self.externalTorqueType.append(None)
                                self.externalTorque.append(precision(0))
                            if externalForcePresent is True:
                                forceType = \
                                    externalForceTorqueDict['forceType']
                                if forceType == 'constant':
                                    self.externalForceType.append(forceType)
                                else:
                                    if rank == 0:
                                        print("ERROR!", flush=True)
                                        print("Invalid external force type!",
                                              flush=True)
                                    os._exit(1)
                                forceValue = \
                                    externalForceTorqueDict['forceValue']
                                if (not isinstance(forceValue, list)):
                                    if rank == 0:
                                        print("ERROR!", flush=True)
                                        print("External force value " +
                                              "must be a list!",
                                              flush=True)
                                    os._exit(1)
                                self.externalForce.\
                                    append(np.array(forceValue,
                                           dtype=precision))
                                self.externalTorqueType.append(None)
                                self.externalTorque.append(precision(0))
                        else:
                            if rank == 0:
                                print("ERROR!", flush=True)
                                print("Unsupported velocity type " +
                                      velDict['type'], flush=True)
                                os._exit(1)
                        velValue = velDict['value']
                        if not isinstance(velValue, list):
                            if rank == 0:
                                print("ERROR!", flush=True)
                                print("Velocity value must be a list",
                                      flush=True)
                            os._exit(1)
                        velValue = np.array(velValue, dtype=precision)
                        self.obsU.append(velValue)
                        velOrigin = velDict['origin']
                        if not isinstance(velOrigin, list):
                            if rank == 0:
                                print("ERROR!", flush=True)
                                print("For 'fixedTranslational' or " +
                                      " 'calculated' type velocity, " +
                                      " origin must be a list", flush=True)
                        velOmega = precision(0)
                        # print(velOmega, velValue, velOrigin)
                        lengthAhead = \
                            np.abs(boundingBox[1, 0] - velOrigin[0])
                        lengthBehind = \
                            np.abs(velOrigin[0] - boundingBox[0, 0])
                        breadthAhead = \
                            np.abs(boundingBox[1, 1] - velOrigin[1])
                        breadthBehind = \
                            np.abs(velOrigin[1] - boundingBox[0, 1])
                        adjustForFloatError = np.zeros((2, 2),
                                                       dtype=np.float64)
                        if velValue[0] > 0:
                            lengthBehind += 1 #- np.abs(velValue[0])
                            adjustForFloatError[0, 0] = 1e-13
                            # lengthAhead += np.abs(velValue[0])
                        elif velValue[0] < 0:
                            lengthAhead += 1 #- np.abs(velValue[0])
                            adjustForFloatError[0, 1] = 1e-13
                            # lengthBehind += np.abs(velValue[0])
                        if velValue[1] > 0:
                            breadthBehind += 1 #- np.abs(velValue[1])
                            adjustForFloatError[1, 0] = 1e-13
                            # breadthAhead += np.abs(velValue[1])
                        elif velValue[1] < 0:
                            breadthAhead += 1 #- np.abs(velValue[1])
                            adjustForFloatError[1, 1] = 1e-13
                            # breadthBehind += np.abs(velValue[1])
                        # lengthAhead += 0.5
                        # lengthBehind += 0.5
                        # breadthAhead += 0.5
                        # breadthBehind += 0.5
                        self.reconstructSolidFunc.\
                            append(obstacleTypes.reconstructRectangle)
                        self.reconstructSolidArgs.\
                            append((lengthAhead, lengthBehind,
                                    breadthAhead, breadthBehind,
                                    adjustForFloatError))
                        self.obsOmega.append(velOmega)
                        origin = \
                            np.int64(np.divide(velOrigin, mesh.delX)) \
                            + np.ones(2, dtype=np.int64)
                        self.obsOrigin.append(origin)
                        nodes, momentOfInertia, obsMass, obsDensity = \
                            obstacleTypes.\
                            rectangle(boundingBox, solid, initialFields,
                                      rho_s, mesh, obsNo, velType=velType,
                                      velValue=velValue, velOmega=velOmega,
                                      periodic=periodic)
                        self.obsNodes.append(nodes)
                        self.momentOfInertia.append(momentOfInertia)
                        self.obsMass.append(obsMass)
                        self.obsDensity.append(obsDensity)
                        self.obsModifiable = True
                        self.writeProperties = velDict['write']
                        if self.writeProperties is True:
                            self.writeInterval = \
                                velDict['writeInterval']
                elif obstacleType == 'inclinedRectangle':
                    centerLine = obstacle[obsName]['centerLine']
                    width = obstacle[obsName]['width']
                    self.radius.append(None)
                    periodic = obstacle[obsName]['periodic']
                    self.reconstructSolidFunc.append(None)
                    self.reconstructSolidArgs.append(None)
                    if periodic is True:
                        self.isPeriodic.append(True)
                    else:
                        self.isPeriodic.append(False)
                    if (isinstance(centerLine, list) and
                            isinstance(width, float) or
                            isinstance(width, int)):
                        centerLine = np.array(centerLine, dtype=np.int64)
                        width = precision(width)
                        self.obsType.append('inclinedRectangle')
                    else:
                        print("ERROR!", flush=True)
                        print("For 'inclinedRectangle' type obstacle center"
                              + "line must be a list and width must be float"
                              + " or int.", flush=True)
                        os._exit(1)
                    static = obstacle[obsName]['static']
                    if static is True:
                        self.modifyObsFunc.append(None)
                        nodes, momentOfInertia, obsMass, obsDensity = \
                            obstacleTypes.\
                            inclinedRectangle(centerLine, width, solid,
                                              initialFields, rho_s, mesh,
                                              obsNo, comm, periodic=periodic)
                        self.obsNodes.append(nodes)
                        self.momentOfInertia.append(momentOfInertia)
                        self.obsMass.append(obsMass)
                        self.obsDensity.append(obsDensity)
                        self.obsOrigin.append(np.zeros(2, dtype=precision))
                        self.velType.append(None)
                        self.obsU.append(np.zeros(2, dtype=precision))
                        self.obsOmega.append(precision(0))
                    else:
                        if rank == 0:
                            print("ERROR!", flush=True)
                            print("'inclinedRectangle' doesn't" +
                                  " support obstacle modification ",
                                  flush=True)
                            os._exit(1)
                elif obstacleType == 'circularConfinement':
                    center = obstacle[obsName]['center']
                    radius = obstacle[obsName]['radius']
                    self.radius.append(radius)
                    periodic = obstacle[obsName]['periodic']
                    self.reconstructSolidFunc.append(None)
                    self.reconstructSolidArgs.append(None)
                    if periodic is True:
                        self.isPeriodic.append(True)
                        print("'Circular confinement' obstacle is" +
                              " always periodic!", flush=True)
                    else:
                        self.isPeriodic.append(True)
                        print("'Circular confinement' obstacle is" +
                              " always periodic!", flush=True)
                    if (isinstance(center, list) and isinstance(radius, float)
                            or isinstance(radius, int)):
                        center = np.array(center, dtype=np.float64)
                        radius = np.float64(radius)
                        self.obsType.append('circularConfinement')
                    else:
                        print("ERROR!", flush=True)
                        print("For 'circularConfinement' type obstacle center"
                              + " must be a list and radius must be a float",
                              flush=True)
                        os._exit(1)
                    static = obstacle[obsName]['static']
                    if static is True:
                        self.modifyObsFunc.append(None)
                        nodes, momentOfInertia, obsMass, obsDensity = \
                            obstacleTypes.\
                            circularConfinement(center, radius, solid,
                                                initialFields,
                                                rho_s, mesh, obsNo,
                                                periodic=periodic)
                        self.obsNodes.append(nodes)
                        self.momentOfInertia.append(momentOfInertia)
                        self.obsMass.append(obsMass)
                        self.obsDensity.append(obsDensity)
                        self.obsOrigin.append(np.zeros(2, dtype=precision))
                        self.velType.append(None)
                        self.obsU.append(np.zeros(2, dtype=precision))
                        self.obsOmega.append(precision(0))
                    else:
                        self.allStatic = False
                        velDict = obstacle[obsName]['U_def']
                        if (velDict['type'] == 'fixedTranslational' or
                                velDict['type'] == 'calculatedRotational'):
                            print("ERROR!", flush=True)
                            print("Only 'fixedRotational' velocity" +
                                  " type is supported for 'rectangle' " +
                                  "obstacle", flush=True)
                            os._exit(1)
                        elif velDict['type'] == 'fixedRotational':
                            velType = velDict['type']
                            self.velType.append(velType)
                            velOrigin = velDict['origin']
                            velOmega = velDict['angularVelocity']
                            if (not isinstance(velOrigin, list) and not
                                    isinstance(velOmega, float)):
                                if rank == 0:
                                    print("ERROR!", flush=True)
                                    print("For 'fixedRotational' type " +
                                          "velocity, origin must be " +
                                          "a list and angular velocity " +
                                          "must be a float",
                                          flush=True)
                                os._exit(1)
                            velOrigin = np.array(velOrigin, dtype=precision)
                            velOmega = precision(velOmega)
                            self.obsU.append(np.zeros(2, dtype=precision))
                            self.obsOmega.append(velOmega)
                            self.obsOrigin.append(np.int64(np.divide(velOrigin,
                                                  mesh.delX)) + np.ones(2,
                                                  dtype=np.int64))
                            self.modifyObsFunc.append(None)
                            nodes, momentOfInertia, obsMass, obsDensity = \
                                obstacleTypes.\
                                circularConfinement(center, radius, solid,
                                                    initialFields, rho_s,
                                                    mesh, obsNo,
                                                    velType=velType,
                                                    velOrigin=velOrigin,
                                                    velOmega=velOmega,
                                                    periodic=periodic)
                            self.obsNodes.append(nodes)
                            self.momentOfInertia.append(momentOfInertia)
                            self.obsMass.append(obsMass)
                            self.obsDensity.append(obsDensity)
                else:
                    print("ERROR!")
                    print("Unsupported obstacle type!", flush=True)
                    os._exit(1)
            if rank == 0:
                print('Reading Obstacle data done!', flush=True)
            # if self.allStatic is False:
            #     self.obsU = np.array(self.obsU, dtype=precision)
            #     self.obsOmega = np.array(self.obsOmega, dtype=precision)
            #     self.obsOrigin = np.array(self.obsOrigin, dtype=precision)
            #     self.obsU_old = np.copy(self.obsU)
            #     self.obsOmega_old = np.copy(self.obsOmega)
            # else:
            #     self.obsU = np.zeros(self.noOfObstacles, dtype=precision)
            #     self.obsU_old = np.zeros_like(self.obsU)
            #     self.obsOmega = np.zeros_like(self.obsU)
            #     self.obsOmega_old = np.zeros_like(self.obsU)
            #     self.obsOrigin = np.zeros_like(self.obsU)
            self.obsU = np.array(self.obsU, dtype=precision)
            self.obsOmega = np.array(self.obsOmega, dtype=precision)
            self.obsOrigin = np.array(self.obsOrigin, dtype=precision)
            self.obsU_old = np.copy(self.obsU)
            self.obsOmega_old = np.copy(self.obsOmega)
            self.momentOfInertia = np.array(self.momentOfInertia,
                                            dtype=precision)
            self.obsMass = np.array(self.obsMass, dtype=precision)
            self.obstaclesGlobal = self.obstacles[:]
            return solid
        except KeyError as e:
            print("ERROR!")
            print(str(e) + " keyword missing in 'obstacle' dictionary",
                  flush=True)
            os._exit(1)

    def freeCommMemory(self):
        if self.allStatic is True:
            del self.vector_send_leftRight, self.vector_send_topBottom, \
                self.vector_recv_leftRight, self.vector_recv_topBottom
            del self.scalar_send_leftRight, self.scalar_send_topBottom, \
                self.scalar_recv_leftRight, self.scalar_recv_topBottom

    def computeFluidSolidNb(self, solid, mesh, lattice, fields, size,
                            mpiParams=None):
        if size == 1:
            nx, ny = 0, 0
            nProc_x, nProc_y = 1, 1
        else:
            nx, ny = mpiParams.nx, mpiParams.ny
            nProc_x, nProc_y = mpiParams.nProc_x, mpiParams.nProc_y
        for obsNo in range(self.noOfObstacles):
            fluidNbNodes, solidNbNodes = \
                findNb(solid, fields.fluidBoundary, fields.solidBoundary,
                       fields.boundaryNode, fields.boundaryNode,
                       mesh.Nx_global, mesh.Ny_global, lattice.c,
                       lattice.noOfDirections, obsNo, nx, ny,
                       nProc_x, nProc_y, size=1)
            self.fluidNbObstacle.append(fluidNbNodes)
            self.solidNbObstacle.append(solidNbNodes)

    def details(self):
        print(self.obstacles)
        print(self.noOfObstacles)
        print(self.fluidNbObstacle)
        print(self.obsNodes)
        print(self.obsModifiable)
        print(self.momentOfInertia)
        print(self.modifyObsFunc)
        print(self.obsU)
        print(self.obsOmega)
        print(self.obsOrigin)
        print(self.radius)
        print("degree of freedom: ", self.degreeOfFreedom)
        print("External Force: ", self.externalForce)
        print("External Torque: ", self.externalTorque)
        print("Translation: ", self.translation)
        print("Rotation: ", self.rotation)

    def writeModifiedObstacleData(self, timeStep, obsForces, obsTorque,
                                  obsU, obsOmega, obsOrigin, capForces,
                                  hydForces, capTorque, hydTorque):
        if not os.path.isdir('postProcessing'):
            os.makedirs('postProcessing')
        if not os.path.isdir('postProcessing/' + str(timeStep)):
            os.makedirs('postProcessing/' + str(timeStep))
        writeFile = open('postProcessing/' + str(timeStep) +
                         '/obstacleProperties.dat', 'w')
        writeFile.write('obs_ID'.ljust(12) + '\t' +
                        'obstacle'.ljust(12) + '\t' +
                        'center_x'.ljust(12) + '\t' +
                        'center_y'.ljust(12) + '\t' +
                        'F_x'.ljust(12) + '\t' +
                        'F_y'.ljust(12) + '\t' +
                        'Fc_x'.ljust(12) + '\t' +
                        'Fc_y'.ljust(12) + '\t' +
                        'Fh_x'.ljust(12) + '\t' +
                        'Fh_y'.ljust(12) + '\t' +
                        'u_x'.ljust(12) + '\t' +
                        'u_y'.ljust(12) + '\t' +
                        'torque'.ljust(12) + '\t' +
                        'torque_c'.ljust(12) + '\t' +
                        'torque_h'.ljust(12) + '\t' +
                        'omega'.ljust(12) + '\n')
        for itr in range(len(self.obstaclesGlobal)):
            writeFile.write(str(itr).ljust(12) + '\t' +
                            (self.obstaclesGlobal[itr]).ljust(12) + '\t' +
                            str(round(obsOrigin[itr, 0] - 1, 10)).ljust(12) + '\t' +
                            str(round(obsOrigin[itr, 1] - 1, 10)).ljust(12) + '\t' +
                            str(round(obsForces[itr, 0], 10)).ljust(12) + '\t' +
                            str(round(obsForces[itr, 1], 10)).ljust(12) + '\t' +
                            str(round(capForces[itr, 0], 10)).ljust(12) + '\t' +
                            str(round(capForces[itr, 1], 10)).ljust(12) + '\t' +
                            str(round(hydForces[itr, 0], 10)).ljust(12) + '\t' +
                            str(round(hydForces[itr, 1], 10)).ljust(12) + '\t' +
                            str(round(obsU[itr, 0], 10)).ljust(12) + '\t' +
                            str(round(obsU[itr, 1], 10)).ljust(12) + '\t' +
                            str(round(obsTorque[itr], 10)).ljust(12) + '\t' +
                            str(round(capTorque[itr], 10)).ljust(12) + '\t' +
                            str(round(hydTorque[itr], 10)).ljust(12) + '\t' +
                            str(round(obsOmega[itr], 10)).ljust(12) + '\n')
        writeFile.close()

    def solidMassSetup_cpu(self, parallel):
        if self.displaySolidMass:
            self.computeSolidMass = numba.njit(computeSolidMass,
                                               parallel=parallel,
                                               cache=False, nogil=True)

    def setupModifyObstacle_cpu(self, parallel):
        for itr in range(self.noOfObstacles):
            if self.modifyObsFunc[itr] is not None:
                self.modifyObsFunc[itr] = numba.njit(self.modifyObsFunc[itr],
                                                     parallel=parallel,
                                                     cache=False,
                                                     nogil=True)
        self.moveObstacle = numba.njit(moveObstacle, parallel=parallel,
                                       cache=False, nogil=True)
        self.initializeObsModifiableFields = \
            numba.njit(initializeObsModifiableFields, parallel=parallel,
                       cache=False, nogil=True)
        # self.moveObstacle = moveObstacle
        self.setInnerSolidNodePhi = \
            numba.njit(setInnerSolidNodePhi, parallel=parallel,
                       cache=False, nogil=True)
        self.updateGhostNodesHysteresis = \
            numba.njit(updateGhostNodesHysteresis, parallel=parallel,
                       cache=False, nogil=True)
        self.normalsCreatedNodes = \
            numba.njit(normalsCreatedNodes, parallel=parallel,
                       cache=False, nogil=True)
        self.reInitSolidFluidBoundaries = \
            numba.njit(reInitSolidFluidBoundaries, parallel=parallel,
                       cache=False, nogil=True)
        self.setVelocityObs = \
            numba.njit(setVelocityObs, parallel=parallel,
                       cache=False, nogil=True)

    def modifyObstacle(self, options, fields, mesh, lattice, boundary,
                       transport, collisionScheme, forcingScheme, size, comm,
                       timeStep, rank, precision, mpiParams=None,
                       phaseField=None, forces=None, torque=None,
                       hysteresis=False):
        if timeStep > 0:
            if mpiParams is None:
                nx, ny = 0, 0
                nProc_x, nProc_y = 0, 0
                N_local = self.N_local
            else:
                nx, ny = mpiParams.nx, mpiParams.ny
                nProc_x, nProc_y = mpiParams.nProc_x, mpiParams.nProc_y
                N_local = mpiParams.N_local
            if self.phase is False:
                rho_l, rho_g = precision(0), precision(0)
                phi_g = precision(0)
                equilibriumFuncFluid = collisionScheme.equilibriumFunc
                equilibriumArgsFluid = collisionScheme.equilibriumArgs
                equilibriumFuncPhase = collisionScheme.equilibriumFunc
                equilibriumArgsPhase = collisionScheme.equilibriumArgs
                h_new = np.zeros((2, 2), dtype=precision)
                phi = np.zeros(2, dtype=precision)
                phi_old = np.zeros_like(phi)
                normalPhi = np.zeros((2, 2), dtype=precision)
                p = np.zeros(2, dtype=precision)
                massAdded = np.zeros(2, dtype=precision)
                deltaM = np.zeros(2, dtype=precision)
            else:
                rho_l, rho_g = transport.rho_l, transport.rho_g
                phi_g = transport.phi_g
                equilibriumFuncFluid = collisionScheme.equilibriumFuncFluid
                equilibriumArgsFluid = collisionScheme.equilibriumArgsFluid
                equilibriumFuncPhase = collisionScheme.equilibriumFuncPhase
                equilibriumArgsPhase = collisionScheme.equilibriumArgsPhase
                h_new = fields.h_new
                phi = fields.phi
                phi_old = fields.phi_old
                normalPhi = fields.normalPhi
                p = fields.p
                massAdded = fields.massAdded
                deltaM = fields.deltaM
            args = (fields.solid_old, fields.solid, phi_old, phi,
                    fields.forceCreation, fields.torqueCreation,
                    fields.forceDes, fields.torqueDes, mesh.Nx, mesh.Ny)
            self.initializeObsModifiableFields(*args, phase=self.phase)
            if size > 1:
                comm.Barrier()
                args = (mesh.Nx, mesh.Ny,
                        fields.rho, self.scalar_send_topBottom,
                        self.scalar_recv_topBottom, self.scalar_send_leftRight,
                        self.scalar_recv_leftRight, mpiParams.nx,
                        mpiParams.ny, mpiParams.nProc_x,
                        mpiParams.nProc_y, comm)
                proc_boundary(*args)
                comm.Barrier()
            for itr in range(self.noOfObstacles):
                if self.modifyObsFunc[itr] is not None:
                    calculated = False
                    # acceleration = 1.6666666666666666e-10
                    # self.obsU[itr, 0] = acceleration * timeStep
                    # U = 1e-4
                    # endTime = 600000
                    # omega = 2 * np.pi / (0.5 * endTime)
                    # self.obsU[itr, 0] = U * np.sin(omega * timeStep)
                    # self.setVelocityObs(mesh.Nx, mesh.Ny, fields.solid,
                    #                     fields.u, fields.boundaryNode,
                    #                     self.obsU[itr])
                    if self.velType[itr] == 'calculated':
                        args = (fields.solid, fields.procBoundary, fields.
                                boundaryNode, self.momentOfInertia[itr],
                                self.obsMass[itr], self.obsOmega,
                                self.obsOmega_old, self.obsU,
                                self.obsU_old, forces[itr],
                                torque[itr], forcingScheme.gravity, fields.u,
                                self.obsOrigin, mesh.Nx, mesh.Ny,
                                mesh.Nx_global, mesh.Ny_global, nx,
                                ny, nProc_x, nProc_y, N_local, size, boundary.
                                x_periodic, boundary.y_periodic,
                                self.reconstructSolidFunc[itr],
                                self.reconstructSolidArgs[itr],
                                self.externalForce[itr],
                                self.externalTorque[itr],
                                self.translation[itr], self.rotation[itr],
                                itr)
                    elif self.velType[itr] == 'fixedVelocity':
                        args = (self.obsMass[itr], self.obsU, self.obsOrigin,
                                mesh.Nx_global, mesh.Ny_global, itr,
                                boundary.x_periodic, boundary.y_periodic)
                    self.modifyObsFunc[itr](*args)
                    if self.velType[itr] == 'calculated':
                        obsVel = np.array([0., 0.], dtype=precision)
                        obsOmega = 0
                        calculated = True
                    elif self.velType[itr] == 'fixedVelocity':
                        obsVel = self.obsU[itr]
                        obsOmega = self.obsOmega[itr]
                    # if timeStep == 1000:
                    #     print(self.obsOrigin[itr], itr)
                    args = (fields.solid_old, fields.solid,
                            fields.procBoundary, fields.boundaryNode,
                            self.obsDensity[itr], fields. rho, p, phi, phi_old,
                            normalPhi, massAdded, deltaM, fields.u, rho_l,
                            rho_g, phi_g, fields.f_new, h_new,
                            self.obsOrigin[itr], obsVel, obsOmega, fields.
                            forceCreation, fields.torqueCreation, fields.
                            forceDes, fields.torqueDes, lattice.c, lattice.w,
                            lattice.noOfDirections, equilibriumFuncFluid,
                            equilibriumArgsFluid,
                            equilibriumFuncPhase, equilibriumArgsPhase,
                            mesh.Nx, mesh.Ny, mesh.Nx_global, mesh.Ny_global,
                            nx, ny, nProc_x, nProc_y, N_local, size,
                            boundary.x_periodic, boundary.y_periodic,
                            calculated, self.reconstructSolidFunc[itr],
                            self.reconstructSolidArgs[itr],
                            self.translation[itr], self.rotation[itr],
                            options.phase, phaseField.segregation, itr)
                    self.moveObstacle(*args)
                    options.forceReconstruction[itr] = \
                        np.sum(fields.forceDes, axis=0) - \
                        np.sum(fields.forceCreation, axis=0)
                    options.torqueReconstruction[itr] = \
                        np.sum(fields.torqueDes) -\
                        np.sum(fields.torqueCreation)
                    if hysteresis is True:
                        pass
                        # args = (phi, phi_old, fields.solid, fields.solid_old,
                        #         fields.procBoundary, fields.boundaryNode,
                        #         lattice.c, lattice.w, lattice.noOfDirections,
                        #         size, nx, ny, nProc_x, nProc_y, mesh.Nx,
                        #         mesh.Ny, itr)
                        # self.updateGhostNodesHysteresis(*args)
                        # args = (phi, fields.solid, fields.solid_old,
                        #         fields.procBoundary, fields.boundaryNode,
                        #         fields.gradPhi, fields.normalPhi,
                        #         fields.lapPhi, lattice.c, lattice.w,
                        #         lattice.cs_2, lattice.noOfDirections, size,
                        #         nx, ny, nProc_x, nProc_y, mesh.Nx, mesh.Ny,
                        #         itr)
                        # self.normalsCreatedNodes(*args)
                else:
                    options.forceReconstruction[itr] = np.zeros(2, precision)
                    options.torqueReconstruction[itr] = precision(0)
            if size > 1:
                comm.Barrier()
                args = (mesh.Nx, mesh.Ny,
                        fields.u, self.vector_send_topBottom,
                        self.vector_recv_topBottom, self.vector_send_leftRight,
                        self.vector_recv_leftRight, mpiParams.nx,
                        mpiParams.ny, mpiParams.nProc_x,
                        mpiParams.nProc_y, comm)
                proc_boundaryGradTerms(*args, inner=True)
                comm.Barrier()
                args = (mesh.Nx, mesh.Ny,
                        fields.solid, self.vector_send_topBottom,
                        self.vector_recv_topBottom, self.vector_send_leftRight,
                        self.vector_recv_leftRight, mpiParams.nx,
                        mpiParams.ny, mpiParams.nProc_x,
                        mpiParams.nProc_y, comm)
                proc_boundaryGradTerms(*args, inner=True)
                comm.Barrier()
            # Set phase field at inner nodes to zero
            if self.phase is True:
                self.setInnerSolidNodePhi(fields.phi, fields.solid,
                                          fields.boundaryNode, fields.
                                          procBoundary, lattice.c,
                                          lattice.noOfDirections, mesh.Nx,
                                          mesh.Ny, size, self.phase)
                if size > 1:
                    args = (mesh.Nx, mesh.Ny, fields.phi,
                            self.scalar_send_topBottom,
                            self.scalar_recv_topBottom,
                            self.scalar_send_leftRight,
                            self.scalar_recv_leftRight,
                            mpiParams.nx, mpiParams.ny, mpiParams.nProc_x,
                            mpiParams.nProc_y, comm)
                    proc_boundaryGradTerms(*args, inner=True)
                    comm.Barrier()
        # if (self.writeProperties is True and timeStep >= 19995
        #         and timeStep <= 20005):
        if (self.writeProperties is True and timeStep %
                self.writeInterval == 0):
            if size > 1:
                obsForces, obsTorque, obsU, obsOmega, obsOrigin = \
                    self.gatherProperties(forces, torque, size, rank,
                                          comm, precision)
            else:
                obsForces = np.array(forces, dtype=precision)
                obsTorque = np.array(torque, dtype=precision)
                obsU = self.obsU
                obsOmega = self.obsOmega
                obsOrigin = self.obsOrigin
                capForces = np.array(options.capForces, dtype=precision)
                hydForces = np.array(options.hydForces, dtype=precision)
                capTorque = np.array(options.capTorque, dtype=precision)
                hydTorque = np.array(options.hydTorque, dtype=precision)
            if rank == 0:
                self.writeModifiedObstacleData(timeStep, obsForces, obsTorque,
                                               obsU, obsOmega, obsOrigin,
                                               capForces, hydForces,
                                               capTorque, hydTorque)

    def gatherProperties(self, forces, torque, size, rank, comm,
                         precision):
        if rank == 0:
            obsTorque = np.zeros(len(self.obstaclesGlobal),
                                 dtype=precision)
            obsOmega = np.zeros(len(self.obstaclesGlobal), dtype=precision)
            obsU = np.zeros((len(self.obstaclesGlobal), 2), dtype=precision)
            obsForces = np.zeros((len(self.obstaclesGlobal), 2),
                                 dtype=precision)
            obsOrigin = np.zeros((len(self.obstaclesGlobal), 2),
                                 dtype=precision)
            for i in range(size):
                if i == 0:
                    for itr, name in enumerate(self.obstaclesGlobal):
                        for itr_local, local_name in enumerate(self.obstacles):
                            if name == local_name:
                                obsForces[itr, 0] = forces[itr_local][0]
                                obsForces[itr, 1] = forces[itr_local][1]
                                obsU[itr, 0] = self.obsU[itr_local][0]
                                obsU[itr, 1] = self.obsU[itr_local][1]
                                obsOmega[itr] = self.obsOmega[itr_local]
                                obsTorque[itr] = torque[itr_local]
                                obsOrigin[itr, 0] = \
                                    self.obsOrigin[itr_local][0]
                                obsOrigin[itr, 1] = \
                                    self.obsOrigin[itr_local][1]
                else:
                    obstacles_local = comm.recv(source=i, tag=1*i)
                    obsOmega_local = comm.recv(source=i, tag=2*i)
                    obsU_local = comm.recv(source=i, tag=3*i)
                    obsOrigin_local = comm.recv(source=i, tag=4*i)
                    for itr, name in enumerate(self.obstaclesGlobal):
                        for itr_local, local_name in \
                                enumerate(obstacles_local):
                            if name == local_name:
                                obsForces[itr, 0] = forces[itr_local][0]
                                obsForces[itr, 1] = forces[itr_local][1]
                                obsU[itr, 0] = obsU_local[itr_local][0]
                                obsU[itr, 1] = obsU_local[itr_local][1]
                                obsOmega[itr] = obsOmega_local[itr_local]
                                obsTorque[itr] = torque[itr_local]
                                obsOrigin[itr, 0] = \
                                    obsOrigin_local[itr_local][0]
                                obsOrigin[itr, 1] = \
                                    obsOrigin_local[itr_local][1]
            return obsForces, obsTorque, obsU, obsOmega, obsOrigin
        else:
            comm.send(self.obstacles, dest=0, tag=1*rank)
            comm.send(self.obsOmega, dest=0, tag=2*rank)
            comm.send(self.obsU, dest=0, tag=3*rank)
            comm.send(self.obsOrigin, dest=0, tag=4*rank)
            return 0, 0, 0, 0, 0

    def setupModifyObstacle_cuda(self, precision):
        self.momentOfInertia_device = \
            cuda.to_device(self.momentOfInertia)
        self.obsU_device = cuda.to_device(self.obsU)
        self.obsU_old_device = cuda.to_device(self.obsU_old)
        for itr in range(self.noOfObstacles):
            self.obsNodes_device.append(cuda.to_device(self.obsNodes[itr]))
            temp = np.full(self.obsNodes[itr].shape[0],
                           fill_value=self.obsOmega[itr],
                           dtype=precision)
            temp = cuda.to_device(temp)
            self.obsOmega_device.append(temp)
            self.obsOmega_old_device.append(temp)
            if self.modifyObsFunc[itr] is not None:
                self.modifyObsFunc_device.append(setObsVelocity_cuda)
                self.modifyObsNo_device.append(itr)
                temp = cuda.to_device(self.obsOrigin[itr])
                self.obsOrigin_device.append(temp)
            else:
                self.modifyObsFunc_device.append(None)
                self.modifyObsNo_device.append(-1)
                temp = cuda.to_device(np.zeros(2, dtype=np.int64))
                self.obsOrigin_device.append(temp)
        self.modifyObsNo_device = \
            cuda.to_device(np.array(self.modifyObsNo_device, dtype=np.int32))

    def modifyObstacle_cuda(self, torque, device, timeStep,
                            blocks, n_threads):
        for itr in range(self.noOfObstacles):
            if self.modifyObsFunc_device[itr] is not None:
                torque_device = cuda.to_device(torque)
                args = (self.obsNodes_device[itr],
                        self.momentOfInertia_device[itr],
                        self.obsOmega_device[itr],
                        self.obsOmega_old_device[itr], torque_device[itr],
                        device.u, self.obsOrigin_device[itr], device.Nx[0],
                        device.Ny[0], self.modifyObsNo_device[itr])
                self.modifyObsFunc_device[itr][blocks, n_threads](*args)
            if (self.writeProperties is True and
                    timeStep % self.writeInterval == 0):
                self.obsOrigin[itr] = self.obsOrigin_device[itr].copy_to_host()
                self.obsOmega[itr] = self.obsOmega_device[itr][0:1].\
                    copy_to_host()[0]
        if (self.writeProperties is True and
                timeStep % self.writeInterval == 0):
            self.writeModifiedObstacleData(timeStep, torque, self.obsOmega,
                                           self.obsOrigin)
