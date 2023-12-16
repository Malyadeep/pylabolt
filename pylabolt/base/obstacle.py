import os
import sys
import numba
import numpy as np
from numba import (prange, cuda)


from pylabolt.base import obstacleTypes
from pylabolt.parallel.MPI_comm import (proc_boundary,
                                        proc_boundaryGradTerms)


@numba.njit
def findNb(solid, boundaryNode, procBoundary, Nx, Ny, c, noOfDirections,
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
                    if (solid[ind_nb, 0] == 1 and solid[ind_nb, 1] == obsNo):
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
                        solidNbNodes.append(ind)
                        break
            else:
                continue
    return np.array(fluidNbNodes, dtype=np.int64), \
        np.array(solidNbNodes, dtype=np.int64)


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


def translateSolid(obsMass, obsU, obsOrigin, Nx_global, Ny_global, obsNo):
    obsOrigin[obsNo, 0] = (obsOrigin[obsNo, 0] + obsU[obsNo, 0] - 1
                           + Nx_global - 2) % (Nx_global - 2) + 1
    obsOrigin[obsNo, 1] = (obsOrigin[obsNo, 1] + obsU[obsNo, 1] - 1
                           + Ny_global - 2) % (Ny_global - 2) + 1


def setObsVelocity(solid, procBoundary, boundaryNode, momentOfInertia,
                   obsMass, obsOmega, obsOmega_old, obsU, obsU_old, force,
                   torque, u, obsOrigin, Nx, Ny, Nx_global, Ny_global, nx, ny,
                   nProc_x, nProc_y, N_local, size, x_periodic, y_periodic,
                   reconstructSolidFunc, reconstructSolidArgs, obsNo):
    obsOmega[obsNo] = obsOmega_old[obsNo] + torque / momentOfInertia
    obsU[obsNo, 0] = obsU_old[obsNo, 0] + force[0] / obsMass
    obsU[obsNo, 1] = obsU_old[obsNo, 1] + force[1] / obsMass
    # obsOmega[obsNo] = 0
    # obsU[obsNo, 0] = 0.01
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
            u[ind, 0] = obsU[obsNo, 0] - r * obsOmega[obsNo] * np.sin(theta)
            u[ind, 1] = obsU[obsNo, 1] + r * obsOmega[obsNo] * np.cos(theta)
    obsOmega_old[obsNo] = obsOmega[obsNo]
    obsU_old[obsNo, 0] = obsU[obsNo, 0]
    obsU_old[obsNo, 1] = obsU[obsNo, 1]
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
    for k in range(noOfDirections):
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
def createPropertiesPhase(ind, boundaryNode, solid_old, rho, p, phi, c, w,
                          noOfDirections, rho_l, rho_g, phi_g, Nx, Ny, nx,
                          ny, nProc_x, nProc_y, size):
    pSum = 0
    phiSum = 0
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
            phiSum += w[k] * phi[ind_nb]
            pSum += w[k] * p[ind_nb]
            denominator += w[k]
    phi[ind] = phiSum / (denominator + 1e-17)
    p[ind] = pSum / (denominator + 1e-17)
    rho[ind] = rho_g + (phi[ind] - phi_g) * (rho_l - rho_g)


def moveObstacle(solid_old, solid, procBoundary, boundaryNode, rhoSolid, rho,
                 p, phi, u, rho_l, rho_g, phi_g, f_new, h_new, obsOrigin,
                 obsVel, forceCreation, torqueCreation, forceDes, torqueDes,
                 c, w, noOfDirections, equilibriumFuncFluid,
                 equilibriumArgsFluid, equilibriumFuncPhase,
                 equilibriumArgsPhase, Nx, Ny, Nx_global,
                 Ny_global, nx, ny, nProc_x, nProc_y, N_local, size,
                 x_periodic, y_periodic, calculated, reconstructSolidFunc,
                 reconstructSolidArgs, phase, obsNo):
    created = 0
    destroyed = 0
    for ind in prange(Nx * Ny):
        solid_old[ind, 0] = solid[ind, 0]
        solid_old[ind, 1] = solid[ind, 1]
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
            if insideSolid:
                if solid_old[ind, 0] == 1 and solid_old[ind, 1] == obsNo:
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
                        p[ind] = 0
                        # phi[ind] = 0
                    destroyed += 1
                    if calculated is False:
                        u[ind, 0] = obsVel[0]
                        u[ind, 1] = obsVel[1]
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
                                              solid_old, rho, p, phi, c, w,
                                              noOfDirections, rho_l, rho_g,
                                              phi_g, Nx, Ny, nx, ny, nProc_x,
                                              nProc_y, size)
                        equilibriumFuncFluid(f_new[ind], u[ind], p[ind],
                                             *equilibriumArgsFluid)
                        equilibriumFuncPhase(h_new[ind], u[ind], phi[ind],
                                             *equilibriumArgsPhase)
                    solid[ind, 0] = 0
                    solid[ind, 1] = 0
                    forceCreation[ind, 0] = rho[ind] * u[ind, 0]
                    forceCreation[ind, 1] = rho[ind] * u[ind, 1]
                    torqueCreation[ind] = (x * u[ind, 1] - y * u[ind, 0]) *\
                        rho[ind]
                    created += 1


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
                self.obstacles.remove('displaySolidMass')
                self.displaySolidMass = obstacle['displaySolidMass']
                # print(self.obstacles)
            for obsNo, obsName in enumerate(self.obstacles):
                obstacleType = obstacle[obsName]['type']
                rho_s = obstacle[obsName]['rho_s']
                self.noOfObstacles += 1
                if obstacleType == 'circle':
                    center = obstacle[obsName]['center']
                    radius = obstacle[obsName]['radius']
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
                    else:
                        self.allStatic = False
                        velDict = obstacle[obsName]['U_def']
                        if velDict['type'] == 'fixedTranslational':
                            velType = 'fixedTranslational'
                            self.velType.append(velType)
                            self.reconstructSolidFunc.\
                                append(obstacleTypes.reconstructCircle)
                            self.reconstructSolidArgs.append((radius, center))
                            velValue = velDict['value']
                            if not isinstance(velValue, list):
                                if rank == 0:
                                    print("ERROR!", flush=True)
                                    print("For 'fixedTranslational' type "
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
                                    print("For 'fixedTranslational' type " +
                                          "velocity, origin must be " +
                                          "a list and angular velocity " +
                                          "must be a float",
                                          flush=True)
                                os._exit(1)
                            self.obsOmega.append(precision(0))
                            self.obsOmega.append(velOmega)
                            self.obsOrigin.append(np.int64(np.divide(velOrigin,
                                                  mesh.delX)) + np.ones(2,
                                                  dtype=np.int64))
                            nodes, momentOfInertia, obsMass, obsDensity = \
                                obstacleTypes.\
                                circle(center, radius, solid, initialFields,
                                       rho_s, mesh, obsNo, velType=velType,
                                       velValue=velValue, periodic=periodic)
                            self.obsNodes.append(nodes)
                            self.momentOfInertia.append(momentOfInertia)
                            self.obsMass.append(obsMass)
                            self.obsDensity.append(obsDensity)
                            self.modifyObsFunc.append(translateSolid)
                            self.obsModifiable = True
                            self.writeProperties = velDict['write']
                            if self.writeProperties is True:
                                self.writeInterval = \
                                    velDict['writeInterval']
                        if velDict['type'] == 'calculated':
                            velType = 'calculated'
                            self.velType.append(velType)
                            self.reconstructSolidFunc.\
                                append(obstacleTypes.reconstructCircle)
                            self.reconstructSolidArgs.append((radius, center))
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
                            self.obsOmega.append(velOmega)
                            self.obsOrigin.append(np.int64(np.divide(velOrigin,
                                                  mesh.delX)) + np.ones(2,
                                                  dtype=np.int64))
                            nodes, momentOfInertia, obsMass, obsDensity = \
                                obstacleTypes.\
                                circle(center, radius, solid, initialFields,
                                       rho_s, mesh, obsNo, velType=velType,
                                       velValue=velValue, periodic=periodic)
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
                        elif (velDict['type'] == 'fixedRotational' or
                              velDict['type'] == 'calculatedRotational'):
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
                            if (options.computeTorque is False and
                                    velType == 'calculatedRotational'):
                                if rank == 0:
                                    print("ERROR!", flush=True)
                                    print("For 'calculatedRotational' type " +
                                          "velocity in obstacle, " +
                                          "'computeTorque' must be 'True'!",
                                          flush=True)
                                    os._exit(1)
                            velOrigin = np.array(velOrigin, dtype=precision)
                            velOmega = precision(velOmega)
                            self.obsU.append(np.zeros(2, dtype=precision))
                            self.obsOmega.append(velOmega)
                            self.obsOrigin.append(np.int64(np.divide(velOrigin,
                                                  mesh.delX)) + np.ones(2,
                                                  dtype=np.int64))
                            if velType == 'calculatedRotational':
                                self.modifyObsFunc.append(setObsVelocity)
                                self.obsModifiable = True
                                self.writeProperties = velDict['write']
                                if self.writeProperties is True:
                                    self.writeInterval = \
                                        velDict['writeInterval']
                            else:
                                self.modifyObsFunc.append(None)
                            nodes, momentOfInertia, obsMass, obsDensity = \
                                obstacleTypes.\
                                circle(center, radius, solid, initialFields,
                                       rho_s, mesh, obsNo, velType=velType,
                                       velOrigin=velOrigin,
                                       velOmega=velOmega, periodic=periodic)
                            self.obsNodes.append(nodes)
                            self.momentOfInertia.append(momentOfInertia)
                            self.obsMass.append(obsMass)
                            self.obsDensity.append(obsDensity)
                elif obstacleType == 'rectangle':
                    boundingBox = obstacle[obsName]['boundingBox']
                    periodic = obstacle[obsName]['periodic']
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
                    else:
                        self.allStatic = False
                        velDict = obstacle[obsName]['U_def']
                        if (velDict['type'] == 'fixedTranslational' or
                                velDict['type'] == 'calculated'):
                            velType = velDict['type']
                            self.velType.append(velType)
                            length = np.abs(boundingBox[1, 0] -
                                            boundingBox[0, 0])
                            breadth = np.abs(boundingBox[1, 1] -
                                             boundingBox[0, 1])
                            self.reconstructSolidFunc.\
                                append(obstacleTypes.reconstructRectangle)
                            self.reconstructSolidArgs.append((length, breadth))
                            velValue = velDict['value']
                            if not isinstance(velValue, list):
                                if rank == 0:
                                    print("ERROR!", flush=True)
                                    print("For 'fixedTranslational' type "
                                          + "velocity, value must be a list",
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
                            if velType == 'calculated':
                                velOmega = velDict['angularVelocity']
                                if not isinstance(velOmega, float):
                                    if rank == 0:
                                        print("ERROR!", flush=True)
                                        print("For 'calculated' type and" +
                                              " angular velocity must be a"
                                              + " float", flush=True)
                                    os._exit(1)
                            else:
                                velOmega = precision(0)
                            # print(velOmega, velValue, velOrigin)
                            self.obsOmega.append(velOmega)
                            self.obsOrigin.append(np.int64(np.divide(velOrigin,
                                                  mesh.delX)) + np.ones(2,
                                                  dtype=np.int64))
                            if velType == 'fixedTranslational':
                                self.modifyObsFunc.append(translateSolid)
                            elif velType == 'calculated':
                                self.modifyObsFunc.append(setObsVelocity)
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
                        elif (velDict['type'] == 'fixedRotational' or
                              velDict['type'] == 'calculatedRotational'):
                            if rank == 0:
                                print("ERROR!", flush=True)
                                print("Only 'fixedTranslational' velocity" +
                                      " type is supported for 'rectangle' " +
                                      "obstacle", flush=True)
                                os._exit(1)
                            # velType = velDict['type']
                            # self.velType.append(velType)
                            # velOrigin = velDict['origin']
                            # velOmega = velDict['angularVelocity']
                            # if (not isinstance(velOrigin, list) and not
                            #         isinstance(velOmega, float)):
                            #     if rank == 0:
                            #         print("ERROR!", flush=True)
                            #         print("For 'fixedRotational' type " +
                            #               "velocity, origin must be " +
                            #               "a list and angular velocity " +
                            #               "must be a float",
                            #               flush=True)
                            #     os._exit(1)
                            # if (options.computeTorque is False and
                            #         velType == 'calculatedRotational'):
                            #     if rank == 0:
                            #         print("ERROR!", flush=True)
                            #         print("For 'calculatedRotational' type " +
                            #               "velocity in obstacle, " +
                            #               "'computeTorque' must be 'True'!",
                            #               flush=True)
                            #         os._exit(1)
                            # velOrigin = np.array(velOrigin, dtype=precision)
                            # velOmega = precision(velOmega)
                            # self.obsU.append(np.zeros(2, dtype=precision))
                            # self.obsOmega.append(velOmega)
                            # self.obsOrigin.append(np.divide(velOrigin,
                            #                       mesh.delX) + np.ones(2,
                            #                       dtype=np.int64))
                            # if velType == 'calculatedRotational':
                            #     self.modifyObsFunc.append(setObsVelocity)
                            #     self.obsModifiable = True
                            #     self.writeProperties = velDict['write']
                            #     if self.writeProperties is True:
                            #         self.writeInterval = \
                            #             velDict['writeInterval']
                            # else:
                            #     self.modifyObsFunc.append(None)
                            # nodes, momentOfInertia, obsMass, obsDensity = \
                            #     obstacleTypes.\
                            #     rectangle(boundingBox, solid, u, rho, rho_s,
                            #               mesh, obsNo, velType=velType,
                            #               velOrigin=velOrigin,
                            #               velOmega=velOmega, periodic=periodic)
                            # self.obsNodes.append(nodes)
                            # self.momentOfInertia.append(momentOfInertia)
                            # self.obsMass.append(obsMass)
                            # self.obsDensity.append(obsDensity)
                elif obstacleType == 'inclinedRectangle':
                    centerLine = obstacle[obsName]['centerLine']
                    width = obstacle[obsName]['width']
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
            if self.allStatic is False:
                self.obsU = np.array(self.obsU, dtype=precision)
                self.obsOmega = np.array(self.obsOmega, dtype=precision)
                self.obsOrigin = np.array(self.obsOrigin, dtype=precision)
                self.obsU_old = np.copy(self.obsU)
                self.obsOmega_old = np.copy(self.obsOmega)
            else:
                self.obsU = np.zeros(self.noOfObstacles, dtype=precision)
                self.obsU_old = np.zeros_like(self.obsU)
                self.obsOmega = np.zeros_like(self.obsU)
                self.obsOmega_old = np.zeros_like(self.obsU)
                self.obsOrigin = np.zeros_like(self.obsU)
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
                findNb(solid, fields.boundaryNode, fields.boundaryNode,
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

    def writeModifiedObstacleData(self, timeStep, obsForces, obsTorque,
                                  obsU, obsOmega, obsOrigin):
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
                        'u_x'.ljust(12) + '\t' +
                        'u_y'.ljust(12) + '\t' +
                        'torque'.ljust(12) + '\t' +
                        'omega'.ljust(12) + '\n')
        for itr in range(len(self.obstaclesGlobal)):
            writeFile.write(str(itr).ljust(12) + '\t' +
                            (self.obstaclesGlobal[itr]).ljust(12) + '\t' +
                            str(round(obsOrigin[itr, 0] - 1, 10)).ljust(12) + '\t' +
                            str(round(obsOrigin[itr, 1] - 1, 10)).ljust(12) + '\t' +
                            str(round(obsForces[itr, 0], 10)).ljust(12) + '\t' +
                            str(round(obsForces[itr, 1], 10)).ljust(12) + '\t' +
                            str(round(obsU[itr, 0], 10)).ljust(12) + '\t' +
                            str(round(obsU[itr, 1], 10)).ljust(12) + '\t' +
                            str(round(obsTorque[itr], 10)).ljust(12) + '\t'
                            + str(round(obsOmega[itr], 10)).ljust(12)
                            + '\n')
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
        # self.moveObstacle = moveObstacle

    def modifyObstacle(self, options, fields, mesh, lattice, boundary,
                       transport, collisionScheme, size, comm, timeStep, rank,
                       precision, mpiParams=None, phaseField=None,
                       forces=None, torque=None):
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
            p = np.zeros(2, dtype=precision)
        else:
            rho_l, rho_g = transport.rho_l, transport.rho_g
            phi_g = transport.phi_g
            equilibriumFuncFluid = collisionScheme.equilibriumFuncFluid
            equilibriumArgsFluid = collisionScheme.equilibriumArgsFluid
            equilibriumFuncPhase = collisionScheme.equilibriumFuncPhase
            equilibriumArgsPhase = collisionScheme.equilibriumArgsPhase
            h_new = fields.h_new
            phi = fields.phi
            p = fields.p
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
                if self.velType[itr] == 'calculated':
                    args = (fields.solid, fields.procBoundary, fields.
                            boundaryNode, self.momentOfInertia[itr],
                            self.obsMass[itr], self.obsOmega,
                            self.obsOmega_old, self.obsU,
                            self.obsU_old, forces[itr],
                            torque[itr], fields.u, self.obsOrigin, mesh.Nx,
                            mesh.Ny, mesh.Nx_global, mesh.Ny_global, nx, ny,
                            nProc_x, nProc_y, N_local, size, boundary.
                            x_periodic, boundary.y_periodic,
                            self.reconstructSolidFunc[itr],
                            self.reconstructSolidArgs[itr], itr)
                elif self.velType[itr] == 'fixedTranslational':
                    args = (self.obsMass[itr], self.obsU, self.obsOrigin,
                            mesh.Nx_global, mesh.Ny_global, itr)
                self.modifyObsFunc[itr](*args)
                solid_old = np.zeros_like(fields.solid)
                forceCreation = np.zeros_like(fields.u)
                forceDes = np.zeros_like(forceCreation)
                torqueCreation = np.zeros_like(fields.rho)
                torqueDes = np.zeros_like(torqueCreation)
                if self.velType[itr] == 'calculated':
                    obsVel = np.array([0., 0.], dtype=precision)
                    calculated = True
                elif self.velType[itr] == 'fixedTranslational':
                    obsVel = self.obsU[itr]
                args = (solid_old, fields.solid, fields.procBoundary,
                        fields.boundaryNode, self.obsDensity[itr], fields.rho,
                        p, phi, fields.u, rho_l, rho_g, phi_g, fields.f_new,
                        h_new, self.obsOrigin[itr], obsVel,
                        forceCreation, torqueCreation, forceDes, torqueDes,
                        lattice.c, lattice.w, lattice.noOfDirections,
                        equilibriumFuncFluid, equilibriumArgsFluid,
                        equilibriumFuncPhase, equilibriumArgsPhase,
                        mesh.Nx, mesh.Ny, mesh.Nx_global, mesh.Ny_global,
                        nx, ny, nProc_x, nProc_y, N_local, size,
                        boundary.x_periodic, boundary.y_periodic, calculated,
                        self.reconstructSolidFunc[itr],
                        self.reconstructSolidArgs[itr], options.phase, itr)
                self.moveObstacle(*args)
                options.forceReconstruction[itr] = \
                    np.sum(forceDes, axis=0) - np.sum(forceCreation, axis=0)
                options.torqueReconstruction[itr] = \
                    np.sum(torqueDes) - np.sum(torqueCreation)
            else:
                options.forceReconstruction[itr] = np.zeros(2, precision)
                options.torqueReconstruction[itr] = precision(0)
        # print(options.forceReconstruction[itr], options.torqueReconstruction[itr])
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
            if rank == 0:
                self.writeModifiedObstacleData(timeStep, obsForces, obsTorque,
                                               obsU, obsOmega, obsOrigin)

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
