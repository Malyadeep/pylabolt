import os
import sys
import numba
import numpy as np
from numba import (prange, cuda)


from pylabolt.base import obstacleTypes
from pylabolt.parallel.MPI_comm import proc_boundary


@numba.njit
def findNb(solid, boundaryNode, Nx, Ny, c, noOfDirections, size, obsNo):
    fluidNbNodes = []
    solidNbNodes = []
    for i in range(Nx):
        for j in range(Ny):
            ind = i * Ny + j
            if solid[ind, 0] != 1 and boundaryNode[ind] != 1:
                for k in range(noOfDirections):
                    i_nb = int(i + c[k, 0] + Nx) % Nx
                    j_nb = int(j + c[k, 1] + Ny) % Ny
                    ind_nb = i_nb * Ny + j_nb
                    if (solid[ind_nb, 0] == 1 and solid[ind_nb, 1] == obsNo
                            and boundaryNode[ind_nb] != 1):
                        fluidNbNodes.append(ind)
                        break
            elif solid[ind, 0] != 0 and boundaryNode[ind] != 1:
                for k in range(noOfDirections):
                    i_nb = int(i + c[k, 0] + Nx) % Nx
                    j_nb = int(j + c[k, 1] + Ny) % Ny
                    ind_nb = i_nb * Ny + j_nb
                    if (solid[ind_nb, 0] == 0 and solid[ind, 1] == obsNo
                            and boundaryNode[ind_nb] != 1):
                        solidNbNodes.append(ind)
                        break
            else:
                continue
    return np.array(fluidNbNodes, dtype=np.int64),\
        np.array(solidNbNodes, dtype=np.int64)


@cuda.jit
def setOmegaFromTorque_cuda(obsNodes, momentOfInertia, omega, omega_old,
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


def setOmegaFromTorque(obsNodes, momentOfInertia, omega, omega_old,
                       torque, u, obsOrigin, Nx, Ny, nx, ny, nProc_x, nProc_y,
                       N_local, size, obsNo):
    omega[obsNo] = omega_old[obsNo] + torque / momentOfInertia
    for itr in prange(obsNodes.shape[0]):
        ind = obsNodes[itr]
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
        x = i_global - obsOrigin[0]
        y = j_global - obsOrigin[1]
        theta = np.arctan2(y, x)
        r = np.sqrt(x**2 + y**2)
        u[ind, 0] = - r * omega[obsNo] * np.sin(theta)
        u[ind, 1] = r * omega[obsNo] * np.cos(theta)
    omega_old[obsNo] = omega[obsNo]


class obstacleSetup:
    def __init__(self, size, mesh, precision, phase=False):
        self.obstacles = []
        self.obstaclesGlobal = []
        self.noOfObstacles = 0
        self.fluidNbObstacle = []
        self.solidNbObstacle = []
        self.obsNodes = []
        self.modifyObsFunc = []
        self.obsModifiable = False
        self.momentOfInertia = []
        self.obsOmega_old = []
        self.obsOmega = []
        self.obsU_old = []
        self.obsU = []
        self.obsOrigin = []
        self.allStatic = True
        self.phase = phase
        self.N_local = np.zeros((2, 2, 0), dtype=np.int64)
        if size > 1:
            self.obs_send_topBottom = np.zeros((mesh.Nx + 2, 2),
                                               dtype=precision)
            self.obs_recv_topBottom = np.zeros((mesh.Nx + 2, 2),
                                               dtype=precision)
            self.obs_send_leftRight = np.zeros((mesh.Ny + 2, 2),
                                               dtype=precision)
            self.obs_recv_leftRight = np.zeros((mesh.Ny + 2, 2),
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
                u = initialFields.u
                rho = initialFields.rho
            else:
                return 0
            if len(obstacle.keys()) == 0:
                if rank == 0:
                    print('Reading Obstacle data done!', flush=True)
                return solid
            else:
                self.obstacles = list(obstacle.keys())
            for obsNo, obsName in enumerate(self.obstacles):
                obstacleType = obstacle[obsName]['type']
                rho_s = obstacle[obsName]['rho_s']
                self.noOfObstacles += 1
                if obstacleType == 'circle':
                    center = obstacle[obsName]['center']
                    radius = obstacle[obsName]['radius']
                    if (isinstance(center, list) and isinstance(radius, float)
                            or isinstance(radius, int)):
                        center = np.array(center, dtype=precision)
                        radius = precision(radius)
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
                        nodes, momentOfInertia = \
                            obstacleTypes.circle(center, radius, solid,
                                                 u, rho, rho_s, mesh, obsNo)
                        self.obsNodes.append(nodes)
                        self.momentOfInertia.append(momentOfInertia)
                    else:
                        self.allStatic = False
                        velDict = obstacle[obsName]['U_def']
                        if velDict['type'] == 'fixedTranslational':
                            velType = 'fixedTranslational'
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
                            self.obsOmega.append(precision(0))
                            self.obsOrigin.append(np.array([0, 0],
                                                  dtype=precision))
                            self.modifyObsFunc.append(None)
                            nodes, momentOfInertia = \
                                obstacleTypes.\
                                circle(center, radius, solid, u, rho, rho_s,
                                       mesh, obsNo, velType=velType,
                                       velValue=velValue)
                            self.obsNodes(nodes)
                            self.momentOfInertia.append(momentOfInertia)
                        elif (velDict['type'] == 'fixedRotational' or
                              velDict['type'] == 'calculatedRotational'):
                            velType = velDict['type']
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
                                self.modifyObsFunc.append(setOmegaFromTorque)
                                self.obsModifiable = True
                                self.writeProperties = velDict['write']
                                if self.writeProperties is True:
                                    self.writeInterval = \
                                        velDict['writeInterval']
                            else:
                                self.modifyObsFunc.append(None)
                            nodes, momentOfInertia = \
                                obstacleTypes.\
                                circle(center, radius, solid, u, rho, rho_s,
                                       mesh, obsNo, velType=velType,
                                       velOrigin=velOrigin,
                                       velOmega=velOmega)
                            self.obsNodes.append(nodes)
                            self.momentOfInertia.append(momentOfInertia)
                elif obstacleType == 'rectangle':
                    boundingBox = obstacle[obsName]['boundingBox']
                    if isinstance(boundingBox, list):
                        boundingBox = np.array(boundingBox, dtype=np.int64)
                    else:
                        print("ERROR!", flush=True)
                        print("For 'rectangle' type obstacle bounding box"
                              + "  must be a list", flush=True)
                        os._exit(1)
                    static = obstacle[obsName]['static']
                    if static is True:
                        self.modifyObsFunc.append(None)
                        nodes, momentOfInertia = \
                            obstacleTypes.rectangle(boundingBox, solid, u, rho,
                                                    rho_s, mesh, obsNo)
                        self.obsNodes.append(nodes)
                        self.momentOfInertia.append(momentOfInertia)
                    else:
                        self.allStatic = False
                        velDict = obstacle[obsName]['U_def']
                        if velDict['type'] == 'fixedTranslational':
                            velType = 'fixedTranslational'
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
                            self.obsOmega.append(precision(0))
                            self.obsOrigin.append(np.array([0, 0],
                                                  dtype=precision))
                            self.modifyObsFunc.append(None)
                            nodes, momentOfInertia = \
                                obstacleTypes.\
                                rectangle(boundingBox, solid, u, rho, rho_s,
                                          mesh, obsNo, velType=velType,
                                          velValue=velValue)
                            self.obsNodes.append(nodes)
                            self.momentOfInertia.append(momentOfInertia)
                        elif (velDict['type'] == 'fixedRotational' or
                              velDict['type'] == 'calculatedRotational'):
                            velType = velDict['type']
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
                            self.obsOrigin.append(np.divide(velOrigin,
                                                  mesh.delX) + np.ones(2,
                                                  dtype=np.int64))
                            if velType == 'calculatedRotational':
                                self.modifyObsFunc.append(setOmegaFromTorque)
                                self.obsModifiable = True
                                self.writeProperties = velDict['write']
                                if self.writeProperties is True:
                                    self.writeInterval = \
                                        velDict['writeInterval']
                            else:
                                self.modifyObsFunc.append(None)
                            nodes, momentOfInertia = \
                                obstacleTypes.\
                                rectangle(boundingBox, solid, u, rho, rho_s,
                                          mesh, obsNo, velType=velType,
                                          velOrigin=velOrigin,
                                          velOmega=velOmega)
                            self.obsNodes.append(nodes)
                            self.momentOfInertia.append(momentOfInertia)
                elif obstacleType == 'inclinedRectangle':
                    centerLine = obstacle[obsName]['centerLine']
                    width = obstacle[obsName]['width']
                    if (isinstance(centerLine, list) and
                            isinstance(width, float) or
                            isinstance(width, int)):
                        centerLine = np.array(centerLine, dtype=np.int64)
                        width = precision(width)
                    else:
                        print("ERROR!", flush=True)
                        print("For 'inclinedRectangle' type obstacle center"
                              + "line must be a list and width must be float"
                              + " or int.", flush=True)
                        os._exit(1)
                    static = obstacle[obsName]['static']
                    if static is True:
                        self.modifyObsFunc.append(None)
                        nodes, momentOfInertia = \
                            obstacleTypes.\
                            inclinedRectangle(centerLine, width, solid,
                                              u, rho, rho_s, mesh,
                                              obsNo, comm)
                        self.obsNodes.append(nodes)
                        self.momentOfInertia.append(momentOfInertia)
                    else:
                        self.allStatic = False
                        velDict = obstacle[obsName]['U_def']
                        if velDict['type'] == 'fixedTranslational':
                            velType = 'fixedTranslational'
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
                            self.obsOmega.append(precision(0))
                            self.obsOrigin.append(np.array([0, 0],
                                                  dtype=precision))
                            self.modifyObsFunc.append(None)
                            nodes, momentOfInertia = \
                                obstacleTypes.\
                                inclinedRectangle(centerLine, width, solid,
                                                  u, rho, rho_s, mesh,
                                                  obsNo, comm, velType=velType,
                                                  velValue=velValue)
                            self.obsNodes.append(nodes)
                            self.momentOfInertia.append(momentOfInertia)
                        elif (velDict['type'] == 'fixedRotational' or
                              velDict['type'] == 'calculatedRotational'):
                            velType = velDict['type']
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
                            self.obsOrigin.append(np.divide(velOrigin,
                                                  mesh.delX) + np.ones(2,
                                                  dtype=np.int64))
                            if velType == 'calculatedRotational':
                                self.modifyObsFunc.append(setOmegaFromTorque)
                                self.obsModifiable = True
                                self.writeProperties = velDict['write']
                                if self.writeProperties is True:
                                    self.writeInterval = \
                                        velDict['writeInterval']
                            else:
                                self.modifyObsFunc.append(None)
                            nodes, momentOfInertia = \
                                obstacleTypes.\
                                inclinedRectangle(centerLine, width, solid,
                                                  u, rho, rho_s, mesh,
                                                  obsNo, comm, velType=velType,
                                                  velOrigin=velOrigin,
                                                  velOmega=velOmega)
                            self.obsNodes.append(nodes)
                            self.momentOfInertia.append(momentOfInertia)
                elif obstacleType == 'circularConfinement':
                    center = obstacle[obsName]['center']
                    radius = obstacle[obsName]['radius']
                    if (isinstance(center, list) and isinstance(radius, float)
                            or isinstance(radius, int)):
                        center = np.array(center, dtype=np.float64)
                        radius = np.float64(radius)
                    else:
                        print("ERROR!", flush=True)
                        print("For 'circularConfinement' type obstacle center"
                              + " must be a list and radius must be a float",
                              flush=True)
                        os._exit(1)
                    static = obstacle[obsName]['static']
                    if static is True:
                        self.modifyObsFunc.append(None)
                        nodes, momentOfInertia = \
                            obstacleTypes.\
                            circularConfinement(center, radius, solid, u, rho,
                                                rho_s, mesh, obsNo)
                        self.obsNodes.append(nodes)
                        self.momentOfInertia.append(momentOfInertia)
                    else:
                        self.allStatic = False
                        velDict = obstacle[obsName]['U_def']
                        if velDict['type'] == 'fixedTranslational':
                            velType = 'fixedTranslational'
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
                            self.obsOmega.append(precision(0))
                            self.obsOrigin.append(np.array([0, 0],
                                                  dtype=precision))
                            self.modifyObsFunc.append(None)
                            nodes, momentOfInertia = \
                                obstacleTypes.\
                                circularConfinement(center, radius, solid, u,
                                                    rho, rho_s, mesh, obsNo,
                                                    velType=velType,
                                                    velValue=velValue)
                            self.obsNodes.append(nodes)
                            self.momentOfInertia.append(momentOfInertia)
                        elif (velDict['type'] == 'fixedRotational' or
                              velDict['type'] == 'calculatedRotational'):
                            velType = velDict['type']
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
                                self.modifyObsFunc.append(setOmegaFromTorque)
                                self.obsModifiable = True
                                self.writeProperties = velDict['write']
                                if self.writeProperties is True:
                                    self.writeInterval = \
                                        velDict['writeInterval']
                            else:
                                self.modifyObsFunc.append(None)
                            nodes, momentOfInertia = \
                                obstacleTypes.\
                                circularConfinement(center, radius, solid, u,
                                                    rho, rho_s, mesh, obsNo,
                                                    velType=velType,
                                                    velOrigin=velOrigin,
                                                    velOmega=velOmega)
                            self.obsNodes.append(nodes)
                            self.momentOfInertia.append(momentOfInertia)
                else:
                    print("ERROR!")
                    print("Unsupported obstacle type!", flush=True)
                    os._exit(1)
            if rank == 0:
                print('Reading Obstacle data done!', flush=True)
            if self.allStatic is False:
                self.obsU = np.array(self.obsU, dtype=precision)
                self.obsOmega = np.array(self.obsOmega, dtype=precision)
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
            self.obstaclesGlobal = self.obstacles[:]
            return solid
        except KeyError as e:
            print("ERROR!")
            print(str(e) + " keyword missing in 'obstacle' dictionary",
                  flush=True)
            os._exit(1)

    def freeCommMemory(self):
        if self.allStatic is True:
            del self.obs_send_leftRight, self.obs_send_topBottom,\
                self.obs_recv_leftRight, self.obs_recv_topBottom

    def computeFluidSolidNb(self, solid, mesh, lattice, fields, size):
        for obsNo in range(self.noOfObstacles):
            fluidNbNodes, solidNbNodes = \
                findNb(solid, fields.boundaryNode, mesh.Nx_global,
                       mesh.Ny_global, lattice.c, lattice.noOfDirections,
                       size, obsNo)
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

    def writeModifiedObstacleData(self, timeStep, torque, omega, origin):
        if not os.path.isdir('postProcessing'):
            os.makedirs('postProcessing')
        if not os.path.isdir('postProcessing/' + str(timeStep)):
            os.makedirs('postProcessing/' + str(timeStep))
        writeFile = open('postProcessing/' + str(timeStep) +
                         '/obstacleProperties.dat', 'w')
        writeFile.write('obs_ID'.ljust(12) + '\t' +
                        'obstacle'.ljust(12) + '\t' +
                        'ref_point'.ljust(12) + '\t' +
                        'torque'.ljust(12) + '\t' +
                        'omega'.ljust(12) + '\n')
        for itr in range(len(self.obstaclesGlobal)):
            if self.phase is True:
                torque_obs = np.sum(torque[itr])
            else:
                torque_obs = torque[itr]
            writeFile.write(str(itr).ljust(12) + '\t' +
                            (self.obstaclesGlobal[itr]).ljust(12) + '\t' +
                            str(origin[itr]).ljust(12) + '\t' +
                            str(round(torque_obs, 10)).ljust(12) + '\t'
                            + str(round(omega[itr], 10)).ljust(12)
                            + '\n')
        writeFile.close()

    def setupModifyObstacle_cpu(self, parallel):
        for itr in range(self.noOfObstacles):
            if self.modifyObsFunc[itr] is not None:
                self.modifyObsFunc[itr] = numba.njit(self.modifyObsFunc[itr],
                                                     parallel=parallel,
                                                     cache=False,
                                                     nogil=True)

    def modifyObstacle(self, torque, fields, mesh, size, comm, timeStep,
                       rank, precision, mpiParams=None):
        if mpiParams is None:
            nx, ny = 0, 0
            nProc_x, nProc_y = 0, 0
            N_local = self.N_local
        else:
            nx, ny = mpiParams.nx, mpiParams.ny
            nProc_x, nProc_y = mpiParams.nProc_x, mpiParams.nProc_y
            N_local = mpiParams.N_local
        for itr in range(self.noOfObstacles):
            if self.modifyObsFunc[itr] is not None:
                # print(self.obsOrigin[itr])
                if self.phase is True:
                    torque_obs = np.sum(torque[itr])
                else:
                    torque_obs = torque[itr]
                args = (self.obsNodes[itr], self.momentOfInertia[itr],
                        self.obsOmega, self.obsOmega_old,
                        torque_obs, fields.u, self.obsOrigin[itr],
                        mesh.Nx, mesh.Ny, nx, ny, nProc_x, nProc_y,
                        N_local, size, itr)
                self.modifyObsFunc[itr](*args)
        if size > 1:
            comm.Barrier()
            args = (mesh.Nx, mesh.Ny,
                    fields.u, self.obs_send_topBottom,
                    self.obs_recv_topBottom, self.obs_send_leftRight,
                    self.obs_recv_leftRight, mpiParams.nx,
                    mpiParams.ny, mpiParams.nProc_x,
                    mpiParams.nProc_y, comm)
            proc_boundary(*args)
            comm.Barrier()
        if (self.writeProperties is True and timeStep %
                self.writeInterval == 0):
            if size > 1:
                torque_obs, omega, origin = \
                    self.gatherProperties(torque, size, rank, comm, precision)
            else:
                torque_obs = torque
                omega = self.obsOmega
                origin = self.obsOrigin
            if rank == 0:
                self.writeModifiedObstacleData(timeStep, torque_obs,
                                               omega, origin)

    def gatherProperties(self, torque, size, rank, comm, precision):
        if rank == 0:
            if self.phase is True:
                torque_obs = np.zeros((len(self.obstaclesGlobal), 2),
                                      dtype=precision)
            else:
                torque_obs = np.zeros(len(self.obstaclesGlobal),
                                      dtype=precision)
            omega = np.zeros(len(self.obstaclesGlobal), dtype=precision)
            origin = np.zeros((len(self.obstaclesGlobal), 2), dtype=precision)
            for i in range(size):
                if i == 0:
                    for itr, name in enumerate(self.obstaclesGlobal):
                        for itr_local, local_name in enumerate(self.obstacles):
                            if name == local_name:
                                omega[itr] = self.obsOmega[itr_local]
                                if self.phase is True:
                                    torque_obs[itr, 0] = torque[itr_local, 0]
                                    torque_obs[itr, 1] = torque[itr_local, 1]
                                else:
                                    torque_obs[itr] = torque[itr_local]
                                origin[itr, 0] = self.obsOrigin[itr_local][0]
                                origin[itr, 1] = self.obsOrigin[itr_local][1]
                else:
                    obstacles_local = comm.recv(source=i, tag=1*i)
                    omega_local = comm.recv(source=i, tag=2*i)
                    origin_local = comm.recv(source=i, tag=3*i)
                    for itr, name in enumerate(self.obstaclesGlobal):
                        for itr_local, local_name in \
                                enumerate(obstacles_local):
                            if name == local_name:
                                omega[itr] = omega_local[itr_local]
                                if self.phase is True:
                                    torque_obs[itr, 0] = torque[itr_local, 0]
                                    torque_obs[itr, 1] = torque[itr_local, 1]
                                else:
                                    torque_obs[itr] = torque[itr_local]
                                origin[itr, 0] = origin_local[itr_local][0]
                                origin[itr, 1] = origin_local[itr_local][1]
            return torque_obs, omega, origin
        else:
            comm.send(self.obstacles, dest=0, tag=1*rank)
            comm.send(self.obsOmega, dest=0, tag=2*rank)
            comm.send(self.obsOrigin, dest=0, tag=3*rank)
            return 0, 0, 0

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
                self.modifyObsFunc_device.append(setOmegaFromTorque_cuda)
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
