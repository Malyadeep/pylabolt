import os
import sys
import numba
import numpy as np


def circle(center, radius, solid, u, mesh, obsNo, velType='fixedTranslational',
           velValue=[0.0, 0.0], velOrigin=0, velOmega=0):
    center_idx = np.int64(np.divide(center, mesh.delX))
    radius_idx = np.int64(radius/mesh.delX)
    origin_idx = np.int64(np.divide(velOrigin, mesh.delX))
    for i in range(mesh.Nx_global):
        for j in range(mesh.Ny_global):
            if ((i - center_idx[0])*(i - center_idx[0]) +
                    (j - center_idx[1])*(j - center_idx[1]) <=
                    radius_idx*radius_idx):
                ind = i * mesh.Ny_global + j
                solid[ind, 0] = 1
                solid[ind, 1] = obsNo
                if velType == 'fixedTranslational':
                    u[ind, 0] = velValue[0]
                    u[ind, 1] = velValue[1]
                elif velType == 'fixedRotational':
                    x = i - origin_idx[0]
                    y = j - origin_idx[1]
                    theta = np.arctan2(y, x)
                    r = np.sqrt(x**2 + y**2)
                    u[ind, 0] = - r * velOmega * np.sin(theta)
                    u[ind, 1] = r * velOmega * np.cos(theta)


def rectangle(boundingBox, solid, mesh, obsNo):
    boundingBox_idx = np.int32(np.divide(boundingBox, mesh.delX))
    for i in range(mesh.Nx_global):
        for j in range(mesh.Ny_global):
            if (i >= boundingBox_idx[0, 0] and i <= boundingBox_idx[1, 0]
                    and j >= boundingBox_idx[0, 1] and
                    j <= boundingBox_idx[1, 1]):
                ind = i * mesh.Ny_global + j
                solid[ind, 0] = 1
                solid[ind, 1] = obsNo


def circularConfinement(center, radius, solid, u, mesh, obsNo,
                        velType='fixedTranslational', velValue=[0.0, 0.0],
                        velOrigin=0, velOmega=0):
    center_idx = np.int64(np.divide(center, mesh.delX))
    radius_idx = np.int64(radius/mesh.delX)
    origin_idx = np.int64(np.divide(velOrigin, mesh.delX))
    for i in range(mesh.Nx_global):
        for j in range(mesh.Ny_global):
            if ((i - center_idx[0])*(i - center_idx[0]) +
                    (j - center_idx[1])*(j - center_idx[1]) >=
                    radius_idx*radius_idx):
                ind = i * mesh.Ny_global + j
                solid[ind, 0] = 1
                solid[ind, 1] = obsNo
                if velType == 'fixedTranslational':
                    u[ind, 0] = velValue[0]
                    u[ind, 1] = velValue[1]
                elif velType == 'fixedRotational':
                    x = i - origin_idx[0]
                    y = j - origin_idx[1]
                    theta = np.arctan2(y, x)
                    r = np.sqrt(x**2 + y**2)
                    u[ind, 0] = - r * velOmega * np.sin(theta)
                    u[ind, 1] = r * velOmega * np.cos(theta)


@numba.njit
def findNb(solid, Nx, Ny, c, noOfDirections, size, obsNo):
    nbNodes = []
    for i in range(Nx):
        for j in range(Ny):
            ind = i * Ny + j
            if solid[ind, 0] != 1:
                for k in range(noOfDirections):
                    i_nb = int(i + c[k, 0] + Nx) % Nx
                    j_nb = int(j + c[k, 1] + Ny) % Ny
                    ind_nb = i_nb * Ny + j_nb
                    if solid[ind_nb, 0] == 1 and solid[ind_nb, 1] == obsNo:
                        nbNodes.append(ind)
                        break
            else:
                continue
    return np.array(nbNodes, dtype=np.int64)


class obstacleSetup:
    def __init__(self):
        self.obstacles = {}
        self.noOfObstacles = 0
        self.fluidNbObstacle = []

    def setObstacle(self, mesh, precision, initialFields, rank):
        workingDir = os.getcwd()
        sys.path.append(workingDir)
        try:
            from simulation import obstacle
            if rank == 0:
                print('Reading Obstacle data...', flush=True)
        except ImportError:
            if rank == 0:
                print('No obstacle defined!', flush=True)
            return
        try:
            if rank == 0:
                solid = np.full((mesh.Nx_global * mesh.Ny_global, 2),
                                fill_value=0, dtype=np.int32)
                u = initialFields.u
            else:
                return 0, 0
            if len(obstacle.keys()) == 0:
                if rank == 0:
                    print('Reading Obstacle data done!', flush=True)
                return solid
            else:
                self.obstacles = list(obstacle.keys())
            for obsNo, obsName in enumerate(self.obstacles):
                obstacleType = obstacle[obsName]['type']
                self.noOfObstacles += 1
                if obstacleType == 'circle':
                    center = obstacle[obsName]['center']
                    radius = obstacle[obsName]['radius']
                    if isinstance(center, list) and isinstance(radius, float):
                        center = np.array(center, dtype=np.float64)
                        radius = np.float64(radius)
                    else:
                        if rank == 0:
                            print("ERROR!", flush=True)
                            print("For 'circle' type obstacle center must"
                                  + " be a list and radius must be a float",
                                  flush=True)
                        os._exit(1)
                    static = obstacle[obsName]['static']
                    if static is True:
                        circle(center, radius, solid, u, mesh, obsNo)
                    else:
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
                            circle(center, radius, solid, u, mesh, obsNo,
                                   velType=velType, velValue=velValue)
                        elif velDict['type'] == 'fixedRotational':
                            velType = 'fixedRotational'
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
                            circle(center, radius, solid, u, mesh, obsNo,
                                   velType=velType, velOrigin=velOrigin,
                                   velOmega=velOmega)
                elif obstacleType == 'rectangle':
                    boundingBox = obstacle[obsName]['boundingBox']
                    if isinstance(boundingBox, list):
                        boundingBox = np.array(boundingBox, dtype=np.float64)
                    else:
                        print("ERROR!", flush=True)
                        print("For 'rectangle' type obstacle bounding box"
                              + "  must be a list", flush=True)
                        os._exit(1)
                    rectangle(boundingBox, solid, mesh, obsNo)
                elif obstacleType == 'circularConfinement':
                    center = obstacle[obsName]['center']
                    radius = obstacle[obsName]['radius']
                    if isinstance(center, list) and isinstance(radius, float):
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
                        circularConfinement(center, radius, solid, u,
                                            mesh, obsNo)
                    else:
                        velDict = obstacle[obsName]['U_def']
                        if velDict['type'] == 'fixedTranslational':
                            velType = 'fixedTranslational'
                            velValue = velDict['value']
                            if not isinstance(velValue, list):
                                if rank == 0:
                                    print("ERROR!", flush=True)
                                    print("For 'circle' type obstacle center "
                                          + " must be a list and radius must" +
                                          " be a float",
                                          flush=True)
                                os._exit(1)
                            velValue = np.array(velValue, dtype=precision)
                            circularConfinement(center, radius, solid, u,
                                                mesh, obsNo, velType=velType,
                                                velValue=velValue)
                        elif velDict['type'] == 'fixedRotational':
                            velType = 'fixedRotational'
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
                            circularConfinement(center, radius, solid, u, mesh,
                                                obsNo, velType=velType,
                                                velOrigin=velOrigin,
                                                velOmega=velOmega)
                else:
                    print("ERROR!")
                    print("Unsupported obstacle type!", flush=True)
                    os._exit(1)
                if rank == 0:
                    print('Reading Obstacle data done!', flush=True)
            return solid
        except KeyError as e:
            print("ERROR!")
            print(str(e) + " keyword missing in 'obstacle' dictionary",
                  flush=True)
            os._exit(1)

    def computeFluidNb(self, solid, mesh, lattice, size):
        for obsNo in range(self.noOfObstacles):
            nbNodes = findNb(solid, mesh.Nx_global, mesh.Ny_global,
                             lattice.c, lattice.noOfDirections, size, obsNo)
            self.fluidNbObstacle.append(nbNodes)

    def details(self):
        print(self.obstacles)
        print(self.noOfObstacles)
        print(self.fluidNbObstacle)
