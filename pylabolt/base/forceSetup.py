import numpy as np
import numba
from numba import prange


class forceSetup:
    def __init__(self):
        self.surfaceNodes = []
        self.surfaceNames = []
        self.surfaceInvList = []        # Only at boundaries
        self.surfaceOutList = []        # Only at boundaries
        self.noOfSurfaces = 0
        self.obstacleFlag = []          # differentiate obstacle and boundary
        self.forces = []
        self.forceFunc = force
        self.surfaceNamesGlobal = []

    def gatherObstacleNodes(self, obstacle):
        for itr in range(obstacle.noOfObstacles):
            self.surfaceNames.append(obstacle.obstacles[itr])
            self.surfaceNamesGlobal.append(obstacle.obstacles[itr])
            self.surfaceNodes.append(obstacle.fluidNbObstacle[itr])
            self.surfaceInvList.append(np.zeros(1, dtype=np.int32))
            self.surfaceOutList.append(np.zeros(1, dtype=np.int32))
            self.obstacleFlag.append(1)
            self.noOfSurfaces += 1

    def gatherBoundaryNodes(self, boundary):
        keyList = boundary.points.keys()
        itr = 0
        for key in keyList:
            for num, surfaces in enumerate(boundary.points[key]):
                if boundary.boundaryEntity[itr] == 'wall':
                    self.surfaceNames.append(key + '_' + str(num))
                    self.surfaceNamesGlobal.append(key + '_' + str(num))
                    self.surfaceNodes.append(boundary.faceList[itr])
                    self.surfaceInvList.append(boundary.invDirections[itr])
                    self.surfaceOutList.append(boundary.outDirections[itr])
                    self.noOfSurfaces += 1
                    self.obstacleFlag.append(0)
                itr += 1

    def forceCalc(self, fields, lattice, mesh, precision):
        self.forces = []
        for itr in range(self.noOfSurfaces):
            temp = np.zeros((self.surfaceNodes[itr].shape[0], 2),
                            dtype=precision)
            self.forceFunc(fields.f, fields.f_new, fields.solid,
                           fields.procBoundary, self.surfaceNodes[itr],
                           self.surfaceInvList[itr], self.surfaceOutList[itr],
                           lattice.c, lattice.invList, self.obstacleFlag[itr],
                           mesh.Nx, mesh.Ny, temp, lattice.noOfDirections)
            self.forces.append(np.sum(temp, axis=0))

    def details(self, rank, mesh, flag):
        print('\n\n\n')
        print(rank, self.surfaceNames)
        print(rank, self.surfaceNamesGlobal)
        print(rank, self.surfaceNodes)
        print(rank, self.obstacleFlag)
        if flag == 'all':
            nx, ny = mesh.Nx_global, mesh.Ny_global
        else:
            nx, ny = mesh.Nx, mesh.Ny
        check = np.zeros((nx, ny), dtype=np.int32)
        for itr in range(self.noOfSurfaces):
            for ind in self.surfaceNodes[itr]:
                i, j = int(ind / ny), int(ind % ny)
                check[i, j] = itr + 1
        print('\n')
        if flag == 'all':
            print(check.T)
        else:
            print(check[1:-1, 1:-1].T)

    def setupForcesParallel_cpu(self, parallel):
        self.forceFunc = numba.njit(self.forceFunc,
                                    parallel=parallel,
                                    cache=False,
                                    nogil=True)


def force(f, f_new, solid, procBoundary, surfaceNodes, surfaceInvList,
          surfaceOutList, c, invList, obstacleFlag, Nx, Ny, forces,
          noOfDirections):
    for itr in prange(surfaceNodes.shape[0]):
        ind = surfaceNodes[itr]
        if procBoundary[ind] != 1:
            if obstacleFlag == 0:
                for k in range(surfaceOutList.shape[0]):
                    forces[itr, 0] += ((- f[ind, surfaceOutList[k]] *
                                       c[surfaceOutList[k], 0]) +
                                       (f_new[ind, surfaceInvList[k]]
                                       * c[surfaceInvList[k], 0]))
                    forces[itr, 1] += ((- f[ind, surfaceOutList[k]] *
                                       c[surfaceOutList[k], 1]) +
                                       (f_new[ind, surfaceInvList[k]]
                                       * c[surfaceInvList[k], 1]))
            elif obstacleFlag == 1:
                i, j = int(ind / Ny), int(ind % Ny)
                for k in range(noOfDirections):
                    i_nb = int(i + c[k, 0] + Nx) % Nx
                    j_nb = int(j + c[k, 1] + Ny) % Ny
                    ind_nb = i_nb * Ny + j_nb
                    if solid[ind_nb, 0] == 1:
                        forces[itr, 0] += ((- f[ind, k] * c[k, 0]) +
                                           (f_new[ind, invList[k]]
                                           * c[invList[k], 0]))
                        forces[itr, 1] += ((- f[ind, k] * c[k, 1]) +
                                           (f_new[ind, invList[k]]
                                           * c[invList[k], 1]))
