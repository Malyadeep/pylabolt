import numpy as np
import numba
import os
import sys
from numba import (prange, cuda)


class options:
    def __init__(self, rank, precision, mesh):
        workingDir = os.getcwd()
        sys.path.append(workingDir)
        try:
            from simulation import options
        except ImportError:
            self.computeForces = False
            self.computeTorque = False
            if rank == 0:
                print('No options specified!')
            return
        try:
            self.computeForces = options['computeForces']
        except KeyError:
            self.computeForces = False
        try:
            self.computeTorque = options['computeTorque']
            if self.computeTorque is True:
                try:
                    self.x_ref = options['x_ref']
                    if not isinstance(self.x_ref, list):
                        print("ERROR! 'x_ref' must be a list containing" +
                              " reference coordinates : [x, y]")
                    else:
                        self.x_ref = np.array(self.x_ref, dtype=precision)
                        self.x_ref_idx = np.int64(np.divide(self.x_ref,
                                                            mesh.delX))
                except KeyError:
                    if rank == 0:
                        print("ERROR! 'x_ref' is a mandatory entry to" +
                              " compute torque!")
                    os._exit(1)
            else:
                self.x_ref_idx = np.zeros(2)
        except KeyError:
            self.computeTorque = False
            self.x_ref_idx = np.zeros(2)
        if self.computeForces is False and self.computeTorque is False:
            if rank == 0:
                print("No valid option specified!")
            return
        self.surfaceNodes = []
        self.surfaceNames = []
        self.surfaceInvList = []        # Only at boundaries
        self.surfaceOutList = []        # Only at boundaries
        self.noOfSurfaces = 0
        self.obstacleFlag = []          # differentiate obstacle and boundary
        self.forces = []
        self.torque = []
        self.forceTorqueFunc = forceTorque
        self.surfaceNamesGlobal = []
        self.N_local = np.zeros((2, 2, 0), dtype=np.int32)

        # Cuda device data
        self.surfaceNodes_device = []
        self.surfaceInvList_device = []
        self.surfaceOutList_device = []
        self.obstacleFlag_device = []
        self.x_ref_idx_device = []
        self.computeForces_device = np.full(1, fill_value=False,
                                            dtype=np.bool_)
        self.computeTorque_device = np.full(1, fill_value=False,
                                            dtype=np.bool_)

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

    def forceTorqueCalc(self, fields, lattice, mesh, precision,
                        size, mpiParams=None):
        self.forces = []
        self.torque = []
        for itr in range(self.noOfSurfaces):
            tempForces = np.zeros((self.surfaceNodes[itr].shape[0], 2),
                                  dtype=precision)
            tempTorque = np.zeros((self.surfaceNodes[itr].shape[0]),
                                  dtype=precision)
            if size == 1:
                nx, ny = 0, 0
                nProc_x, nProc_y = 1, 1
            else:
                nx, ny = mpiParams.nx, mpiParams.ny
                nProc_x, nProc_y = mpiParams.nProc_x, mpiParams.nProc_y
            args = (fields.f, fields.f_new, fields.solid,
                    fields.procBoundary, self.surfaceNodes[itr],
                    self.surfaceInvList[itr], self.surfaceOutList[itr],
                    lattice.c, lattice.invList, self.obstacleFlag[itr],
                    mesh.Nx, mesh.Ny, tempForces, tempTorque,
                    lattice.noOfDirections, self.computeForces,
                    self.computeTorque, self.x_ref_idx, self.N_local,
                    nx, ny, nProc_x, nProc_y, size)
            self.forceTorqueFunc(*args)
            self.forces.append(np.sum(tempForces, axis=0))
            self.torque.append(np.sum(tempTorque, axis=0))

    def details(self, rank, mesh, solid, u, flag='local'):
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
            print()
            print(solid.reshape(mesh.Nx, mesh.Ny, 2)[:, :, 0].T)
            print()

        else:
            print(check[1:-1, 1:-1].T)
            print()
            print(solid.reshape(mesh.Nx, mesh.Ny, 2)[:, :, 0])
            print()

    def setupForcesParallel_cpu(self, parallel):
        self.forceTorqueFunc = numba.njit(self.forceTorqueFunc,
                                          parallel=parallel,
                                          cache=False,
                                          nogil=True)

    def setupForcesParallel_cuda(self):
        self.x_ref_idx_device = cuda.to_device(self.x_ref_idx)
        self.computeForces_device = \
            cuda.to_device(np.array([self.computeForces], dtype=np.bool_))
        self.computeTorque_device = \
            cuda.to_device(np.array([self.computeTorque], dtype=np.bool_))
        self.obstacleFlag_device = np.array(self.obstacleFlag, dtype=np.int32)
        for itr in range(self.noOfSurfaces):
            self.surfaceNodes_device.append(cuda.to_device(self.
                                            surfaceNodes[itr]))
            self.surfaceInvList_device.append(cuda.to_device(self.
                                              surfaceInvList[itr]))
            self.surfaceOutList_device.append(cuda.to_device(self.
                                              surfaceOutList[itr]))

    def forceTorqueCalc_cuda(self, device, precision, n_threads, blocks):
        self.forces = []
        self.torque = []
        tempSum = np.zeros(2, dtype=precision)
        for itr in range(self.noOfSurfaces):
            tempForces = np.zeros((self.surfaceNodes[itr].shape[0], 2),
                                  dtype=precision)
            tempTorque = np.zeros((self.surfaceNodes[itr].shape[0]),
                                  dtype=precision)
            tempForces_device = cuda.to_device(tempForces)
            tempTorque_device = cuda.to_device(tempTorque)
            args = (device.f, device.f_new, device.solid,
                    self.surfaceNodes_device[itr], self.surfaceInvList[itr],
                    self.surfaceOutList_device[itr], device.c, device.invList,
                    self.obstacleFlag_device[itr], device.Nx[0], device.Ny[0],
                    tempForces_device, tempTorque_device,
                    device.noOfDirections[0], self.computeForces_device[0],
                    self.computeTorque_device[0], self.x_ref_idx_device)
            forceTorque_cuda[blocks, n_threads](*args)
            temp = []
            temp.append(forceTorqueReduce(tempForces_device[:, 0]))
            temp.append(forceTorqueReduce(tempForces_device[:, 1]))
            self.forces.append(np.array(temp, dtype=precision))
            tempSum = forceTorqueReduce(tempTorque_device)
            self.torque.append(tempSum)

    def writeForces(self, timeStep, names, forces, torque):
        if not os.path.isdir('postProcessing'):
            os.makedirs('postProcessing')
        if not os.path.isdir('postProcessing/' + str(timeStep)):
            os.makedirs('postProcessing/' + str(timeStep))
        if self.computeForces is True:
            writeFile = open('postProcessing/' + str(timeStep) + '/forces.dat',
                             'w')
            writeFile.write('surface_ID'.ljust(12) + '\t' +
                            'surface'.ljust(12) + '\t' +
                            'F_x'.ljust(12) + '\t' +
                            'F_y'.ljust(12) + '\n')
            for itr in range(len(names)):
                writeFile.write(str(itr).ljust(12) + '\t' +
                                (names[itr]).ljust(12) + '\t' +
                                str(round(forces[itr, 0], 10)).ljust(12) + '\t'
                                + str(round(forces[itr, 1], 10)).ljust(12)
                                + '\n')
            writeFile.close()
        if self.computeTorque is True:
            writeFile = open('postProcessing/' + str(timeStep) + '/torque.dat',
                             'w')
            writeFile.write('surface_ID'.ljust(12) + '\t' +
                            'surface'.ljust(12) + '\t' +
                            'T'.ljust(12) + '\n')
            for itr in range(len(names)):
                writeFile.write(str(itr).ljust(12) + '\t' +
                                (names[itr]).ljust(12) + '\t' +
                                str(round(torque[itr], 10)).ljust(12)
                                + '\n')
            writeFile.close()


def forceTorque(f, f_new, solid, procBoundary, surfaceNodes, surfaceInvList,
                surfaceOutList, c, invList, obstacleFlag, Nx, Ny, forces,
                torque, noOfDirections, computeForces, computeTorque,
                x_ref, N_local, nx, ny, nProc_x, nProc_y, size):
    for itr in prange(surfaceNodes.shape[0]):
        ind = surfaceNodes[itr]
        if procBoundary[ind] != 1:
            if obstacleFlag == 0:
                for k in range(surfaceOutList.shape[0]):
                    value_0 = ((- f[ind, surfaceOutList[k]] *
                               c[surfaceOutList[k], 0]) +
                               (f_new[ind, surfaceInvList[k]]
                               * c[surfaceInvList[k], 0]))
                    value_1 = ((- f[ind, surfaceOutList[k]] *
                               c[surfaceOutList[k], 1]) +
                               (f_new[ind, surfaceInvList[k]]
                               * c[surfaceInvList[k], 1]))
                    if computeForces is True:
                        forces[itr, 0] += value_0
                        forces[itr, 1] += value_1
                    if computeTorque is True:
                        i_local, j_local = int(ind / Ny), int(ind % Ny)
                        if size == 1:
                            i, j = i_local, j_local
                        else:
                            Nx_local, Ny_local = Nx - 2, Ny - 2
                            if nx == nProc_x - 1:
                                Nx_local = N_local[nx - 1, ny, 0]
                            if ny == nProc_y - 1:
                                Ny_local = N_local[nx, ny - 1, 1]
                            i = nx * Nx_local + i_local - 1
                            j = ny * Ny_local + j_local - 1
                        r_0 = i - x_ref[0]
                        r_1 = j - x_ref[1]
                        torque[itr] += (r_0 * value_1 - r_1 * value_0)
            elif obstacleFlag == 1:
                i, j = int(ind / Ny), int(ind % Ny)
                for k in range(noOfDirections):
                    i_nb = int(i + c[k, 0] + Nx) % Nx
                    j_nb = int(j + c[k, 1] + Ny) % Ny
                    ind_nb = i_nb * Ny + j_nb
                    if solid[ind_nb, 0] == 1:
                        value_0 = ((- f[ind, k] * c[k, 0]) +
                                   (f_new[ind, invList[k]]
                                   * c[invList[k], 0]))
                        value_1 = ((- f[ind, k] * c[k, 1]) +
                                   (f_new[ind, invList[k]]
                                   * c[invList[k], 1]))
                        if computeForces is True:
                            forces[itr, 0] += value_0
                            forces[itr, 1] += value_1
                        if computeTorque is True:
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
                            r_0 = i_global - x_ref[0]
                            r_1 = j_global - x_ref[1]
                            torque[itr] += (r_0 * value_1 - r_1 * value_0)


@cuda.reduce
def forceTorqueReduce(a, b):
    return a + b


@cuda.jit
def forceTorque_cuda(f, f_new, solid, surfaceNodes, surfaceInvList,
                     surfaceOutList, c, invList, obstacleFlag, Nx, Ny, forces,
                     torque, noOfDirections, computeForces, computeTorque,
                     x_ref):
    ind = cuda.grid(1)
    if ind < Nx * Ny:
        for itr in range(surfaceNodes.shape[0]):
            if ind == surfaceNodes[itr]:
                if obstacleFlag == 0:
                    for k in range(surfaceOutList.shape[0]):
                        value_0 = ((- f[ind, surfaceOutList[k]] *
                                   c[surfaceOutList[k], 0]) +
                                   (f_new[ind, surfaceInvList[k]]
                                   * c[surfaceInvList[k], 0]))
                        value_1 = ((- f[ind, surfaceOutList[k]] *
                                   c[surfaceOutList[k], 1]) +
                                   (f_new[ind, surfaceInvList[k]]
                                   * c[surfaceInvList[k], 1]))
                        if computeForces is True:
                            forces[itr, 0] += value_0
                            forces[itr, 1] += value_1
                        if computeTorque is True:
                            i, j = np.int64(ind / Ny), np.int64(ind % Ny)
                            r_0 = i - x_ref[0]
                            r_1 = j - x_ref[1]
                            torque[itr] += (r_0 * value_1 - r_1 * value_0)
                elif obstacleFlag == 1:
                    i, j = int(ind / Ny), int(ind % Ny)
                    for k in range(noOfDirections):
                        i_nb = int(i + c[k, 0] + Nx) % Nx
                        j_nb = int(j + c[k, 1] + Ny) % Ny
                        ind_nb = i_nb * Ny + j_nb
                        if solid[ind_nb, 0] == 1:
                            value_0 = ((- f[ind, k] * c[k, 0]) +
                                       (f_new[ind, invList[k]]
                                       * c[invList[k], 0]))
                            value_1 = ((- f[ind, k] * c[k, 1]) +
                                       (f_new[ind, invList[k]]
                                       * c[invList[k], 1]))
                            if computeForces is True:
                                forces[itr, 0] += value_0
                                forces[itr, 1] += value_1
                            if computeTorque is True:
                                r_0 = i - x_ref[0]
                                r_1 = j - x_ref[1]
                                torque[itr] += (r_0 * value_1 - r_1 * value_0)
            else:
                continue
    else:
        return
