import numpy as np
import numba
import os
import sys
from numba import (prange, cuda)

import pylabolt.parallel.cudaReduce as cudaReduce


@numba.njit
def solidNormal(surfaceNodes, solidNodes, solidNbNodesWhole, normal,
                normalFluid, solid, procBoundary, boundaryNode, cs_2,
                c, w, noOfDirections, Nx, Ny, size):
    for itr in prange(solidNodes.shape[0]):
        ind = solidNodes[itr]
        if (solid[ind, 0] != 0 and procBoundary[ind] != 1 and
                boundaryNode[ind] != 1):
            i, j = int(ind / Ny), int(ind % Ny)
            gradSolidSum_x, gradSolidSum_y = 0., 0.
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
                gradSolidSum_x += c[k, 0] * w[k] * solid[ind_nb, 0]
                gradSolidSum_y += c[k, 1] * w[k] * solid[ind_nb, 0]
            gradSolid_x = cs_2 * gradSolidSum_x
            gradSolid_y = cs_2 * gradSolidSum_y
            magGradSolid = np.sqrt(gradSolid_x * gradSolid_x +
                                   gradSolid_y * gradSolid_y)
            normal[itr, 0] = - gradSolid_x / (magGradSolid + 1e-17)
            normal[itr, 1] = - gradSolid_y / (magGradSolid + 1e-17)
            # if solid[int(i * Ny + j - 3), 0] == 1:
            #     normal[itr, 0] = -sinAngle
            #     normal[itr, 1] = cosAngle
    for itr in prange(surfaceNodes.shape[0]):
        ind = surfaceNodes[itr]
        solidNbNodesWhole[ind] = 1
        if (solid[ind, 0] != 1 and procBoundary[ind] != 1 and
                boundaryNode[ind] != 1):
            i, j = int(ind / Ny), int(ind % Ny)
            gradSolidSum_x, gradSolidSum_y = 0., 0.
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
                gradSolidSum_x += c[k, 0] * w[k] * solid[ind_nb, 0]
                gradSolidSum_y += c[k, 1] * w[k] * solid[ind_nb, 0]
            gradSolid_x = cs_2 * gradSolidSum_x
            gradSolid_y = cs_2 * gradSolidSum_y
            magGradSolid = np.sqrt(gradSolid_x * gradSolid_x +
                                   gradSolid_y * gradSolid_y)
            normalFluid[itr, 0] = - gradSolid_x / (magGradSolid + 1e-17)
            normalFluid[itr, 1] = - gradSolid_y / (magGradSolid + 1e-17)
            # if solid[int(i * Ny + j - 4), 0] == 1:
            #     normalFluid[itr, 0] = -sinAngle
            #     normalFluid[itr, 1] = cosAngle


class options:
    def __init__(self, rank, precision, mesh, wetting=False, phase=False):
        workingDir = os.getcwd()
        sys.path.append(workingDir)
        noOptions = False
        try:
            from simulation import options
        except ImportError:
            self.computeForces = False
            self.computeTorque = False
            if rank == 0:
                print('No options specified!')
            noOptions = True
        if noOptions is False:
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
                            self.x_ref = np.array(self.x_ref, dtype=np.int64)
                            self.x_ref_idx = \
                                np.int64(np.divide(self.x_ref, mesh.delX)) + \
                                np.ones(2, dtype=np.int64)
                    except KeyError:
                        if rank == 0:
                            print("ERROR! 'x_ref' is a mandatory entry to" +
                                  " compute torque!")
                        os._exit(1)
                else:
                    self.x_ref_idx = np.ones(2, dtype=np.int64)
            except KeyError:
                self.computeTorque = False
                self.x_ref_idx = np.ones(2, dtype=np.int64)
        if (self.computeForces is False and self.computeTorque is False
                and wetting is False):
            if rank == 0:
                print("Surface force computation not selected!")
            return
        elif wetting is True:
            if rank == 0:
                print("Wetting boundaries present!")
        if phase is True:
            self.phase = True
        self.surfaceNodes = []
        self.surfaceNormals = []
        self.surfaceNormalsFluid = []
        self.solidNbNodes = []
        self.phiInterpolated = []
        self.surfaceNames = []
        self.surfaceInvList = []        # Only at boundaries
        self.surfaceOutList = []        # Only at boundaries
        self.noOfSurfaces = 0
        self.obstacleFlag = []          # differentiate obstacle and boundary
        self.forces = []
        self.torque = []
        if self.phase is False:
            self.forceTorqueFunc = forceTorque
        else:
            self.forceTorqueFuncPhase = forceTorquePhase
        self.surfaceNamesGlobal = []
        self.N_local = np.zeros((2, 2, 0), dtype=np.int64)

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
            self.solidNbNodes.append(obstacle.solidNbObstacle[itr])
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
                    # self.solidNbNodes.\
                    #     append(np.concatenate((boundary.nbList[itr],
                    #            boundary.cornerList[itr])))
                    self.solidNbNodes.\
                        append(boundary.nbList[itr])
                    self.surfaceInvList.append(boundary.invDirections[itr])
                    self.surfaceOutList.append(boundary.outDirections[itr])
                    self.noOfSurfaces += 1
                    self.obstacleFlag.append(0)
                itr += 1

    def computeSolidNormals(self, fields, lattice, mesh, size, precision):
        for itr in range(self.noOfSurfaces):
            tempNormal = np.zeros((self.solidNbNodes[itr].shape[0], 2),
                                  dtype=precision)
            tempNormalFluid = np.zeros((self.surfaceNodes[itr].shape[0],
                                       2), dtype=precision)
            solidNormal(self.surfaceNodes[itr], self.solidNbNodes[itr],
                        fields.solidNbNodesWhole, tempNormal, tempNormalFluid,
                        fields.solid, fields.procBoundary, fields.boundaryNode,
                        lattice.cs_2, lattice.c, lattice.w,
                        lattice.noOfDirections, mesh.Nx, mesh.Ny, size)
            self.surfaceNormals.append(tempNormal)
            self.surfaceNormalsFluid.append(tempNormalFluid)
            temp = np.zeros(self.solidNbNodes[itr].shape[0], dtype=precision)
            self.phiInterpolated.append(temp)

    def forceTorqueCalc(self, fields, transport, lattice, mesh, precision,
                        size, timeStep, mpiParams=None, ref_index=None,
                        phaseField=None):
        self.forces = []
        self.torque = []
        for itr in range(self.noOfSurfaces):
            tempForces = np.zeros((self.surfaceNodes[itr].shape[0], 2),
                                  dtype=precision)
            tempTorque = np.zeros((self.surfaceNodes[itr].shape[0]),
                                  dtype=precision)
            tempCapForces = np.zeros((self.solidNbNodes[itr].shape[0], 2),
                                     dtype=precision)
            tempCapTorque = np.zeros((self.solidNbNodes[itr].shape[0]),
                                     dtype=precision)
            idx = self.x_ref_idx
            if ref_index is not None and len(ref_index) > 0:
                if self.obstacleFlag[itr] == 1:
                    idx = ref_index[itr]
            if size == 1:
                nx, ny = 0, 0
                nProc_x, nProc_y = 1, 1
                N_local = self.N_local
            else:
                nx, ny = mpiParams.nx, mpiParams.ny
                nProc_x, nProc_y = mpiParams.nProc_x, mpiParams.nProc_y
                N_local = mpiParams.N_local
            # print(idx)
            if self.phase is False:
                args = (fields.f, fields.f_new, fields.solid,
                        fields.procBoundary, self.surfaceNodes[itr],
                        self.surfaceInvList[itr], self.surfaceOutList[itr],
                        lattice.c, lattice.invList, self.obstacleFlag[itr],
                        mesh.Nx, mesh.Ny, tempForces, tempTorque,
                        lattice.noOfDirections, self.computeForces,
                        self.computeTorque, idx, N_local,
                        nx, ny, nProc_x, nProc_y, size)
                self.forceTorqueFunc(*args)
            else:
                # args = (fields.f, fields.f_new, fields.solid, fields.phi,
                #         fields.rho, fields.normalPhi, fields.procBoundary,
                #         self.phiInterpolated[itr], transport.sigma,
                #         phaseField.interfaceWidth, self.surfaceNodes[itr],
                #         self.solidNbNodes[itr], self.surfaceNormals[itr],
                #         self.surfaceInvList[itr], self.surfaceOutList[itr],
                #         lattice.c, lattice.invList, self.obstacleFlag[itr],
                #         mesh.Nx, mesh.Ny, tempForces, tempTorque,
                #         tempCapForces, tempCapTorque,
                #         lattice.noOfDirections, self.computeForces,
                #         self.computeTorque, idx, N_local,
                #         nx, ny, nProc_x, nProc_y, size, fields.stressTensor,
                #         fields.p, lattice.cs, fields.curvature, fields.gradPhi,
                #         fields.forceField, lattice.w)
                args = (fields.f, fields.f_new, fields.solid, fields.phi,
                        fields.rho, fields.normalPhi, fields.procBoundary,
                        transport.sigma, phaseField.interfaceWidth,
                        self.surfaceNodes[itr], self.surfaceNormalsFluid[itr],
                        self.surfaceInvList[itr], self.surfaceOutList[itr],
                        lattice.c, lattice.invList, self.obstacleFlag[itr],
                        mesh.Nx, mesh.Ny, tempForces, tempTorque,
                        lattice.noOfDirections, self.computeForces,
                        self.computeTorque, idx, N_local,
                        nx, ny, nProc_x, nProc_y, size, fields.stressTensor,
                        fields.p, lattice.cs, fields.curvature, fields.gradPhi,
                        fields.forceField, fields.solidNbNodesWhole, lattice.w)
                normal, phiNormal, normalSolid, delta, pressForce,\
                    viscForce, surfForce, phiNodes =\
                    self.forceTorqueFuncPhase(*args)
                if itr == 0:
                    profile = tempForces
                    # temp = np.array(np.sum(tempCapForces, axis=0))
                    # np.savetxt('output/' + str(timeStep) + '/forces.dat', temp)
                    # np.savez('output/' + str(timeStep) + '/nodes.npz',
                    #          solidNodes=self.solidNbNodes[itr])
                    # np.savez('output/' + str(timeStep) + '/profile.npz',
                    #          force=profile)
                    # np.savez('output/' + str(timeStep) + '/normal.npz',
                    #          normal=normal)
                    # np.savez('output/' + str(timeStep) + '/normalSolid.npz',
                    #          normalSolid=normalSolid)
                    # np.savez('output/' + str(timeStep) + '/phiNormal.npz',
                    #          normalPhi=phiNormal)
                    # np.savez('output/' + str(timeStep) + '/delta.npz',
                    #          delta=delta)
                    # np.savez('output/' + str(timeStep) + '/pressForce.npz',
                    #          pressForce=pressForce)
                    # np.savez('output/' + str(timeStep) + '/viscForce.npz',
                    #          viscForce=viscForce)
                    # np.savez('output/' + str(timeStep) + '/surfForce.npz',
                    #          surfForce=surfForce)
            self.forces.append(np.sum(tempForces, axis=0))
            self.torque.append(np.sum(tempTorque, axis=0))

    def details(self, rank):
        # print('\n\n\n')
        print(rank, self.surfaceNames)
        # print(rank, self.surfaceNamesGlobal)
        print(rank, self.surfaceNodes)
        print(rank, self.surfaceNormals)
        print(rank, self.solidNbNodes)
        # print(rank, self.obstacleFlag)
        # print(rank, self.x_ref_idx)
        pass

    def setupForcesParallel_cpu(self, parallel):
        if self.phase is True:
            # pass
            self.forceTorqueFuncPhase = numba.njit(self.forceTorqueFuncPhase,
                                                   parallel=parallel,
                                                   cache=False,
                                                   nogil=True)
        else:
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

    def forceTorqueCalc_cuda(self, device, precision, n_threads, blocks,
                             blockSize, ref_index=None):
        self.forces = []
        self.torque = []
        cudaReduce.arrayShape = 3
        for itr in range(self.noOfSurfaces):
            ref_index_device = self.x_ref_idx_device
            if ref_index is not None and self.obstacleFlag[itr] == 1:
                ref_index_device = ref_index[itr]
            tempForcesTorque = np.zeros((self.surfaceNodes[itr].shape[0], 3),
                                        dtype=precision)
            tempForcesTorque_device = cuda.to_device(tempForcesTorque)
            args = (device.f, device.f_new, device.solid,
                    self.surfaceNodes_device[itr],
                    self.surfaceInvList_device[itr],
                    self.surfaceOutList_device[itr], device.c, device.invList,
                    self.obstacleFlag_device[itr], device.Nx[0], device.Ny[0],
                    tempForcesTorque_device, device.noOfDirections[0],
                    self.computeForces_device[0], self.computeTorque_device[0],
                    ref_index_device)
            forceTorque_cuda[blocks, n_threads](*args)
            cudaReduce.cudaSum(blocks, n_threads, blockSize,
                               tempForcesTorque_device)
            self.forces.append(np.array([tempForcesTorque_device[0, 0],
                                        tempForcesTorque_device[0, 1]]))
            self.torque.append(tempForcesTorque_device[0, 2])

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
                                str(round(forces[itr, 0], 10)).ljust(12) +
                                '\t' + str(round(forces[itr, 1], 10)).
                                ljust(12) + '\n')
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


def forceTorque(f, f_new, solid, procBoundary, surfaceNodes, surfaceInvList,
                surfaceOutList, c, invList, obstacleFlag, Nx, Ny, forces,
                torque, noOfDirections, computeForces, computeTorque,
                x_ref, N_local, nx, ny, nProc_x, nProc_y, size):
    for itr in prange(surfaceNodes.shape[0]):
        ind = surfaceNodes[itr]
        if procBoundary[ind] != 1:
            if obstacleFlag == 0:
                for k in range(surfaceOutList.shape[0]):
                    value_0 = ((f[ind, surfaceOutList[k]] *
                               c[surfaceOutList[k], 0]) -
                               (f_new[ind, surfaceInvList[k]]
                               * c[surfaceInvList[k], 0]))
                    value_1 = ((f[ind, surfaceOutList[k]] *
                               c[surfaceOutList[k], 1]) -
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
                        value_0 = ((f[ind, k] * c[k, 0]) -
                                   (f_new[ind, invList[k]]
                                   * c[invList[k], 0]))
                        value_1 = ((f[ind, k] * c[k, 1]) -
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


# def forceTorquePhase(f, f_new, solid, phi, rho, normalPhi, procBoundary,
#                      phiInterpolated, sigma, interfaceWidth, surfaceNodes,
#                      solidNodes, surfaceNormals, surfaceInvList,
#                      surfaceOutList, c, invList, obstacleFlag,
#                      Nx, Ny, forces, torque, capForces, capTorque,
#                      noOfDirections, computeForces, computeTorque, x_ref,
#                      N_local, nx, ny, nProc_x, nProc_y, size, stressTensor,
#                      p, cs, curvature, gradPhi, forceField, w):
#     normal = np.zeros((solidNodes.shape[0], 2), dtype=np.float64)
#     phiNormal = np.zeros((solidNodes.shape[0], 2), dtype=np.float64)
#     normalSolid = np.zeros((solidNodes.shape[0], 2), dtype=np.float64)
#     delta = np.zeros(surfaceNodes.shape[0], dtype=np.float64)
#     pressForce = np.zeros((surfaceNodes.shape[0], 2), dtype=np.float64)
#     viscForce = np.zeros((surfaceNodes.shape[0], 2), dtype=np.float64)
#     surfForce = np.zeros((surfaceNodes.shape[0], 2), dtype=np.float64)
#     phiNodes = np.zeros(solidNodes.shape[0], dtype=np.float64)
#     for itr in prange(surfaceNodes.shape[0]):
#         ind = surfaceNodes[itr]
#         if procBoundary[ind] != 1:
#             if obstacleFlag == 0:
#                 for k in range(surfaceOutList.shape[0]):
#                     value_0 = ((f[ind, surfaceOutList[k]] *
#                                c[surfaceOutList[k], 0]) -
#                                (f_new[ind, surfaceInvList[k]]
#                                * c[surfaceInvList[k], 0]))
#                     value_1 = ((f[ind, surfaceOutList[k]] *
#                                c[surfaceOutList[k], 1]) -
#                                (f_new[ind, surfaceInvList[k]]
#                                * c[surfaceInvList[k], 1]))
#                     if computeForces is True:
#                         forces[itr, 0] += value_0 * rho[ind]
#                         forces[itr, 1] += value_1 * rho[ind]
#                     if computeTorque is True:
#                         i_local, j_local = int(ind / Ny), int(ind % Ny)
#                         if size == 1:
#                             i, j = i_local, j_local
#                         else:
#                             Nx_local, Ny_local = Nx - 2, Ny - 2
#                             if nx == nProc_x - 1:
#                                 Nx_local = N_local[nx - 1, ny, 0]
#                             if ny == nProc_y - 1:
#                                 Ny_local = N_local[nx, ny - 1, 1]
#                             i = nx * Nx_local + i_local - 1
#                             j = ny * Ny_local + j_local - 1
#                         r_0 = i - x_ref[0]
#                         r_1 = j - x_ref[1]
#                         torque[itr] += (r_0 * value_1 - r_1 * value_0)
#             elif obstacleFlag == 1:
#                 i, j = int(ind / Ny), int(ind % Ny)
#                 for k in range(noOfDirections):
#                     i_nb = int(i + c[k, 0] + Nx) % Nx
#                     j_nb = int(j + c[k, 1] + Ny) % Ny
#                     ind_nb = i_nb * Ny + j_nb
#                     if solid[ind_nb, 0] == 1:
#                         value_0 = ((f[ind, k] * c[k, 0]) -
#                                    (f_new[ind, invList[k]]
#                                    * c[invList[k], 0]))
#                         value_1 = ((f[ind, k] * c[k, 1]) -
#                                    (f_new[ind, invList[k]]
#                                    * c[invList[k], 1]))
#                         if computeForces is True:
#                             forces[itr, 0] += value_0 * rho[ind]
#                             forces[itr, 1] += value_1 * rho[ind]
#                         if computeTorque is True:
#                             if size > 1:
#                                 i_local, j_local = i, j
#                                 Nx_local, Ny_local = Nx - 2, Ny - 2
#                                 if nx == nProc_x - 1:
#                                     Nx_local = N_local[nx - 1, ny, 0]
#                                 if ny == nProc_y - 1:
#                                     Ny_local = N_local[nx, ny - 1, 1]
#                                 i_global = nx * Nx_local + i_local - 1
#                                 j_global = ny * Ny_local + j_local - 1
#                             else:
#                                 i_global, j_global = i, j
#                             r_0 = i_global - x_ref[0]
#                             r_1 = j_global - x_ref[1]
#                             torque[itr] += (r_0 * value_1 - r_1 * value_0)
#     for itr in prange(solidNodes.shape[0]):
#         ind = solidNodes[itr]
#         if procBoundary[ind] != 1:
#             i, j = int(ind / Ny), int(ind % Ny)
#             normalSum_x, normalSum_y = 0., 0.
#             denominator = 0.
#             for k in range(noOfDirections):
#                 i_nb = i + int(c[k, 0])
#                 j_nb = j + int(c[k, 1])
#                 if size == 1:
#                     i_nb = (i_nb + Nx) % Nx
#                     j_nb = (j_nb + Ny) % Ny
#                 ind_nb = i_nb * Ny + j_nb
#                 normalSum_x += w[k] * normalPhi[ind_nb, 0] *\
#                     (1 - solid[ind_nb, 0])
#                 normalSum_y += w[k] * normalPhi[ind_nb, 1] *\
#                     (1 - solid[ind_nb, 0])
#                 denominator += w[k] * (1 - solid[ind_nb, 0])
#             normalInterpolated_x = normalSum_x / (denominator + 1e-17)
#             normalInterpolated_y = normalSum_y / (denominator + 1e-17)
#             normalDotTangent = \
#                 np.abs(-normalInterpolated_x * surfaceNormals[itr, 1] +
#                        normalInterpolated_y * surfaceNormals[itr, 0])
#             if obstacleFlag == 0:
#                 tangent = normalInterpolated_x * surfaceNormals[itr, 1] -\
#                     surfaceNormals[itr, 0] * normalInterpolated_y
#                 normalCap_x = -normalInterpolated_y * tangent
#                 normalCap_y = normalInterpolated_x * tangent
#                 magNormal = np.sqrt(normalCap_x * normalCap_x +
#                                     normalCap_y * normalCap_y)
#                 # capForce_x = 24 * sigma * phi[ind] * phi[ind] * \
#                 #     (phi[ind] - 1) * (phi[ind] - 1) *\
#                 #     (normalCap_x / (magNormal + 1e-17)) /\
#                 #     interfaceWidth
#                 # capForce_y = 24 * sigma * phi[ind] * phi[ind] *\
#                 #     (phi[ind] - 1) * (phi[ind] - 1) *\
#                 #     (normalCap_y / (magNormal + 1e-17)) /\
#                 #     interfaceWidth
#                 normal[itr, 0] = normalCap_x / (magNormal + 1e-17)
#                 normal[itr, 1] = normalCap_y / (magNormal + 1e-17)
#                 phiNormal[itr, 0] = normalInterpolated_x
#                 phiNormal[itr, 1] = normalInterpolated_y
#                 normalSolid[itr, 0] = surfaceNormals[itr, 0]
#                 normalSolid[itr, 1] = surfaceNormals[itr, 1]
#                 capForces[itr, 0] = 4 * sigma * phi[ind] / interfaceWidth *\
#                     (1 - phi[ind]) * (normalCap_x / (magNormal + 1e-17))\
#                     * normalDotTangent
#                 capForces[itr, 1] = 4 * sigma * phi[ind] / interfaceWidth *\
#                     (1 - phi[ind]) * (normalCap_y / (magNormal + 1e-17))\
#                     * normalDotTangent
#                 if computeTorque is True:
#                     capTorque[itr] += (r_0 * capForces[itr, 1] -
#                                        r_1 * capForces[itr, 0])
#             elif obstacleFlag == 1:
#                 tangent = normalInterpolated_x * surfaceNormals[itr, 1] -\
#                     surfaceNormals[itr, 0] * normalInterpolated_y
#                 normalCap_x = -normalInterpolated_y * tangent
#                 normalCap_y = normalInterpolated_x * tangent
#                 magNormal = np.sqrt(normalCap_x * normalCap_x +
#                                     normalCap_y * normalCap_y)
#                 # capForce_x = 24 * sigma * phi[ind] * phi[ind] * \
#                 #     (phi[ind] - 1) * (phi[ind] - 1) *\
#                 #     (normalCap_x / (magNormal + 1e-17)) /\
#                 #     interfaceWidth
#                 # capForce_y = 24 * sigma * phi[ind] * phi[ind] *\
#                 #     (phi[ind] - 1) * (phi[ind] - 1) *\
#                 #     (normalCap_y / (magNormal + 1e-17)) /\
#                 #     interfaceWidth
#                 normal[itr, 0] = normalCap_x / (magNormal + 1e-17)
#                 normal[itr, 1] = normalCap_y / (magNormal + 1e-17)
#                 phiNormal[itr, 0] = normalInterpolated_x
#                 phiNormal[itr, 1] = normalInterpolated_y
#                 normalSolid[itr, 0] = surfaceNormals[itr, 0]
#                 normalSolid[itr, 1] = surfaceNormals[itr, 1]
#                 capForces[itr, 0] = 4 * sigma * phi[ind] / interfaceWidth *\
#                     (1 - phi[ind]) * (normalCap_x / (magNormal + 1e-17))\
#                     * normalDotTangent
#                 capForces[itr, 1] = 4 * sigma * phi[ind] / interfaceWidth *\
#                     (1 - phi[ind]) * (normalCap_y / (magNormal + 1e-17))\
#                     * normalDotTangent
#                 if computeTorque is True:
#                     capTorque[itr] += (r_0 * capForces[itr, 1] -
#                                        r_1 * capForces[itr, 0])
#     return normal, phiNormal, normalSolid, delta, pressForce,\
#         viscForce, surfForce, phiNodes


def forceTorquePhase(f, f_new, solid, phi, rho, normalPhi, procBoundary, sigma,
                     interfaceWidth, surfaceNodes, surfaceNormalsFluid,
                     surfaceInvList, surfaceOutList, c, invList, obstacleFlag,
                     Nx, Ny, forces, torque, noOfDirections, computeForces,
                     computeTorque, x_ref, N_local, nx, ny, nProc_x,
                     nProc_y, size, stressTensor, p, cs, curvature, gradPhi,
                     forceField, solidNodesWhole, w):
    normal = np.zeros((surfaceNodes.shape[0], 2), dtype=np.float64)
    phiNormal = np.zeros((surfaceNodes.shape[0], 2), dtype=np.float64)
    normalSolid = np.zeros((surfaceNodes.shape[0], 2), dtype=np.float64)
    delta = np.zeros(surfaceNodes.shape[0], dtype=np.float64)
    pressForce = np.zeros((surfaceNodes.shape[0], 2), dtype=np.float64)
    viscForce = np.zeros((surfaceNodes.shape[0], 2), dtype=np.float64)
    surfForce = np.zeros((surfaceNodes.shape[0], 2), dtype=np.float64)
    phiNodes = np.zeros(surfaceNodes.shape[0], dtype=np.float64)
    for itr in prange(surfaceNodes.shape[0]):
        ind = surfaceNodes[itr]
        if procBoundary[ind] != 1:
            if phi[ind] >= 0.05 and phi[ind] <= 0.95:
                phiNodes[itr] = ind
            # force from stress tensor
            pressForce[itr, 0] = - p[ind] * surfaceNormalsFluid[itr, 0] *\
                rho[ind] * cs * cs
            pressForce[itr, 1] = - p[ind] * surfaceNormalsFluid[itr, 1] *\
                rho[ind] * cs * cs
            viscForce[itr, 0] = \
                (stressTensor[ind, 0, 0] * surfaceNormalsFluid[itr, 0] +
                 stressTensor[ind, 0, 1] * surfaceNormalsFluid[itr, 1]) *\
                rho[ind]
            viscForce[itr, 1] = \
                (stressTensor[ind, 1, 0] * surfaceNormalsFluid[itr, 0] +
                 stressTensor[ind, 1, 1] * surfaceNormalsFluid[itr, 1]) *\
                rho[ind]
            surfForce[itr, 0] = forceField[ind, 0]
            surfForce[itr, 1] = forceField[ind, 1]
            #############################################################
            normalDotTangent = \
                np.abs(-normalPhi[ind, 0] * surfaceNormalsFluid[itr, 1] +
                       normalPhi[ind, 1] * surfaceNormalsFluid[itr, 0])
            # distance_1 = ((normalPhi[ind, 0] + surfaceNormalsFluid[itr, 1]) *
            #               (normalPhi[ind, 0] + surfaceNormalsFluid[itr, 1])) +\
            #              ((normalPhi[ind, 1] - surfaceNormalsFluid[itr, 0]) *
            #               (normalPhi[ind, 1] - surfaceNormalsFluid[itr, 0]))
            # distance_2 = ((normalPhi[ind, 0] - surfaceNormalsFluid[itr, 1]) *
            #               (normalPhi[ind, 0] - surfaceNormalsFluid[itr, 1])) +\
            #              ((normalPhi[ind, 1] + surfaceNormalsFluid[itr, 0]) *
            #               (normalPhi[ind, 1] + surfaceNormalsFluid[itr, 0]))
            # if distance_1 < distance_2:
            #     normalDotTangent = \
            #         (-normalPhi[ind, 0] * surfaceNormalsFluid[itr, 1] +
            #          normalPhi[ind, 1] * surfaceNormalsFluid[itr, 0])
            # elif distance_1 >= distance_2:
            #     normalDotTangent = \
            #         (normalPhi[ind, 0] * surfaceNormalsFluid[itr, 1] -
            #          normalPhi[ind, 1] * surfaceNormalsFluid[itr, 0])
            deltaFunc = 4 * phi[ind] * (1 - phi[ind]) / interfaceWidth
            delta[itr] = deltaFunc * normalDotTangent
            if obstacleFlag == 0:
                for k in range(surfaceOutList.shape[0]):
                    value_0 = ((f[ind, surfaceOutList[k]] *
                               c[surfaceOutList[k], 0]) -
                               (f_new[ind, surfaceInvList[k]]
                               * c[surfaceInvList[k], 0]))
                    value_1 = ((f[ind, surfaceOutList[k]] *
                               c[surfaceOutList[k], 1]) -
                               (f_new[ind, surfaceInvList[k]]
                               * c[surfaceInvList[k], 1]))
                    if computeForces is True:
                        forces[itr, 0] += value_0 * rho[ind]
                        forces[itr, 1] += value_1 * rho[ind]
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
                tangent = normalPhi[ind, 0] * surfaceNormalsFluid[itr, 1] -\
                    surfaceNormalsFluid[itr, 0] * normalPhi[ind, 1]
                normalCap_x = -normalPhi[ind, 1] * tangent
                normalCap_y = normalPhi[ind, 0] * tangent
                magNormal = np.sqrt(normalCap_x * normalCap_x +
                                    normalCap_y * normalCap_y)
                # capForce_x = 24 * sigma * phi[ind] * phi[ind] * \
                #     (phi[ind] - 1) * (phi[ind] - 1) *\
                #     (normalCap_x / (magNormal + 1e-17)) /\
                #     interfaceWidth
                # capForce_y = 24 * sigma * phi[ind] * phi[ind] *\
                #     (phi[ind] - 1) * (phi[ind] - 1) *\
                #     (normalCap_y / (magNormal + 1e-17)) /\
                #     interfaceWidth
                normal[itr, 0] = normalCap_x / (magNormal + 1e-17)
                normal[itr, 1] = normalCap_y / (magNormal + 1e-17)
                phiNormal[itr, 0] = normalPhi[ind, 0]
                phiNormal[itr, 1] = normalPhi[ind, 1]
                normalSolid[itr, 0] = surfaceNormalsFluid[itr, 0]
                normalSolid[itr, 1] = surfaceNormalsFluid[itr, 1]
                delx, dely = 0, 0
                if solidNodesWhole[int((i + 1) * Ny + j)] == 1:
                    delx += 0.5
                if solidNodesWhole[int((i - 1) * Ny + j)] == 1:
                    delx += 0.5
                if solidNodesWhole[int(i * Ny + j + 1)] == 1:
                    dely += 0.5
                if solidNodesWhole[int(i * Ny + j - 1)] == 1:
                    dely += 0.5
                delPhi = np.abs(gradPhi[ind, 0]) * delx +\
                    np.abs(gradPhi[ind, 1]) * dely
                capForce_x = 6 * sigma * phi[ind] * (1 - phi[ind]) *\
                    (normalCap_x / (magNormal + 1e-17)) * delPhi
                capForce_y = 6 * sigma * phi[ind] * (1 - phi[ind]) *\
                    (normalCap_y / (magNormal + 1e-17)) * delPhi
                forces[itr, 0] += capForce_x
                forces[itr, 1] += capForce_y
                if computeTorque is True:
                    torque[itr] += (r_0 * capForce_y - r_1 * capForce_x)
            elif obstacleFlag == 1:
                i, j = int(ind / Ny), int(ind % Ny)
                for k in range(noOfDirections):
                    i_nb = int(i + c[k, 0] + Nx) % Nx
                    j_nb = int(j + c[k, 1] + Ny) % Ny
                    ind_nb = i_nb * Ny + j_nb
                    if solid[ind_nb, 0] == 1:
                        value_0 = ((f[ind, k] * c[k, 0]) -
                                   (f_new[ind, invList[k]]
                                   * c[invList[k], 0]))
                        value_1 = ((f[ind, k] * c[k, 1]) -
                                   (f_new[ind, invList[k]]
                                   * c[invList[k], 1]))
                        if computeForces is True:
                            forces[itr, 0] += value_0 * rho[ind]
                            forces[itr, 1] += value_1 * rho[ind]
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
                # forces[itr, 0] -= 0.5 * forceField[ind, 0]
                # forces[itr, 1] -= 0.5 * forceField[ind, 1]
                tangent = normalPhi[ind, 0] * surfaceNormalsFluid[itr, 1] -\
                    surfaceNormalsFluid[itr, 0] * normalPhi[ind, 1]
                normalCap_x = -normalPhi[ind, 1] * tangent
                normalCap_y = normalPhi[ind, 0] * tangent
                magNormal = np.sqrt(normalCap_x * normalCap_x +
                                    normalCap_y * normalCap_y)
                # capForce_x = 24 * sigma * phi[ind] * phi[ind] * \
                #     (phi[ind] - 1) * (phi[ind] - 1) *\
                #     (normalCap_x / (magNormal + 1e-17)) /\
                #     interfaceWidth
                # capForce_y = 24 * sigma * phi[ind] * phi[ind] *\
                #     (phi[ind] - 1) * (phi[ind] - 1) *\
                #     (normalCap_y / (magNormal + 1e-17)) /\
                #     interfaceWidth
                normal[itr, 0] = normalCap_x / (magNormal + 1e-17)
                normal[itr, 1] = normalCap_y / (magNormal + 1e-17)
                phiNormal[itr, 0] = normalPhi[ind, 0]
                phiNormal[itr, 1] = normalPhi[ind, 1]
                normalSolid[itr, 0] = surfaceNormalsFluid[itr, 0]
                normalSolid[itr, 1] = surfaceNormalsFluid[itr, 1]
                delx, dely = 0, 0
                if solidNodesWhole[int((i + 1) * Ny + j)] == 1:
                    delx += 0.5
                if solidNodesWhole[int((i - 1) * Ny + j)] == 1:
                    delx += 0.5
                if solidNodesWhole[int(i * Ny + j + 1)] == 1:
                    dely += 0.5
                if solidNodesWhole[int(i * Ny + j - 1)] == 1:
                    dely += 0.5
                delPhi = np.abs(gradPhi[ind, 0]) * delx +\
                    np.abs(gradPhi[ind, 1]) * dely
                capForce_x = 6 * sigma * phi[ind] * (1 - phi[ind]) * \
                    (normalCap_x / (magNormal + 1e-17)) * delPhi
                capForce_y = 6 * sigma * phi[ind] * (1 - phi[ind]) *\
                    (normalCap_y / (magNormal + 1e-17)) * delPhi
                forces[itr, 0] += capForce_x
                forces[itr, 1] += capForce_y
                if computeTorque is True:
                    torque[itr] += (r_0 * capForce_y - r_1 * capForce_x)
    return normal, phiNormal, normalSolid, delta, pressForce,\
        viscForce, surfForce, phiNodes


@cuda.jit
def forceTorque_cuda(f, f_new, solid, surfaceNodes, surfaceInvList,
                     surfaceOutList, c, invList, obstacleFlag, Nx, Ny,
                     forceTorque, noOfDirections, computeForces, computeTorque,
                     x_ref):
    ind = cuda.grid(1)
    if ind < Nx * Ny:
        for itr in range(surfaceNodes.shape[0]):
            if ind == surfaceNodes[itr]:
                if obstacleFlag == 0:
                    for k in range(surfaceOutList.shape[0]):
                        value_0 = ((f[ind, surfaceOutList[k]] *
                                   c[surfaceOutList[k], 0]) -
                                   (f_new[ind, surfaceInvList[k]]
                                   * c[surfaceInvList[k], 0]))
                        value_1 = ((f[ind, surfaceOutList[k]] *
                                   c[surfaceOutList[k], 1]) -
                                   (f_new[ind, surfaceInvList[k]]
                                   * c[surfaceInvList[k], 1]))
                        if computeForces is True:
                            forceTorque[itr, 0] += value_0
                            forceTorque[itr, 1] += value_1
                        if computeTorque is True:
                            i, j = np.int64(ind / Ny), np.int64(ind % Ny)
                            r_0 = i - x_ref[0]
                            r_1 = j - x_ref[1]
                            forceTorque[itr, 2] += (r_0 * value_1 -
                                                    r_1 * value_0)
                elif obstacleFlag == 1:
                    i, j = int(ind / Ny), int(ind % Ny)
                    for k in range(noOfDirections):
                        i_nb = int(i + c[k, 0] + Nx) % Nx
                        j_nb = int(j + c[k, 1] + Ny) % Ny
                        ind_nb = i_nb * Ny + j_nb
                        if solid[ind_nb, 0] == 1:
                            value_0 = ((f[ind, k] * c[k, 0]) -
                                       (f_new[ind, invList[k]]
                                       * c[invList[k], 0]))
                            value_1 = ((f[ind, k] * c[k, 1]) -
                                       (f_new[ind, invList[k]]
                                       * c[invList[k], 1]))
                            if computeForces is True:
                                forceTorque[itr, 0] += value_0
                                forceTorque[itr, 1] += value_1
                            if computeTorque is True:
                                r_0 = i - x_ref[0]
                                r_1 = j - x_ref[1]
                                forceTorque[itr, 2] += (r_0 * value_1 -
                                                        r_1 * value_0)
            else:
                continue
    else:
        return
