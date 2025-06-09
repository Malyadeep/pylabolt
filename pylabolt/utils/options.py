import numpy as np
import numba
import os
import sys
from numba import (prange, cuda)

import pylabolt.parallel.cudaReduce as cudaReduce
from pylabolt.base.obstacle import findNb, reInitSolidFluidBoundaries


def initializeSolidBoundaryNodes(Nx, Ny, solidNbNodesWhole):
    for ind in prange(Nx * Ny):
        solidNbNodesWhole[ind] = 0


def solidNormalCircle(surfaceNodes, solidNodes, solidNbNodesWhole, normal,
                      normalFluid, solid, procBoundary, boundaryNode, center,
                      radius, Nx, Ny, nx, ny, nProc_x, nProc_y, N_local, size):
    for itr in prange(solidNodes.shape[0]):
        ind = solidNodes[itr]
        if (solid[ind, 0] != 0 and procBoundary[ind] != 1 and
                boundaryNode[ind] != 1):
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
            normalSolid_x_solidSide = i_global - center[0]
            normalSolid_y_solidSide = j_global - center[1]
            magSolid = \
                np.sqrt(normalSolid_x_solidSide * normalSolid_x_solidSide +
                        normalSolid_y_solidSide * normalSolid_y_solidSide +
                        1e-19)
            normal[itr, 0] = normalSolid_x_solidSide / magSolid
            normal[itr, 1] = normalSolid_y_solidSide / magSolid
    for itr in prange(surfaceNodes.shape[0]):
        ind = surfaceNodes[itr]
        solidNbNodesWhole[ind] = 1
        if (solid[ind, 0] != 1 and procBoundary[ind] != 1 and
                boundaryNode[ind] != 1):
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
            normalSolid_x_fluidSide = i_global - center[0]
            normalSolid_y_fluidSide = j_global - center[1]
            magSolid = \
                np.sqrt(normalSolid_x_fluidSide * normalSolid_x_fluidSide +
                        normalSolid_y_fluidSide * normalSolid_y_fluidSide +
                        1e-19)
            normalFluid[itr, 0] = normalSolid_x_fluidSide / magSolid
            normalFluid[itr, 1] = normalSolid_y_fluidSide / magSolid


def solidNormal(surfaceNodes, solidNodes, solidNbNodesWhole, normal,
                normalFluid, solid, procBoundary, boundaryNode, cs_2,
                c, w, noOfDirections, Nx, Ny, nx, ny, nProc_x, nProc_y,
                size):
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
                    if (i + int(2 * c[k, 0]) == 0 and nx == 0 or
                            i + int(2 * c[k, 0]) == Nx - 1 and
                            nx == nProc_x - 1):
                        i_nb = i_nb + int(c[k, 0])
                    if (j + int(2 * c[k, 1]) == 0 and ny == 0 or
                            j + int(2 * c[k, 1]) == Ny - 1 and
                            ny == nProc_y - 1):
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
        elif (solid[ind, 0] != 0 and procBoundary[ind] != 1 and
                boundaryNode[ind] == 1):
            i, j = int(ind / Ny), int(ind % Ny)
            gradSolidSum_x, gradSolidSum_y = 0., 0.
            for k in range(noOfDirections):
                i_nb = i + int(c[k, 0])
                j_nb = j + int(c[k, 1])
                ind_nb = int(i_nb * Ny + j_nb)
                if (i_nb >= Nx or j_nb >= Ny or i_nb < 0 or j_nb < 0 or
                        procBoundary[ind_nb] == 1):
                    gradSolidSum_x += c[k, 0] * w[k] * 1
                    gradSolidSum_y += c[k, 1] * w[k] * 1
                else:
                    gradSolidSum_x += c[k, 0] * w[k] * solid[ind_nb, 0]
                    gradSolidSum_y += c[k, 1] * w[k] * solid[ind_nb, 0]
            gradSolid_x = cs_2 * gradSolidSum_x
            gradSolid_y = cs_2 * gradSolidSum_y
            magGradSolid = np.sqrt(gradSolid_x * gradSolid_x +
                                   gradSolid_y * gradSolid_y)
            normal[itr, 0] = - gradSolid_x / (magGradSolid + 1e-17)
            normal[itr, 1] = - gradSolid_y / (magGradSolid + 1e-17)
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
                    if (i + int(2 * c[k, 0]) == 0 and nx == 0 or
                            i + int(2 * c[k, 0]) == Nx - 1 and
                            nx == nProc_x - 1):
                        i_nb = i_nb + int(c[k, 0])
                    if (j + int(2 * c[k, 1]) == 0 and ny == 0 or
                            j + int(2 * c[k, 1]) == Ny - 1 and
                            ny == nProc_y - 1):
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


# def solidNormalBoundary(surfaceNodes, solidNodes, solidNbNodesWhole,
#                         normalBoundary, normal, normalFluid, solid,
#                         procBoundary, boundaryNode, Nx, Ny):
def solidNormalBoundary(surfaceNodes, solidNodes, solidNbNodesWhole,
                        normalBoundary, normal, normalFluid, solid,
                        procBoundary, boundaryNode, cs_2, c, w, noOfDirections,
                        Nx, Ny, nx, ny, nProc_x, nProc_y, size):
    for itr in prange(solidNodes.shape[0]):
        ind = solidNodes[itr]
        # It is a boundary node
        if (solid[ind, 0] != 0 and procBoundary[ind] != 1 and
                boundaryNode[ind] == 1):
            i, j = int(ind / Ny), int(ind % Ny)
            cornerCond = (i == 0 and j == 0) or\
                (i == 0 and j == Ny - 1) or\
                (i == Nx - 1 and j == 0) or\
                (i == Nx - 1 and j == Ny - 1)
            if cornerCond is False:
                normal[itr, 0] = normalBoundary[0]
                normal[itr, 1] = normalBoundary[1]
            else:
                gradSolidSum_x, gradSolidSum_y = 0., 0.
                denominator = 0.
                for k in range(noOfDirections):
                    i_nb = i + int(c[k, 0])
                    j_nb = j + int(c[k, 1])
                    if size == 1:
                        if (i_nb < 0 or i_nb == Nx
                                or j_nb < 0 or j_nb == Ny):
                            pass
                        else:
                            ind_nb = int(i_nb * Ny + j_nb)
                            gradSolidSum_x += c[k, 0] * w[k] * solid[ind_nb, 0]
                            gradSolidSum_y += c[k, 1] * w[k] * solid[ind_nb, 0]
                            denominator += w[k]
                    elif size > 1 and boundaryNode[ind] == 2:
                        if (i + int(2 * c[k, 0]) == 0 and nx == 0 or
                                i + int(2 * c[k, 0]) == Nx - 1 and
                                nx == nProc_x - 1):
                            i_nb = i_nb + int(c[k, 0])
                        if (j + int(2 * c[k, 1]) == 0 and ny == 0 or
                                j + int(2 * c[k, 1]) == Ny - 1 and
                                ny == nProc_y - 1):
                            j_nb = j_nb + int(c[k, 1])
                gradSolid_x = cs_2 * gradSolidSum_x / (denominator + 1e-20)
                gradSolid_y = cs_2 * gradSolidSum_y / (denominator + 1e-20)
                magGradSolid = np.sqrt(gradSolid_x * gradSolid_x +
                                       gradSolid_y * gradSolid_y)
                normal[itr, 0] = gradSolid_x / (magGradSolid + 1e-17)
                normal[itr, 1] = gradSolid_y / (magGradSolid + 1e-17)
    for itr in prange(surfaceNodes.shape[0]):
        ind = surfaceNodes[itr]
        solidNbNodesWhole[ind] = 1
        # It is not a boundary node
        if (solid[ind, 0] != 1 and procBoundary[ind] != 1 and
                boundaryNode[ind] != 1):
            # normalFluid[itr, 0] = normalBoundary[0]
            # normalFluid[itr, 1] = normalBoundary[1]
            i, j = int(ind / Ny), int(ind % Ny)
            gradSolidSum_x, gradSolidSum_y = 0., 0.
            denominator = 0.
            for k in range(noOfDirections):
                i_nb = i + int(c[k, 0])
                j_nb = j + int(c[k, 1])
                ind_nb = int(i_nb * Ny + j_nb)
                if (size == 1 and boundaryNode[ind_nb] == 1):
                    gradSolidSum_x += c[k, 0] * w[k] * solid[ind_nb, 0]
                    gradSolidSum_y += c[k, 1] * w[k] * solid[ind_nb, 0]
                    denominator += w[k]
                elif size > 1 and boundaryNode[ind] == 2:
                    if (i + int(2 * c[k, 0]) == 0 and nx == 0 or
                            i + int(2 * c[k, 0]) == Nx - 1 and
                            nx == nProc_x - 1):
                        i_nb = i_nb + int(c[k, 0])
                    if (j + int(2 * c[k, 1]) == 0 and ny == 0 or
                            j + int(2 * c[k, 1]) == Ny - 1 and
                            ny == nProc_y - 1):
                        j_nb = j_nb + int(c[k, 1])
            gradSolid_x = cs_2 * gradSolidSum_x / (denominator + 1e-20)
            gradSolid_y = cs_2 * gradSolidSum_y / (denominator + 1e-20)
            magGradSolid = np.sqrt(gradSolid_x * gradSolid_x +
                                   gradSolid_y * gradSolid_y)
            normalFluid[itr, 0] = - gradSolid_x / (magGradSolid + 1e-17)
            normalFluid[itr, 1] = - gradSolid_y / (magGradSolid + 1e-17)


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
                self.scheme = options["scheme"]
            except KeyError:
                self.scheme = 1
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
        else:
            self.phase = False
        self.surfaceNodes = []
        self.surfaceNormals = []
        self.surfaceNormalsFluid = []
        self.solidNbNodes = []
        self.surfaceNames = []
        self.surfaceInvList = []        # Only at boundaries
        self.surfaceOutList = []        # Only at boundaries
        self.noOfSurfaces = 0
        self.obstacleFlag = []          # differentiate obstacle and boundary
        self.boundaryType = []
        self.wallBoundaryNo = []
        self.forces = []
        self.torque = []
        self.forceReconstruction = []
        self.torqueReconstruction = []
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
            self.wallBoundaryNo.append(None)
            if self.phase is True:
                self.boundaryType.append('none')
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
                    solidNbNodes = \
                        np.zeros(boundary.nbList[itr].shape[0] +
                                 boundary.cornerList[itr].shape[0],
                                 dtype=np.int64)
                    solidNbNodes[0] = boundary.cornerList[itr][0]
                    solidNbNodes[-1] = boundary.cornerList[itr][1]
                    solidNbNodes[1:-1] = np.copy(boundary.nbList[itr])
                    self.solidNbNodes.\
                        append(solidNbNodes)
                    self.surfaceInvList.append(boundary.invDirections[itr])
                    self.surfaceOutList.append(boundary.outDirections[itr])
                    self.noOfSurfaces += 1
                    self.obstacleFlag.append(0)
                    self.wallBoundaryNo.append(itr)
                    if self.phase is True:
                        self.boundaryType.\
                            append(boundary.boundaryTypePhase[itr])
                itr += 1

    def initializeForceReconstruction(self, precision):
        for itr in range(self.noOfSurfaces):
            self.forceReconstruction.append(np.zeros(2, dtype=precision))
            self.torqueReconstruction.append(precision(0))

    def computeSolidNormals(self, fields, lattice, mesh, size, precision,
                            obstacle, boundary, mpiParams=None):
        if not os.path.isdir("output"):
            os.makedirs("output")
        if size == 1:
            nx, ny = 0, 0
            nProc_x, nProc_y = 1, 1
            N_local = self.N_local
        else:
            nx, ny = mpiParams.nx, mpiParams.ny
            nProc_x, nProc_y = mpiParams.nProc_x, mpiParams.nProc_y
            N_local = mpiParams.N_local
        for itr in range(self.noOfSurfaces):
            tempNormal = np.zeros((self.solidNbNodes[itr].shape[0], 2),
                                  dtype=precision)
            tempNormalFluid = np.zeros((self.surfaceNodes[itr].shape[0],
                                       2), dtype=precision)
            if self.obstacleFlag[itr] == 1:
                if obstacle.obsType[itr] == "circle":
                    # if obstacle.obsType[itr] == "none":
                    solidNormalCircle(self.surfaceNodes[itr],
                                      self.solidNbNodes[itr],
                                      fields.solidNbNodesWhole,
                                      tempNormal, tempNormalFluid,
                                      fields.solid, fields.
                                      procBoundary, fields.
                                      boundaryNode, obstacle.
                                      obsOrigin[itr], obstacle.
                                      radius[itr], mesh.Nx, mesh.Ny,
                                      nx, ny, nProc_x, nProc_y,
                                      N_local, size)
                else:
                    solidNormal(self.surfaceNodes[itr],
                                self.solidNbNodes[itr],
                                fields.solidNbNodesWhole, tempNormal,
                                tempNormalFluid, fields.solid, fields.
                                procBoundary, fields.boundaryNode,
                                lattice.cs_2, lattice.c, lattice.w,
                                lattice.noOfDirections, mesh.Nx,
                                mesh.Ny, nx, ny, nProc_x, nProc_y,
                                size)
            else:
                boundaryNo = self.wallBoundaryNo[itr]
                # solidNormalBoundary(self.surfaceNodes[itr],
                #                     self.solidNbNodes[itr],
                #                     fields.solidNbNodesWhole,
                #                     boundary.normalBoundary[boundaryNo],
                #                     tempNormal, tempNormalFluid, fields.solid,
                #                     fields.procBoundary, fields.boundaryNode,
                #                     mesh.Nx, mesh.Ny)
                solidNormalBoundary(self.surfaceNodes[itr],
                                    self.solidNbNodes[itr],
                                    fields.solidNbNodesWhole,
                                    boundary.normalBoundary[boundaryNo],
                                    tempNormal, tempNormalFluid, fields.solid,
                                    fields.procBoundary, fields.boundaryNode,
                                    lattice.cs_2, lattice.c, lattice.w,
                                    lattice.noOfDirections, mesh.Nx,
                                    mesh.Ny, nx, ny, nProc_x, nProc_y,
                                    size)
            self.surfaceNormals.append(tempNormal)
            self.surfaceNormalsFluid.append(tempNormalFluid)
            # np.savez("output/normals_surface_0_" + str(itr) + ".npz",
            #          solid=fields.solid, fluidNbNodes=self.surfaceNodes[itr],
            #          solidNbNodes=self.solidNbNodes[itr],
            #          fluidSideNormals=self.surfaceNormalsFluid[itr],
            #          solidSideNormals=self.surfaceNormals[itr])

    def computeMovingSolidBoundary(self, mesh, lattice, fields, size,
                                   precision, obstacle, timeStep, saveInterval,
                                   mpiParams=None):
        if size == 1:
            nx, ny = 0, 0
            nProc_x, nProc_y = 1, 1
            N_local = self.N_local
        else:
            nx, ny = mpiParams.nx, mpiParams.ny
            nProc_x, nProc_y = mpiParams.nProc_x, mpiParams.nProc_y
            N_local = mpiParams.N_local
        if self.phase is True:
            self.initializeSolidBoundaryNodes(mesh.Nx, mesh.Ny,
                                              fields.solidNbNodesWhole)
        obstacle.reInitSolidFluidBoundaries(mesh.Nx, mesh.Ny,
                                            fields.fluidBoundary,
                                            fields.solidBoundary,
                                            fields.fluidBoundary_old,
                                            fields.solidBoundary_old)
        for itr in range(self.noOfSurfaces):
            if self.obstacleFlag[itr] == 1:
                args = (fields.solid, fields.fluidBoundary,
                        fields.solidBoundary, fields.boundaryNode,
                        fields.procBoundary, mesh.Nx, mesh.Ny,
                        lattice.c, lattice.noOfDirections,
                        itr, nx, ny, nProc_x, nProc_y)
                fluidNbNodes, solidNbNodes = findNb(*args, size=size)
                self.surfaceNodes[itr] = fluidNbNodes
                self.solidNbNodes[itr] = solidNbNodes
                if self.phase is True:
                    tempNormal = np.zeros((self.solidNbNodes[itr].shape[0], 2),
                                          dtype=precision)
                    tempNormalFluid = \
                        np.zeros((self.surfaceNodes[itr].shape[0], 2),
                                 dtype=precision)
                    if obstacle.obsType[itr] == "circle":
                        # if obstacle.obsType[itr] == "none":
                        self.solidNormalCircle(fluidNbNodes, solidNbNodes,
                                               fields.solidNbNodesWhole,
                                               tempNormal, tempNormalFluid,
                                               fields.solid, fields.
                                               procBoundary, fields.
                                               boundaryNode, obstacle.
                                               obsOrigin[itr], obstacle.
                                               radius[itr], mesh.Nx, mesh.Ny,
                                               nx, ny, nProc_x, nProc_y,
                                               N_local, size)
                    else:
                        self.solidNormal(fluidNbNodes, solidNbNodes,
                                         fields.solidNbNodesWhole, tempNormal,
                                         tempNormalFluid, fields.solid, fields.
                                         procBoundary, fields.boundaryNode,
                                         lattice.cs_2, lattice.c, lattice.w,
                                         lattice.noOfDirections, mesh.Nx,
                                         mesh.Ny, nx, ny, nProc_x, nProc_y,
                                         size)
                    self.surfaceNormals[itr] = tempNormal
                    self.surfaceNormalsFluid[itr] = tempNormalFluid
            # if timeStep % saveInterval == 0:
            #     np.savez("output/normals_surface_" + str(timeStep) +
            #              "_" + str(itr) + ".npz",
            #              solid=fields.solid,
            #              fluidNbNodes=self.surfaceNodes[itr],
            #              solidNbNodes=self.solidNbNodes[itr],
            #              fluidSideNormals=self.surfaceNormalsFluid[itr],
            #              solidSideNormals=self.surfaceNormals[itr])

    def forceTorqueCalc(self, fields, transport, lattice, mesh, precision,
                        obstacle, size, timeStep, mpiParams=None,
                        ref_index=None, phaseField=None, warmup=False):
        self.forces = []
        self.torque = []
        self.capForces = []
        self.hydForces = []
        self.capTorque = []
        self.hydTorque = []
        fields.deltaFuncFluid = np.zeros((mesh.Nx * mesh.Ny),
                                         dtype=precision)
        fields.delPhiFluid = np.zeros((mesh.Nx * mesh.Ny),
                                      dtype=precision)
        fields.contactAngleLocalFluid = np.zeros((mesh.Nx * mesh.Ny),
                                                 dtype=precision)
        # print(ref_index, obstacle.modifyObsFunc)
        for itr in range(self.noOfSurfaces):
            tempForces = np.zeros((self.surfaceNodes[itr].shape[0], 2),
                                  dtype=precision)
            tempTorque = np.zeros((self.surfaceNodes[itr].shape[0]),
                                  dtype=precision)
            tempForcesCap = np.zeros((self.surfaceNodes[itr].shape[0], 2),
                                     dtype=precision)
            tempForcesHyd = np.zeros((self.surfaceNodes[itr].shape[0], 2),
                                     dtype=precision)
            tempTorqueCap = np.zeros((self.surfaceNodes[itr].shape[0]),
                                     dtype=precision)
            tempTorqueHyd = np.zeros((self.surfaceNodes[itr].shape[0]),
                                     dtype=precision)
            positionVector = np.zeros_like(tempForcesCap)
            leastDistance = np.zeros_like(tempTorque)
            interfaceTangent = np.zeros_like(tempForcesCap)
            delta = np.zeros_like(tempTorqueCap)
            contactAngleLocal = np.zeros_like(tempTorqueCap)
            cosThetaLocal = np.zeros_like(tempTorqueCap)
            delPhi = np.zeros_like(tempTorqueCap)
            idx = self.x_ref_idx
            if (ref_index is not None and len(ref_index) > 0 and
                    self.obstacleFlag[itr] == 1):
                if obstacle.modifyObsFunc[itr] is not None:
                    idx = ref_index[itr]
            if size == 1:
                nx, ny = 0, 0
                nProc_x, nProc_y = 1, 1
                N_local = self.N_local
            else:
                nx, ny = mpiParams.nx, mpiParams.ny
                nProc_x, nProc_y = mpiParams.nProc_x, mpiParams.nProc_y
                N_local = mpiParams.N_local
            if self.phase is False:
                args = (fields.f, fields.f_new, fields.solid,
                        fields.procBoundary, fields.boundaryNode,
                        self.surfaceNodes[itr], self.surfaceInvList[itr],
                        self.surfaceOutList[itr], lattice.c, lattice.invList,
                        self.obstacleFlag[itr], mesh.Nx, mesh.Ny,
                        mesh.Nx_global, mesh.Ny_global, tempForces,
                        tempTorque, lattice.noOfDirections,
                        self.computeForces, self.computeTorque, idx, N_local,
                        nx, ny, nProc_x, nProc_y, size, timeStep)
                self.forceTorqueFunc(*args)
            else:
                if phaseField.segregation is True:
                    phiWeight = phaseField.phiWeight
                else:
                    phiWeight = np.zeros(lattice.noOfDirections,
                                         dtype=precision)
                args = (fields.f, fields.f_new, fields.solid, fields.phi,
                        fields.rho, fields.normalPhi, fields.boundaryNode,
                        fields.procBoundary, transport.sigma,
                        phaseField.interfaceWidth, self.surfaceNodes[itr],
                        self.surfaceNormalsFluid[itr],
                        self.surfaceInvList[itr], self.surfaceOutList[itr],
                        lattice.c, lattice.invList, self.obstacleFlag[itr],
                        mesh.Nx, mesh.Ny, mesh.Nx_global, mesh.Ny_global,
                        tempForces, tempTorque, lattice.noOfDirections,
                        self.computeForces, self.computeTorque, idx, N_local,
                        nx, ny, nProc_x, nProc_y, size, fields.stressTensor,
                        fields.p, lattice.cs, fields.curvature, fields.gradPhi,
                        fields.forceField, fields.solidNbNodesWhole, lattice.w,
                        phiWeight, tempForcesCap, tempForcesHyd, tempTorqueCap,
                        tempTorqueHyd, positionVector, leastDistance,
                        interfaceTangent, fields.deltaFuncFluid,
                        fields.contactAngleLocalFluid, cosThetaLocal,
                        fields.delPhiFluid, self.scheme)
                self.forceTorqueFuncPhase(*args)
            if self.obstacleFlag[itr] == 1:
                externalForce = obstacle.externalForce[itr]
                externalTorque = obstacle.externalTorque[itr]
            else:
                externalForce = np.zeros(2, dtype=precision)
                externalTorque = precision(0)
            self.forces.append(np.sum(tempForces, axis=0) +
                               self.forceReconstruction[itr] +
                               externalForce)
            self.capForces.append(np.sum(tempForcesCap, axis=0))
            self.hydForces.append(np.sum(tempForcesHyd, axis=0))
            self.capTorque.append(np.sum(tempTorqueCap, axis=0))
            self.hydTorque.append(np.sum(tempTorqueHyd, axis=0))
            self.torque.append(np.sum(tempTorque, axis=0) +
                               self.torqueReconstruction[itr] +
                               externalTorque)
            # if not os.path.isdir("postProcessing"):
            #     os.makedirs("postProcessing")
            # if timeStep % 1000 == 0 and itr == 0:
            #     np.savez("postProcessing/forceTorque_surface_" +
            #              str(itr) + "_t_" + str(timeStep) + ".npz",
            #              surfaceNodes=self.surfaceNodes[itr],
            #              surfaceNormals=self.surfaceNormalsFluid[itr],
            #              tempForces=tempForces, tempForcesCap=tempForcesCap,
            #              tempForcesHyd=tempForcesHyd, tempTorque=tempTorque,
            #              tempTorqueCap=tempTorqueCap,
            #              tempTorqueHyd=tempTorqueHyd,
            #              positionVector=positionVector,
            #              leastDistance=leastDistance,
            #              interfaceTangent=interfaceTangent,
            #              delta=delta, delPhi=delPhi,
            #              contactAngleLocal=contactAngleLocal,
            #              cosThetaLocal=cosThetaLocal,
            #              solid=fields.solid, phi=fields.phi,
            #              fluidBoundary=fields.fluidBoundary)
        # print(self.forces)

    def details(self, rank):
        # print('\n\n\n')
        print(rank, self.noOfSurfaces)
        print(rank, self.surfaceNames)
        # print(rank, self.surfaceNamesGlobal)
        print(rank, self.surfaceNodes)
        print(rank, self.obstacleFlag)
        # print(rank, self.x_ref_idx)
        pass

    def setupForcesParallel_cpu(self, parallel):
        if self.phase is True:
            self.forceTorqueFuncPhase = numba.njit(self.forceTorqueFuncPhase,
                                                   parallel=parallel,
                                                   cache=False,
                                                   nogil=True)
        else:
            self.forceTorqueFunc = numba.njit(self.forceTorqueFunc,
                                              parallel=parallel,
                                              cache=False,
                                              nogil=True)
        self.solidNormal = numba.njit(solidNormal, parallel=parallel,
                                      cache=False, nogil=True)
        self.solidNormalCircle = \
            numba.njit(solidNormalCircle, parallel=parallel, cache=False,
                       nogil=True)
        self.initializeSolidBoundaryNodes =\
            numba.njit(initializeSolidBoundaryNodes, parallel=parallel,
                       cache=False, nogil=True)

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

    def writeForces(self, timeStep, names, forces, torque, capForces,
                    hydForces, capTorque, hydTorque):
        if not os.path.isdir('postProcessing'):
            os.makedirs('postProcessing')
        if not os.path.isdir('postProcessing/' + str(timeStep)):
            os.makedirs('postProcessing/' + str(timeStep))
        if self.computeForces is True:
            writeFile = open('postProcessing/' + str(timeStep) + '/forces.dat',
                             'w')
            writeFile.write('surface_ID'.ljust(12) + '\t' +
                            'surface'.ljust(12) + '\t' +
                            'F_x'.ljust(18) + '\t' +
                            'F_y'.ljust(18) + '\t' +
                            'FCap_x'.ljust(18) + '\t' +
                            'FCap_y'.ljust(18) + '\t' +
                            'FHyd_x'.ljust(18) + '\t' +
                            'FHyd_y'.ljust(18) + '\n')
            for itr in range(len(names)):
                writeFile.\
                    write(str(itr).ljust(12) + '\t' +
                          (names[itr]).ljust(12) + '\t' +
                          str(round(forces[itr, 0], 16)).ljust(18) + '\t' +
                          str(round(forces[itr, 1], 16)).ljust(18) + '\t' +
                          str(round(capForces[itr, 0], 16)).ljust(18) + '\t' +
                          str(round(capForces[itr, 1], 16)).ljust(18) + '\t' +
                          str(round(hydForces[itr, 0], 16)).ljust(18) + '\t' +
                          str(round(hydForces[itr, 1], 16)).ljust(18) + '\n')
        if self.computeTorque is True:
            writeFile = open('postProcessing/' + str(timeStep) + '/torque.dat',
                             'w')
            writeFile.write('surface_ID'.ljust(12) + '\t' +
                            'surface'.ljust(12) + '\t' +
                            'T'.ljust(18) + '\t' +
                            'T_cap'.ljust(18) + '\t' +
                            'T_hyd'.ljust(18) + '\n')
            for itr in range(len(names)):
                writeFile.\
                    write(str(itr).ljust(12) + '\t' +
                          (names[itr]).ljust(12) + '\t' +
                          str(round(torque[itr], 16)).ljust(18) + '\t' +
                          str(round(capTorque[itr], 16)).ljust(18) + '\t' +
                          str(round(hydTorque[itr], 16)).ljust(18) + '\n')


def forceTorque(f, f_new, solid, procBoundary, boundaryNode, surfaceNodes,
                surfaceInvList, surfaceOutList, c, invList, obstacleFlag,
                Nx, Ny, Nx_global, Ny_global, forces, torque, noOfDirections,
                computeForces, computeTorque, x_ref, N_local, nx, ny,
                nProc_x, nProc_y, size, timeStep):
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
                            leastDist = r_0 * r_0 + r_1 * r_1
                            checkDist = ((i_global - x_ref[0] - Nx_global + 2) *
                                         (i_global - x_ref[0] - Nx_global + 2) +
                                         (j_global - x_ref[1]) *
                                         (j_global - x_ref[1]))
                            if checkDist < leastDist:
                                leastDist = checkDist
                                r_0 = i_global - x_ref[0] - Nx_global + 2
                                r_1 = j_global - x_ref[1]
                            checkDist = ((i_global - x_ref[0] + Nx_global - 2) *
                                         (i_global - x_ref[0] + Nx_global - 2) +
                                         (j_global - x_ref[1]) *
                                         (j_global - x_ref[1]))
                            if checkDist < leastDist:
                                leastDist = checkDist
                                r_0 = i_global - x_ref[0] + Nx_global - 2
                                r_1 = j_global - x_ref[1]
                            checkDist = ((i_global - x_ref[0]) *
                                         (i_global - x_ref[0]) +
                                         (j_global - x_ref[1] - Ny_global + 2) *
                                         (j_global - x_ref[1] - Ny_global + 2))
                            if checkDist < leastDist:
                                leastDist = checkDist
                                r_0 = i_global - x_ref[0]
                                r_1 = j_global - x_ref[1] - Ny_global + 2
                            checkDist = ((i_global - x_ref[0]) *
                                         (i_global - x_ref[0]) +
                                         (j_global - x_ref[1] + Ny_global - 2) *
                                         (j_global - x_ref[1] + Ny_global - 2))
                            if checkDist < leastDist:
                                leastDist = checkDist
                                r_0 = i_global - x_ref[0]
                                r_1 = j_global - x_ref[1] + Ny_global - 2
                            checkDist = ((i_global - x_ref[0] - Nx_global + 2) *
                                         (i_global - x_ref[0] - Nx_global + 2) +
                                         (j_global - x_ref[1] - Ny_global + 2) *
                                         (j_global - x_ref[1] - Ny_global + 2))
                            if checkDist < leastDist:
                                leastDist = checkDist
                                r_0 = i_global - x_ref[0] - Nx_global + 2
                                r_1 = j_global - x_ref[1] - Ny_global + 2
                            checkDist = ((i_global - x_ref[0] + Nx_global - 2) *
                                         (i_global - x_ref[0] + Nx_global - 2) +
                                         (j_global - x_ref[1] - Ny_global + 2) *
                                         (j_global - x_ref[1] - Ny_global + 2))
                            if checkDist < leastDist:
                                leastDist = checkDist
                                r_0 = i_global - x_ref[0] + Nx_global - 2
                                r_1 = j_global - x_ref[1] - Ny_global + 2
                            checkDist = ((i_global - x_ref[0] - Nx_global + 2) *
                                         (i_global - x_ref[0] - Nx_global + 2) +
                                         (j_global - x_ref[1] + Ny_global - 2) *
                                         (j_global - x_ref[1] + Ny_global - 2))
                            if checkDist < leastDist:
                                leastDist = checkDist
                                r_0 = i_global - x_ref[0] - Nx_global + 2
                                r_1 = j_global - x_ref[1] + Ny_global - 2
                            checkDist = ((i_global - x_ref[0] + Nx_global - 2) *
                                         (i_global - x_ref[0] + Nx_global - 2) +
                                         (j_global - x_ref[1] + Ny_global - 2) *
                                         (j_global - x_ref[1] + Ny_global - 2))
                            if checkDist < leastDist:
                                leastDist = checkDist
                                r_0 = i_global - x_ref[0] + Nx_global - 2
                                r_1 = j_global - x_ref[1] + Ny_global - 2
                            torque[itr] += (r_0 * value_1 - r_1 * value_0)


@numba.njit
def computeFieldsWall(i, j, ind, surfaceNormalsFluid_x, surfaceNormalsFluid_y,
                      boundaryNode, phi, gradPhi, Nx, Ny, size):
    i_nb = i - 0.5 * surfaceNormalsFluid_x
    j_nb = j - 0.5 * surfaceNormalsFluid_y
    i_next, j_next = int(np.ceil(i_nb)), int(np.ceil(j_nb))
    i_prev, j_prev = int(np.floor(i_nb)), int(np.floor(j_nb))
    if i_next == i_prev and j_next == j_prev:
        deltaX, deltaY = i_next - i, j_next - j
    else:
        deltaX, deltaY = i_next - i_prev, j_next - j_prev
    i_next_phi, j_next_phi = i_next, j_next
    i_prev_phi, j_prev_phi = i_prev, j_prev
    if size == 1 and boundaryNode[ind] == 2:
        if i_next == Nx - 1:
            i_next_phi = (i_next + 2 + Nx) % Nx
        if j_next == Ny - 1:
            j_next_phi = (j_next + 2 + Ny) % Ny
        if i_prev == 0:
            i_prev_phi = (i_prev - 2 + Nx) % Nx
        if j_prev == 0:
            j_prev_phi = (j_prev - 2 + Ny) % Ny
    elif size > 1 and boundaryNode[ind] == 2:
        if i_next == Nx - 2:
            i_next_phi = i_next + 1
        if j_next == Ny - 2:
            j_next_phi = j_next + 1
        if i_prev == 1:
            i_prev_phi = i_prev - 1
        if j_prev == 1:
            j_prev_phi = j_prev - 1
    if deltaX == 0:
        if surfaceNormalsFluid_y > 0:
            phiWall = \
                (phi[int(i_prev_phi * Ny + j_prev_phi)] +
                 phi[ind]) * 0.5
            gradPhiWall_x = \
                (gradPhi[int(i_prev_phi * Ny + j_prev_phi), 0] +
                 gradPhi[ind, 0]) * 0.5
            gradPhiWall_y = \
                (gradPhi[int(i_prev_phi * Ny + j_prev_phi), 1] +
                 gradPhi[ind, 1]) * 0.5
        elif surfaceNormalsFluid_y < 0:
            phiWall = \
                (phi[int(i_prev_phi * Ny + j_next_phi)] +
                 phi[ind]) * 0.5
            gradPhiWall_x = \
                (gradPhi[int(i_prev_phi * Ny + j_next_phi), 0] +
                 gradPhi[ind, 0]) * 0.5
            gradPhiWall_y = \
                (gradPhi[int(i_prev_phi * Ny + j_next_phi), 1] +
                 gradPhi[ind, 1]) * 0.5
    elif deltaY == 0:
        if surfaceNormalsFluid_x > 0:
            phiWall = \
                (phi[int(i_prev_phi * Ny + j_prev_phi)] +
                 phi[ind]) * 0.5
            gradPhiWall_x = \
                (gradPhi[int(i_prev_phi * Ny + j_prev_phi), 0] +
                 gradPhi[ind, 0]) * 0.5
            gradPhiWall_y = \
                (gradPhi[int(i_prev_phi * Ny + j_prev_phi), 1] +
                 gradPhi[ind, 1]) * 0.5
        elif surfaceNormalsFluid_x < 0:
            phiWall = \
                (phi[int(i_next_phi * Ny + j_prev_phi)] +
                 phi[ind]) * 0.5
            gradPhiWall_x = \
                (gradPhi[int(i_next_phi * Ny + j_prev_phi), 0] +
                 gradPhi[ind, 0]) * 0.5
            gradPhiWall_y = \
                (gradPhi[int(i_next_phi * Ny + j_prev_phi), 1] +
                 gradPhi[ind, 1]) * 0.5
    elif deltaX != 0 and deltaY != 0:
        phiWall = ((i_next - i_nb) * (j_next - j_nb) *
                   phi[i_prev_phi * Ny + j_prev_phi] +
                   (i_next - i_nb) * (j_nb - j_prev) *
                   phi[i_prev_phi * Ny + j_next_phi] +
                   (i_nb - i_prev) * (j_next - j_nb) *
                   phi[i_next_phi * Ny + j_prev_phi] +
                   (i_nb - i_prev) * (j_nb - j_prev) *
                   phi[i_next_phi * Ny + j_next_phi]) /\
            (deltaX * deltaY)
        gradPhiWall_x =\
            ((i_next - i_nb) * (j_next - j_nb) *
             gradPhi[i_prev_phi * Ny + j_prev_phi, 0] +
             (i_next - i_nb) * (j_nb - j_prev) *
             gradPhi[i_prev_phi * Ny + j_next_phi, 0] +
             (i_nb - i_prev) * (j_next - j_nb) *
             gradPhi[i_next_phi * Ny + j_prev_phi, 0] +
             (i_nb - i_prev) * (j_nb - j_prev) *
             gradPhi[i_next_phi * Ny + j_next_phi, 0])
        gradPhiWall_y =\
            ((i_next - i_nb) * (j_next - j_nb) *
             gradPhi[i_prev_phi * Ny + j_prev_phi, 1] +
             (i_next - i_nb) * (j_nb - j_prev) *
             gradPhi[i_prev_phi * Ny + j_next_phi, 1] +
             (i_nb - i_prev) * (j_next - j_nb) *
             gradPhi[i_next_phi * Ny + j_prev_phi, 1] +
             (i_nb - i_prev) * (j_nb - j_prev) *
             gradPhi[i_next_phi * Ny + j_next_phi, 1])
    return phiWall, gradPhiWall_x, gradPhiWall_y


def forceTorquePhase(f, f_new, solid, phi, rho, normalPhi, boundaryNode,
                     procBoundary, sigma, interfaceWidth, surfaceNodes,
                     surfaceNormalsFluid, surfaceInvList, surfaceOutList,
                     c, invList, obstacleFlag, Nx, Ny, Nx_global, Ny_global,
                     forces, torque, noOfDirections, computeForces,
                     computeTorque, x_ref, N_local, nx, ny, nProc_x, nProc_y,
                     size, stressTensor, p, cs, curvature, gradPhi,
                     forceField, solidNodesWhole, w, phiWeight, capForces,
                     hydForces, capTorque, hydTorque, positionVector,
                     leastDistance, interfaceTangent, deltaFluid,
                     contactAngleLocalFluid, cosThetaLocal, delPhiAllFluid,
                     scheme):
    for itr in prange(surfaceNodes.shape[0]):
        ind = surfaceNodes[itr]
        if procBoundary[ind] != 1:
            if obstacleFlag == 0:
                i, j = int(ind / Ny), int(ind % Ny)
                for k in range(surfaceOutList.shape[0]):
                    value_0 = ((f[ind, surfaceOutList[k]] *
                               c[surfaceOutList[k], 0]) -
                               (f_new[ind, surfaceInvList[k]]
                               * c[surfaceInvList[k], 0]))
                    value_1 = ((f[ind, surfaceOutList[k]] *
                               c[surfaceOutList[k], 1]) -
                               (f_new[ind, surfaceInvList[k]]
                               * c[surfaceInvList[k], 1]))
                    # if i == 21 and j == 1:
                    #     print("k = ", surfaceOutList[k], ", k_inv = ",
                    #           surfaceInvList[k])
                    #     print("x")
                    #     print(value_0,
                    #           (f[ind, surfaceOutList[k]] *
                    #            c[surfaceOutList[k], 0]),
                    #           (f_new[ind, surfaceInvList[k]]
                    #            * c[surfaceInvList[k], 0]))
                    #     print("y")
                    #     print(value_1,
                    #           (f[ind, surfaceOutList[k]] *
                    #            c[surfaceOutList[k], 1]),
                    #           (f_new[ind, surfaceInvList[k]]
                    #            * c[surfaceInvList[k], 1]))
                    #     # print("\n")
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
                        positionVector[itr, 0] = r_0
                        positionVector[itr, 1] = r_1
                        leastDistance[itr] = np.sqrt(r_0 * r_0 + r_1 * r_1)
                        torque[itr] += (r_0 * value_1 - r_1 * value_0)
                # if i == 21 and j == 1:
                #     print(forces[itr, :], rho[ind])
                #     print("\n")
                hydTorque[itr] = torque[itr]
                tangent = normalPhi[ind, 0] * surfaceNormalsFluid[itr, 1] -\
                    surfaceNormalsFluid[itr, 0] * normalPhi[ind, 1]
                normalCap_x = -normalPhi[ind, 1] * tangent
                normalCap_y = normalPhi[ind, 0] * tangent
                magNormal = np.sqrt(normalCap_x * normalCap_x +
                                    normalCap_y * normalCap_y)
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
                capForces[itr, 0] = capForce_x
                capForces[itr, 1] = capForce_y
                hydForces[itr, 0] = forces[itr, 0]
                hydForces[itr, 1] = forces[itr, 1]
                forces[itr, 0] += capForce_x
                forces[itr, 1] += capForce_y
                if computeTorque is True:
                    torque[itr] += (r_0 * capForce_y - r_1 * capForce_x)
                    capTorque[itr] = (r_0 * capForce_y - r_1 * capForce_x)
            elif obstacleFlag == 1:
                i, j = int(ind / Ny), int(ind % Ny)
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
                    leastDist = r_0 * r_0 + r_1 * r_1
                    checkDist = ((i_global - x_ref[0] - Nx_global + 2) *
                                 (i_global - x_ref[0] - Nx_global + 2) +
                                 (j_global - x_ref[1]) *
                                 (j_global - x_ref[1]))
                    if checkDist < leastDist:
                        leastDist = checkDist
                        r_0 = i_global - x_ref[0] - Nx_global + 2
                        r_1 = j_global - x_ref[1]
                    checkDist = ((i_global - x_ref[0] + Nx_global - 2) *
                                 (i_global - x_ref[0] + Nx_global - 2) +
                                 (j_global - x_ref[1]) *
                                 (j_global - x_ref[1]))
                    if checkDist < leastDist:
                        leastDist = checkDist
                        r_0 = i_global - x_ref[0] + Nx_global - 2
                        r_1 = j_global - x_ref[1]
                    checkDist = ((i_global - x_ref[0]) *
                                 (i_global - x_ref[0]) +
                                 (j_global - x_ref[1] - Ny_global + 2) *
                                 (j_global - x_ref[1] - Ny_global + 2))
                    if checkDist < leastDist:
                        leastDist = checkDist
                        r_0 = i_global - x_ref[0]
                        r_1 = j_global - x_ref[1] - Ny_global + 2
                    checkDist = ((i_global - x_ref[0]) *
                                 (i_global - x_ref[0]) +
                                 (j_global - x_ref[1] + Ny_global - 2) *
                                 (j_global - x_ref[1] + Ny_global - 2))
                    if checkDist < leastDist:
                        leastDist = checkDist
                        r_0 = i_global - x_ref[0]
                        r_1 = j_global - x_ref[1] + Ny_global - 2
                    checkDist = ((i_global - x_ref[0] - Nx_global + 2) *
                                 (i_global - x_ref[0] - Nx_global + 2) +
                                 (j_global - x_ref[1] - Ny_global + 2) *
                                 (j_global - x_ref[1] - Ny_global + 2))
                    if checkDist < leastDist:
                        leastDist = checkDist
                        r_0 = i_global - x_ref[0] - Nx_global + 2
                        r_1 = j_global - x_ref[1] - Ny_global + 2
                    checkDist = ((i_global - x_ref[0] + Nx_global - 2) *
                                 (i_global - x_ref[0] + Nx_global - 2) +
                                 (j_global - x_ref[1] - Ny_global + 2) *
                                 (j_global - x_ref[1] - Ny_global + 2))
                    if checkDist < leastDist:
                        leastDist = checkDist
                        r_0 = i_global - x_ref[0] + Nx_global - 2
                        r_1 = j_global - x_ref[1] - Ny_global + 2
                    checkDist = ((i_global - x_ref[0] - Nx_global + 2) *
                                 (i_global - x_ref[0] - Nx_global + 2) +
                                 (j_global - x_ref[1] + Ny_global - 2) *
                                 (j_global - x_ref[1] + Ny_global - 2))
                    if checkDist < leastDist:
                        leastDist = checkDist
                        r_0 = i_global - x_ref[0] - Nx_global + 2
                        r_1 = j_global - x_ref[1] + Ny_global - 2
                    checkDist = ((i_global - x_ref[0] + Nx_global - 2) *
                                 (i_global - x_ref[0] + Nx_global - 2) +
                                 (j_global - x_ref[1] + Ny_global - 2) *
                                 (j_global - x_ref[1] + Ny_global - 2))
                    if checkDist < leastDist:
                        leastDist = checkDist
                        r_0 = i_global - x_ref[0] + Nx_global - 2
                        r_1 = j_global - x_ref[1] + Ny_global - 2
                positionVector[itr, 0] = r_0
                positionVector[itr, 1] = r_1
                leastDistance[itr] = leastDist
                for k in range(noOfDirections):
                    i_nb = int(i + c[k, 0] + Nx) % Nx
                    j_nb = int(j + c[k, 1] + Ny) % Ny
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
                    if solid[ind_nb, 0] == 1:
                        # value_0 = ((f[ind, k] - phiWeight[k]) * c[k, 0] -
                        #            (f_new[ind, invList[k]] - phiWeight[k])
                        #            * c[invList[k], 0])
                        # value_1 = ((f[ind, k] - phiWeight[k]) * c[k, 1] -
                        #            (f_new[ind, invList[k]] - phiWeight[k])
                        #            * c[invList[k], 1])
                        value_0 = (f[ind, k] * c[k, 0] -
                                   f_new[ind, invList[k]]
                                   * c[invList[k], 0])
                        value_1 = (f[ind, k] * c[k, 1] -
                                   f_new[ind, invList[k]]
                                   * c[invList[k], 1])
                        # if i == 21 and j == 2:
                        #     print("k = ", surfaceOutList[k], ", k_inv = ",
                        #           surfaceInvList[k])
                        #     print("x")
                        #     print(value_0,
                        #           (f[ind, k] * c[k, 0]),
                        #           (f_new[ind, invList[k]]
                        #            * c[invList[k], 0]))
                        #     print("y")
                        #     print(value_1,
                        #           (f[ind, k] * c[k, 1]),
                        #           (f_new[ind, invList[k]]
                        #            * c[invList[k], 1]))
                        #     # print("\n")
                        if computeForces is True:
                            forces[itr, 0] += value_0 * rho[ind]
                            forces[itr, 1] += value_1 * rho[ind]
                        if computeTorque is True:
                            torque[itr] += (r_0 * value_1 - r_1 * value_0)
                # if i == 21 and j == 2:
                #     print(forces[itr, :], rho[ind])
                #     print("\n")
                hydTorque[itr] = torque[itr]
                phiWall, gradPhiWall_x, gradPhiWall_y =\
                    computeFieldsWall(i, j, ind, surfaceNormalsFluid[itr, 0],
                                      surfaceNormalsFluid[itr, 1],
                                      boundaryNode, phi, gradPhi, Nx, Ny, size)
                magGradPhiWall = np.sqrt(gradPhiWall_x * gradPhiWall_x +
                                         gradPhiWall_y * gradPhiWall_y)
                normalPhiWall_x = gradPhiWall_x / (magGradPhiWall + 1e-17)
                normalPhiWall_y = gradPhiWall_y / (magGradPhiWall + 1e-17)
                tangent = normalPhiWall_x * surfaceNormalsFluid[itr, 1] -\
                    surfaceNormalsFluid[itr, 0] * normalPhiWall_y
                delx, dely = 0, 0
                if solidNodesWhole[int((i + 1) * Ny + j)] == 1:
                    delx += 0.5
                if solidNodesWhole[int((i - 1) * Ny + j)] == 1:
                    delx += 0.5
                if solidNodesWhole[int(i * Ny + j + 1)] == 1:
                    dely += 0.5
                if solidNodesWhole[int(i * Ny + j - 1)] == 1:
                    dely += 0.5
                interfaceTangent_x = -normalPhiWall_y * tangent
                interfaceTangent_y = normalPhiWall_x * tangent
                solidTangent_x = - surfaceNormalsFluid[itr, 1]
                solidTangent_y = surfaceNormalsFluid[itr, 0]
                magTangent = \
                    np.sqrt(interfaceTangent_x * interfaceTangent_x +
                            interfaceTangent_y * interfaceTangent_y)\
                    + 1e-20
                # normalPhiDotSolidTangent = \
                #     np.abs(normalPhi[ind, 0] * solidTangent_x +
                #            normalPhi[ind, 1] * solidTangent_y)
                gradPhiDotSolidTangent = \
                    gradPhiWall_x * solidTangent_x +\
                    gradPhiWall_y * solidTangent_y
                gradPhiDotSolidNormal = \
                    gradPhiWall_x * surfaceNormalsFluid[itr, 0] +\
                    gradPhiWall_y * surfaceNormalsFluid[itr, 1]
                ###########################################################
                if scheme == 1:
                    '''
                    Zhang scheme
                    scheme - 1
                    '''
                    delPhi = np.abs(gradPhiWall_x) * delx +\
                        np.abs(gradPhiWall_y) * dely
                    capForce_x = 6 * sigma * phiWall * (1 - phiWall) * \
                        (interfaceTangent_x / magTangent) * delPhi
                    capForce_y = 6 * sigma * phiWall * (1 - phiWall) * \
                        (interfaceTangent_y / magTangent) * delPhi
                    '''
                    '''
                ###########################################################
                elif scheme == 2:
                    # print("Hi")
                    '''
                    Projected New
                    scheme - 2
                    '''
                    # magGrad = np.abs(4 * phi[ind] * (1 - phi[ind]) /
                    #                  interfaceWidth)
                    # gradPhi_x = magGrad * normalPhi[ind, 0]
                    # gradPhi_y = magGrad * normalPhi[ind, 1]
                    # delPhi = np.abs(gradPhi_x) * delx +\
                    #     np.abs(gradPhi_y) * dely
                    # capForce_x = 6 * sigma * phi[ind] * (1 - phi[ind]) * \
                    #     (interfaceTangent_x / magTangent) * delPhi
                    # capForce_y = 6 * sigma * phi[ind] * (1 - phi[ind]) * \
                    #     (interfaceTangent_y / magTangent) * delPhi
                    delVector_x = delx * solidTangent_x / \
                        (np.abs(solidTangent_x) + 1e-17)
                    delVector_y = dely * solidTangent_y / \
                        (np.abs(solidTangent_y) + 1e-17)
                    dels = delVector_x * solidTangent_x +\
                        delVector_y * solidTangent_y
                    deln = delVector_x * surfaceNormalsFluid[itr, 0] +\
                        delVector_y * surfaceNormalsFluid[itr, 1]
                    delPhi = np.abs(gradPhiDotSolidTangent * dels +
                                    gradPhiDotSolidNormal * deln)
                    capForce_x = \
                        sigma * 6.0 * phiWall * (1.0 - phiWall) *\
                        interfaceTangent_x * delPhi / magTangent
                    capForce_y = \
                        sigma * 6.0 * phiWall * (1.0 - phiWall) *\
                        interfaceTangent_y * delPhi / magTangent
                    '''
                    '''
                ###########################################################
                hydForces[itr, 0] = forces[itr, 0]
                hydForces[itr, 1] = forces[itr, 1]
                forces[itr, 0] += capForce_x
                forces[itr, 1] += capForce_y
                capForces[itr, 0] = capForce_x
                capForces[itr, 1] = capForce_y
                # if i < Nx // 2:
                #     capForces[itr, 0] = capForce_x
                #     capForces[itr, 1] = capForce_y
                # else:
                #     capForces[itr, 0] = 0
                #     capForces[itr, 1] = 0
                interfaceTangent[itr, 0] = (normalCap_x / (magNormal + 1e-17))
                interfaceTangent[itr, 1] = (normalCap_y / (magNormal + 1e-17))
                # delta[itr] = 6 * phi[ind] * (1 - phi[ind])
                # delPhiAll[itr] = delPhi
                deltaFluid[ind] = 6 * phiWall * (1 - phiWall)
                delPhiAllFluid[ind] = delPhi
                cosTheta = surfaceNormalsFluid[itr, 0] * normalPhiWall_x +\
                    surfaceNormalsFluid[itr, 1] * normalPhiWall_y
                # contactAngleLocal[itr] = np.arccos(- cosTheta)
                cosThetaLocal[itr] = cosTheta
                contactAngleLocalFluid[ind] = np.arccos(-cosTheta)
                i, j = int(ind / Ny), int(ind % Ny)
                if computeTorque is True:
                    torque[itr] = (r_0 * forces[itr, 1] - r_1 * forces[itr, 0])
                    capTorque[itr] = (r_0 * capForce_y - r_1 * capForce_x)
                else:
                    torque[itr] = 0
                    capTorque[itr] = 0


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
