import numpy as np
import os
import numba
from numba import cuda

from pylabolt.base import (boundaryConditions, boundaryConditions_cuda,
                           scalarBoundaryConditions)


@numba.njit
def markBoundaryElements(Nx, Ny, invList, noOfDirections, boundaryIndices,
                         boundaryNode, solid, periodicFlag=False, wall=False):
    faceList = []
    nbList = []
    cornerList = []
    # print(boundaryIndices)
    indX_i, indX_f = boundaryIndices[0, 0], boundaryIndices[1, 0]
    indY_i, indY_f = boundaryIndices[0, 1], boundaryIndices[1, 1]
    diffX = indX_f - indX_i
    diffY = indY_f - indY_i
    bottom, top = False, False
    left, right = False, False
    if diffY == 1 and Ny != 1:
        if Nx > 1:
            indX_f += 2
        if indY_i == 0:
            bottom = True
            if noOfDirections == 9:
                outDirections = np.array([7, 4, 8], dtype=np.int32)
                invDirections = np.array([invList[7], invList[4], invList[8]],
                                         dtype=np.int32)
            if noOfDirections == 3:
                outDirections = np.array([2], dtype=np.int32)
                invDirections = np.array([invList[2]], dtype=np.int32)
        elif indY_i + 2 == Ny - 1:
            indY_i += 2
            indY_f += 2
            top = True
            if noOfDirections == 9:
                outDirections = np.array([6, 2, 5], dtype=np.int32)
                invDirections = np.array([invList[6], invList[2], invList[5]],
                                         dtype=np.int32)
            if noOfDirections == 3:
                outDirections = np.array([1], dtype=np.int32)
                invDirections = np.array([invList[1]], dtype=np.int32)
    if diffX == 1 and Nx != 1:
        if Ny > 1:
            indY_f += 2
        if indX_i == 0:
            left = True
            if noOfDirections == 9:
                outDirections = np.array([6, 3, 7], dtype=np.int32)
                invDirections = np.array([invList[6], invList[3], invList[7]],
                                         dtype=np.int32)
            if noOfDirections == 3:
                outDirections = np.array([2], dtype=np.int32)
                invDirections = np.array([invList[2]], dtype=np.int32)
        elif indX_i + 2 == Nx - 1:
            indX_i += 2
            indX_f += 2
            right = True
            if noOfDirections == 9:
                outDirections = np.array([1, 5, 8], dtype=np.int32)
                invDirections = np.array([invList[1], invList[5], invList[8]],
                                         dtype=np.int32)
            if noOfDirections == 3:
                outDirections = np.array([1], dtype=np.int32)
                invDirections = np.array([invList[1]], dtype=np.int32)
    for ind in range(Nx * Ny):
        for i in range(indX_i, indX_f):
            for j in range(indY_i, indY_f):
                currentId = int(i * Ny + j)
                if currentId == ind:
                    boundaryNode[ind] = 1
                    if wall is True:
                        solid[ind, 0] = 1
                    if top is True and i > 0 and i < Nx - 1:
                        fluidBoundaryNode = int(i * Ny + (j - 1))
                        faceList.append(fluidBoundaryNode)
                        if periodicFlag is True:
                            boundaryNode[fluidBoundaryNode] = 2
                            if solid[fluidBoundaryNode, 0] == 1:
                                solid[ind, 0] = 1
                        else:
                            boundaryNode[fluidBoundaryNode] = 0
                        nbList.append(ind)
                    elif bottom is True and i > 0 and i < Nx - 1:
                        fluidBoundaryNode = int(i * Ny + (j + 1))
                        faceList.append(fluidBoundaryNode)
                        if periodicFlag is True:
                            boundaryNode[fluidBoundaryNode] = 2
                            if solid[fluidBoundaryNode, 0] == 1:
                                solid[ind, 0] = 1
                        else:
                            boundaryNode[fluidBoundaryNode] = 0
                        nbList.append(ind)
                    elif left is True and j > 0 and j < Ny - 1:
                        fluidBoundaryNode = int((i + 1) * Ny + j)
                        faceList.append(fluidBoundaryNode)
                        if periodicFlag is True:
                            boundaryNode[fluidBoundaryNode] = 2
                            if solid[fluidBoundaryNode, 0] == 1:
                                solid[ind, 0] = 1
                        else:
                            boundaryNode[fluidBoundaryNode] = 0
                        nbList.append(ind)
                    elif right is True and j > 0 and j < Ny - 1:
                        fluidBoundaryNode = int((i - 1) * Ny + j)
                        faceList.append(fluidBoundaryNode)
                        if periodicFlag is True:
                            boundaryNode[fluidBoundaryNode] = 2
                            if solid[fluidBoundaryNode, 0] == 1:
                                solid[ind, 0] = 1
                        else:
                            boundaryNode[fluidBoundaryNode] = 0
                        nbList.append(ind)
                    if i == 0:
                        if j == 0 or j == Ny - 1:
                            cornerList.append(ind)
                    if i == Nx - 1:
                        if j == 0 or j == Ny - 1:
                            cornerList.append(ind)
    return np.array(faceList, dtype=np.int64),\
        outDirections,\
        invDirections, np.array(nbList, dtype=np.int64),\
        np.array(cornerList, dtype=np.int64)


@numba.njit
def initializeBoundaryElementsFluid(scalar, vector, rho, u, nbList,
                                    cornerList):
    for ind in nbList:
        rho[ind] = scalar
        u[ind, 0] = vector[0]
        u[ind, 1] = vector[1]
    for ind in cornerList:
        rho[ind] = scalar
        u[ind, 0] = vector[0]
        u[ind, 1] = vector[1]


@numba.njit
def initializeVariableElementsFluid(scalar, vector, rho, u, nbList,
                                    cornerList):
    itr = 0
    for ind in nbList:
        rho[ind] = scalar
        u[ind, 0] = vector[itr, 0]
        u[ind, 1] = vector[itr, 1]
        itr += 1
    for ind in cornerList:
        rho[ind] = scalar
        u[ind, 0] = vector[itr, 0]
        u[ind, 1] = vector[itr, 1]
        itr += 1


@numba.njit
def initializeBoundaryElementsScalar(scalar, phi, nbList, cornerList):
    for ind in nbList:
        phi[ind] = scalar
    for ind in cornerList:
        phi[ind] = scalar


class boundary:
    def __init__(self, boundaryDict):
        self.boundaryDict = boundaryDict
        self.boundaryVectorFluid = []
        self.boundaryScalarFluid = []
        self.variableU_func = []
        self.boundaryScalarPhase = []
        self.boundaryScalarT = []
        self.boundaryTypeFluid = []
        self.boundaryTypePhase = []
        self.boundaryTypeT = []
        self.boundaryEntity = []
        self.boundaryFuncFluid = []
        self.boundaryFuncPhase = []
        self.boundaryFuncT = []
        self.points = {}
        self.nameList = list(boundaryDict.keys())
        self.boundaryIndices = []
        self.noOfBoundaries = 0
        self.faceList = []
        self.nbList = []
        self.cornerList = []
        self.invDirections = []
        self.outDirections = []
        self.fluid, self.phase = False, False
        self.T = False
        self.argsFluid = []
        self.argsPhase = []
        self.argsT = []

        # Cuda device data
        self.boundaryVector_device = []
        self.boundaryScalar_device = []
        self.boundaryIndices_device = []
        self.faceList_device = []
        self.invDirections_device = []
        self.outDirections_device = []

    def readBoundaryDict(self, initialFields, lattice, mesh, rank, precision):
        for item in self.nameList:
            dataList = list(self.boundaryDict[item].keys())
            tempPoints = []
            try:
                entity = self.boundaryDict[item]['entity']
                if entity != 'wall' and entity != 'patch':
                    if rank == 0:
                        print("ERROR! Invalid 'entity' type. Must be 'wall'" +
                              " or 'patch'", flush=True)
                    os._exit(0)
            except KeyError:
                if rank == 0:
                    print("ERROR! Missing keyword 'entity' " +
                          "in boundaryDict!")
                os._exit(0)
            flag = False
            for data in dataList:
                if data == 'fluid':
                    fluidBoundary = self.boundaryDict[item][data]
                    flag = True
                    self.fluid = True
                    vectorValueTempFluid, scalarValueTempFluid, \
                        boundaryTypeTempFluid, variableU_funcTemp = \
                        self.readFluidBoundary(fluidBoundary, initialFields,
                                               lattice, rank, precision)
                elif data == 'phase':
                    scalarBoundary = self.boundaryDict[item][data]
                    flag = True
                    self.phase = True
                    scalarValueTempPhase, boundaryTypeTempPhase = \
                        self.readScalarBoundary(scalarBoundary, initialFields,
                                                rank, precision)
                elif data == 'T':
                    scalarBoundary = self.boundaryDict[item][data]
                    flag = True
                    self.T = True
                    scalarValueTempT, boundaryTypeTempT = \
                        self.readScalarBoundary(scalarBoundary, initialFields,
                                                rank, precision)
                elif (data != 'fluid' and data != 'phase' and data != 'T'
                        and data != 'entity'):
                    tempPoints.append(self.boundaryDict[item][data])
                    self.boundaryEntity.append(entity)
                    if self.fluid is True:
                        self.boundaryTypeFluid.append(boundaryTypeTempFluid)
                        self.boundaryVectorFluid.append(vectorValueTempFluid)
                        self.boundaryScalarFluid.append(scalarValueTempFluid)
                        self.variableU_func.append(variableU_funcTemp)
                        self.boundaryFuncFluid.\
                            append(getattr(boundaryConditions,
                                   boundaryTypeTempFluid))
                    if self.phase is True:
                        self.boundaryTypePhase.append(boundaryTypeTempPhase)
                        self.boundaryScalarPhase.append(scalarValueTempPhase)
                        self.boundaryFuncPhase.\
                            append(getattr(scalarBoundaryConditions,
                                   boundaryTypeTempPhase))
                    if self.T is True:
                        self.boundaryTypeT.append(boundaryTypeTempT)
                        self.boundaryScalarT.append(scalarValueTempT)
                        self.boundaryFuncT.\
                            append(getattr(scalarBoundaryConditions,
                                   boundaryTypeTempT))
                self.points[item] = tempPoints
                if flag is False:
                    if rank == 0:
                        print("ERROR! No boundary conditions defined!",
                              flush=True)
                    os._exit(0)
        self.noOfBoundaries = len(self.boundaryEntity)

    def readFluidBoundary(self, fluidBoundary, initialFields, lattice,
                          rank, precision):
        try:
            boundaryTypeTempFluid = fluidBoundary['type']
            variableU_funcTemp = None
            if boundaryTypeTempFluid == 'fixedU':
                try:
                    if isinstance(fluidBoundary['value'], list):
                        vectorValueTemp = \
                            np.array(fluidBoundary['value'],
                                     dtype=precision)
                        scalarValueTemp = initialFields.defaultRho
                    else:
                        if rank == 0:
                            print("ERROR!")
                            print("For 'fixedU' value must be a list "
                                  + "of components: [x1, x2]",
                                  flush=True)
                        os._exit(1)
                except KeyError:
                    if rank == 0:
                        print('Key Error!')
                        print("'value' keyword required for type " +
                              "'fixedU'", flush=True)
                    os._exit(1)
            elif boundaryTypeTempFluid == 'fixedPressure':
                try:
                    if isinstance(fluidBoundary['value'],
                                  float):
                        vectorValueTemp = initialFields.defaultU
                        scalarValueTemp = precision(fluidBoundary['value']
                                                    * lattice.cs_2)
                    else:
                        if rank == 0:
                            print("ERROR!")
                            print("For 'fixedPressure' value must be" +
                                  " a float", flush=True)
                        os._exit(0)
                except KeyError:
                    if rank == 0:
                        print("ERROR!")
                        print("'value' keyword required for type " +
                              "'fixedPressure'", flush=True)
                    os._exit(0)
            elif boundaryTypeTempFluid == 'bounceBack':
                vectorValueTemp = initialFields.defaultU
                scalarValueTemp = initialFields.defaultRho
            elif boundaryTypeTempFluid == 'periodic':
                vectorValueTemp = initialFields.defaultU
                scalarValueTemp = initialFields.defaultRho
            elif boundaryTypeTempFluid == 'zeroGradient':
                vectorValueTemp = initialFields.defaultU
                scalarValueTemp = initialFields.defaultRho
            elif boundaryTypeTempFluid == 'variableU':
                import simulation
                try:
                    funcName = fluidBoundary['func']
                    variableU_funcTemp = \
                        getattr(simulation, funcName)
                except KeyError:
                    if rank == 0:
                        print("ERROR!")
                        print("'func' keyword required for type " +
                              "'variableU'", flush=True)
                    os._exit(0)
                except AttributeError:
                    if rank == 0:
                        print("ERROR!")
                        print(funcName + " is not defined in " +
                              "simulation.py", flush=True)
                    os._exit(0)
                vectorValueTemp = initialFields.defaultU
                scalarValueTemp = initialFields.defaultRho
            else:
                if rank == 0:
                    print("ERROR! " + fluidBoundary['type'] +
                          " is not a valid boundary condition!",
                          flush=True)
                    print('Please check boundary conditions available',
                          flush=True)
                    print("Refer to the tutorials and documentation",
                          flush=True)
                os._exit(0)
        except KeyError:
            if rank == 0:
                print("ERROR! 'type' keyword is required!",
                      flush=True)
        return (vectorValueTemp, scalarValueTemp, boundaryTypeTempFluid,
                variableU_funcTemp)

    def readScalarBoundary(self, scalarBoundary, initialFields, rank,
                           precision):
        try:
            boundaryTypeTempScalar = scalarBoundary['type']
            if boundaryTypeTempScalar == 'fixedValue':
                try:
                    if isinstance(scalarBoundary['value'],
                                  float):
                        scalarValueTemp = precision(scalarBoundary['value'])
                    else:
                        if rank == 0:
                            print("ERROR!")
                            print("For 'fixedValue' value must be" +
                                  " a float", flush=True)
                        os._exit(0)
                except KeyError:
                    if rank == 0:
                        print("ERROR!")
                        print("'value' keyword required for type " +
                              "'fixedValue'", flush=True)
                    os._exit(0)
            elif boundaryTypeTempScalar == 'zeroGradient':
                scalarValueTemp = precision(0.0)
            elif boundaryTypeTempScalar == 'bounceBack':
                scalarValueTemp = precision(0.0)
            elif boundaryTypeTempScalar == 'periodic':
                scalarValueTemp = precision(0.0)
            else:
                if rank == 0:
                    print("ERROR! " + scalarBoundary['type'] +
                          " is not a valid boundary condition!",
                          flush=True)
                    print('Please check boundary conditions available',
                          flush=True)
                    print("Refer to the tutorials and documentation",
                          flush=True)
                os._exit(0)
        except KeyError:
            if rank == 0:
                print("ERROR! 'type' keyword is required!",
                      flush=True)
        return scalarValueTemp, boundaryTypeTempScalar

    def initializeBoundary(self, lattice, mesh, fields, solid, precision):
        for name in self.nameList:
            pointArray = np.array(self.points[name])
            for k in range(pointArray.shape[0]):
                if k == 0:
                    delS = mesh.delX
                else:
                    delS = mesh.delX
                tempIndex_i = np.rint(pointArray[k, 0]/delS).\
                    astype(int)
                tempIndex_f = np.rint(pointArray[k, 1]/delS).\
                    astype(int) + 1
                self.boundaryIndices.append([tempIndex_i, tempIndex_f])
        self.boundaryIndices = np.array(self.boundaryIndices)
        for itr in range(self.noOfBoundaries):
            periodicFlag, wallFlag = False, False
            if self.fluid is True:
                if self.boundaryTypeFluid[itr] == 'periodic':
                    periodicFlag = True
                if self.boundaryEntity[itr] == 'wall':
                    wallFlag = True
            if self.phase is True:
                if self.boundaryTypePhase[itr] == 'periodic':
                    periodicFlag = True
            if self.T is True:
                if self.boundaryTypeT[itr] == 'periodic':
                    periodicFlag = True
            args = (mesh.Nx_global, mesh.Ny_global, lattice.invList,
                    lattice.noOfDirections, self.boundaryIndices[itr],
                    fields.boundaryNode, solid)
            tempFaceList, tempOutDirections, tempInvDirections,\
                tempNbList, tempCornerList = \
                markBoundaryElements(*args, periodicFlag=periodicFlag,
                                     wall=wallFlag)
            self.faceList.append(tempFaceList)
            self.nbList.append(tempNbList)
            self.cornerList.append(tempCornerList)
            self.invDirections.append(tempInvDirections)
            self.outDirections.append(tempOutDirections)
            if self.boundaryTypeFluid[itr] == 'variableU':
                boundaryVectorTemp = []
                for ind in tempNbList:
                    x, y = int(ind / mesh.Ny_global), int(ind % mesh.Ny_global)
                    value = np.array(self.variableU_func(x, y),
                                     dtype=precision)
                    boundaryVectorTemp.append(value)
                for ind in tempCornerList:
                    x, y = int(ind / mesh.Ny_global), int(ind % mesh.Ny_global)
                    value = np.array(self.variableU_func(x, y),
                                     dtype=precision)
                    boundaryVectorTemp.append(value)
                self.boundaryVectorFluid[itr] = np.array(boundaryVectorTemp,
                                                         dtype=precision)
            if self.fluid is True:
                if self.boundaryTypeFluid[itr] != 'variableU':
                    initializeBoundaryElementsFluid(self.boundaryScalarFluid
                                                    [itr], self.
                                                    boundaryVectorFluid
                                                    [itr], fields.rho,
                                                    fields.u, tempNbList,
                                                    tempCornerList)
                else:
                    initializeVariableElementsFluid(self.boundaryScalarFluid
                                                    [itr], self.
                                                    boundaryVectorFluid
                                                    [itr], fields.rho,
                                                    fields.u, tempNbList,
                                                    tempCornerList)
            if self.phase is True:
                initializeBoundaryElementsScalar(self.boundaryScalarPhase[itr],
                                                 fields.phi, tempNbList,
                                                 tempCornerList)
            if self.T is True:
                initializeBoundaryElementsScalar(self.boundaryScalarT[itr],
                                                 fields.T, tempNbList,
                                                 tempCornerList)

    def setBoundaryArgs(self, lattice, mesh, fields, collisionScheme):
        for itr in range(self.noOfBoundaries):
            if self.fluid is True:
                if self.boundaryTypeFluid[itr] == 'fixedU':
                    args = [fields.f, fields.f_new, fields.rho, fields.u,
                            fields.solid, self.faceList[itr],
                            self.outDirections[itr], self.invDirections[itr],
                            lattice.c, lattice.w, lattice.cs,
                            mesh.Nx, mesh.Ny]
                    self.argsFluid.append(args)
                elif self.boundaryTypeFluid[itr] == 'variableU':
                    args = [fields.f, fields.f_new, fields.rho, fields.u,
                            fields.solid, self.faceList[itr],
                            self.outDirections[itr], self.invDirections[itr],
                            lattice.c, lattice.w, lattice.cs,
                            mesh.Nx, mesh.Ny]
                    self.argsFluid.append(args)
                elif self.boundaryTypeFluid[itr] == 'fixedPressure':
                    args = [fields.f, fields.f_new, fields.rho, fields.u,
                            fields.solid, self.faceList[itr],
                            self.outDirections[itr], self.invDirections[itr],
                            self.nbList[itr], lattice.c, lattice.w, lattice.cs,
                            mesh.Nx, mesh.Ny]
                    self.argsFluid.append(args)
                elif self.boundaryTypeFluid[itr] == 'bounceBack':
                    args = [fields.f, fields.f_new, fields.solid,
                            self.faceList[itr], self.outDirections[itr],
                            self.invDirections[itr]]
                    self.argsFluid.append(args)
                elif self.boundaryTypeFluid[itr] == 'zeroGradient':
                    args = [fields.f, fields.f_new, fields.solid,
                            self.faceList[itr], self.outDirections[itr],
                            self.invDirections[itr], lattice.c, mesh.Nx,
                            mesh.Ny]
                    self.argsFluid.append(args)
                else:
                    args = ()
                    self.argsFluid.append(args)
            if self.phase is True:
                if self.boundaryTypePhase[itr] == 'fixedValue':
                    args = [fields.h_new, fields.u, fields.phi,
                            fields.solid, self.faceList[itr],
                            self.nbList[itr],
                            collisionScheme.equilibriumFuncPhase,
                            collisionScheme.equilibriumArgsPhase]
                    self.argsPhase.append(args)
                elif self.boundaryTypePhase[itr] == 'zeroGradient':
                    args = [fields.h, fields.h_new, fields.solid,
                            self.faceList[itr], self.outDirections[itr],
                            self.invDirections[itr], lattice.c, mesh.Nx,
                            mesh.Ny]
                    self.argsPhase.append(args)
                elif self.boundaryTypePhase[itr] == 'bounceBack':
                    args = [fields.h, fields.h_new, fields.solid,
                            self.faceList[itr], self.outDirections[itr],
                            self.invDirections[itr]]
                    self.argsPhase.append(args)
                else:
                    args = ()
                    self.argsPhase.append(args)

    def details(self):
        print(self.nameList)
        print(self.boundaryTypePhase, self.boundaryTypeFluid)
        print(self.boundaryEntity)
        print(self.boundaryVectorFluid)
        print(self.boundaryScalarPhase, self.boundaryScalarFluid)
        print(self.boundaryIndices)
        print(self.points)
        print(self.faceList)
        print(self.cornerList)
        print(self.outDirections)
        print(self.invDirections)
        print(self.boundaryFuncPhase, self.boundaryFuncFluid)
        print(self.fluid)
        print(self.argsFluid)
        # print(self.argsFluid[2])

    def setupBoundary_cpu(self, parallel):
        for itr in range(self.noOfBoundaries):
            if self.fluid is True:
                self.boundaryFuncFluid[itr] = \
                    numba.njit(self.boundaryFuncFluid[itr], parallel=parallel,
                               cache=False, nogil=True)
            if self.phase is True:
                self.boundaryFuncPhase[itr] = \
                    numba.njit(self.boundaryFuncPhase[itr], parallel=parallel,
                               cache=False, nogil=True)
            if self.T is True:
                self.boundaryFuncT[itr] = \
                    numba.njit(self.boundaryFuncT[itr], parallel=parallel,
                               cache=False, nogil=True)

    def setBoundary(self, fields, lattice, mesh, equilibriumFunc=None,
                    equilibriumArgs=None, initialPhase=False):
        for itr in range(self.noOfBoundaries):
            if self.fluid is True and initialPhase is False:
                self.boundaryFuncFluid[itr](*self.argsFluid[itr])
            if self.phase is True:
                if (initialPhase is True and
                        self.boundaryTypePhase[itr] == 'fixedValue'):
                    args = self.argsPhase[itr]
                    args[1] = fields.u_initialize
                    self.boundaryFuncPhase[itr](*args)
                else:
                    self.boundaryFuncPhase[itr](*self.argsPhase[itr])
            if self.T is True:
                pass

    def setupBoundary_cuda(self):
        self.boundaryScalar_device = cuda.to_device(
            self.boundaryScalarFluid
        )
        for itr in range(self.noOfBoundaries):
            self.boundaryIndices_device.append(cuda.to_device(
                self.boundaryIndices[itr]
            ))
            self.faceList_device.append(cuda.to_device(
                self.faceList[itr]
            ))
            self.invDirections_device.append(cuda.to_device(
                self.invDirections[itr]
            ))
            self.outDirections_device.append(cuda.to_device(
                self.outDirections[itr]
            ))
            self.boundaryVector_device.append(cuda.to_device(
                self.boundaryVectorFluid[itr]
            ))
            self.boundaryFunc.append(getattr(boundaryConditions_cuda,
                                             self.boundaryType[itr]))

    def setBoundary_cuda(self, n_threads, blocks, device):
        for itr in range(self.noOfBoundaries):
            args = (device.f, device.f_new, device.rho, device.u,
                    self.faceList_device[itr], self.outDirections_device[itr],
                    self.invDirections_device[itr],
                    self.boundaryVector_device[itr],
                    self.boundaryScalar_device[itr], device.c, device.w,
                    device.cs[0], device.Nx[0], device.Ny[0])
            self.boundaryFunc[itr][blocks, n_threads](*args)
