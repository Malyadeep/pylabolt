import numpy as np
import os
import numba
from numba import cuda

from pylabolt.base import boundaryConditions
from pylabolt.base.cuda import boundaryConditions_cuda


@numba.njit
def initializeBoundaryElements(Nx, Ny, invList, noOfDirections,
                               boundaryIndices):
    faceList = []
    indX_i, indX_f = boundaryIndices[0, 0], boundaryIndices[1, 0]
    indY_i, indY_f = boundaryIndices[0, 1], boundaryIndices[1, 1]
    diffX = indX_f - indX_i
    diffY = indY_f - indY_i
    if diffY == 1:
        if indY_i == 0:
            if noOfDirections == 9:
                outDirections = np.array([7, 4, 8], dtype=np.int32)
                invDirections = np.array([invList[7], invList[4], invList[8]],
                                         dtype=np.int32)
        elif indY_i == Ny - 1:
            if noOfDirections == 9:
                outDirections = np.array([6, 2, 5], dtype=np.int32)
                invDirections = np.array([invList[6], invList[2], invList[5]],
                                         dtype=np.int32)
    if diffX == 1:
        if indX_i == 0:
            if noOfDirections == 9:
                outDirections = np.array([6, 3, 7], dtype=np.int32)
                invDirections = np.array([invList[6], invList[3], invList[7]],
                                         dtype=np.int32)
        elif indX_i == Nx - 1:
            if noOfDirections == 9:
                outDirections = np.array([1, 5, 8], dtype=np.int32)
                invDirections = np.array([invList[1], invList[5], invList[8]],
                                         dtype=np.int32)
    for ind in range(Nx * Ny):
        for i in range(indX_i, indX_f):
            for j in range(indY_i, indY_f):
                currentId = int(i * Ny + j)
                if currentId == ind:
                    faceList.append(ind)
    return np.array(faceList, dtype=np.int64),\
        outDirections,\
        invDirections


class boundary:
    def __init__(self, boundaryDict):
        self.boundaryDict = boundaryDict
        self.boundaryVector = []
        self.boundaryScalar = []
        self.boundaryType = []
        self.boundaryFunc = []
        self.points = {}
        self.nameList = list(boundaryDict.keys())
        self.boundaryIndices = []
        self.noOfBoundaries = 0
        self.faceList = []
        self.invDirections = []
        self.outDirections = []

        # Cuda device data
        self.boundaryVector_device = []
        self.boundaryScalar_device = []
        self.boundaryIndices_device = []
        self.faceList_device = []
        self.invDirections_device = []
        self.outDirections_device = []
        self.noOfBoundaries_device = np.zeros(1, np.int32)

    def readBoundaryDict(self):
        for item in self.nameList:
            dataList = list(self.boundaryDict[item].keys())
            tempPoints = []
            flag = False
            for data in dataList:
                if data == 'type':
                    flag = True
                    if self.boundaryDict[item][data] == 'fixedU':
                        try:
                            self.boundaryDict[item]['value']
                        except Exception:
                            print('Key Error!')
                            print("'value' keyword required for type " +
                                  "'fixedU'")
                            os._exit(1)
                        if isinstance(self.boundaryDict[item]['value'], list):
                            vectorValue = np.array(self.boundaryDict[item]
                                                   ['value'], dtype=np.float64)
                            scalarValue = 0.0
                        else:
                            print("ERROR!")
                            print("For 'fixedU' value must be a list of"
                                  + " components: [x1, x2]")
                            os._exit(1)
                    elif self.boundaryDict[item][data] == 'fixedPressure':
                        try:
                            if isinstance(self.boundaryDict[item]['value'],
                                          float):
                                vectorValue = np.zeros(2, dtype=np.float64)
                                scalarValue = np.float64(self.boundaryDict
                                                         [item]['value'])
                            else:
                                print("ERROR!")
                                print("For 'fixedPressure' value must be a " +
                                      "float")
                                os._exit(0)
                        except Exception:
                            print("ERROR!")
                            print("'value' keyword required for type " +
                                  "'fixedPressure'")
                            os._exit(0)
                    elif self.boundaryDict[item][data] == 'bounceBack':
                        vectorValue = np.zeros(2, dtype=np.float64)
                        scalarValue = np.float64(0.0)
                    elif self.boundaryDict[item][data] == 'periodic':
                        vectorValue = np.zeros(2, dtype=np.float64)
                        scalarValue = np.float64(0.0)
                    elif self.boundaryDict[item][data] == 'zeroGradient':
                        vectorValue = np.zeros(2, dtype=np.float64)
                        scalarValue = np.float64(0.0)
                    else:
                        print("ERROR! " + self.boundaryDict[item][data] +
                              " is not a valid boundary condition!")
                        print('Please check boundary conditions available')
                        print('Refer to the tutorials and documentation')
                        os._exit(0)
                elif data != 'value':
                    tempPoints.append(self.boundaryDict[item][data])
                    self.boundaryType.append(self.boundaryDict[item]
                                             ['type'])
                    self.boundaryVector.append(vectorValue)
                    self.boundaryScalar.append(scalarValue)
                self.points[item] = tempPoints
                if flag is False:
                    print("ERROR! 'type' keyword not defined")
                    os._exit(0)
        self.noOfBoundaries = len(self.boundaryType)

    def initializeBoundary(self, lattice, mesh, fields):
        for name in self.nameList:
            pointArray = np.array(self.points[name])
            for k in range(pointArray.shape[0]):
                tempIndex_i = np.rint(pointArray[k, 0]/mesh.delX).\
                    astype(int)
                tempIndex_f = np.rint(pointArray[k, 1]/mesh.delX).\
                    astype(int) + 1
                self.boundaryIndices.append([tempIndex_i, tempIndex_f])
        self.boundaryIndices = np.array(self.boundaryIndices)
        for itr in range(self.noOfBoundaries):
            args = (mesh.Nx_global, mesh.Ny_global, lattice.invList,
                    lattice.noOfDirections, self.boundaryIndices[itr])
            tempFaceList, tempOutDirections, tempInvDirections = \
                initializeBoundaryElements(*args)
            self.faceList.append(tempFaceList)
            self.invDirections.append(tempInvDirections)
            self.outDirections.append(tempOutDirections)

    def details(self):
        print(self.nameList)
        print(self.boundaryType)
        print(self.boundaryVector)
        print(self.boundaryScalar)
        print(self.boundaryIndices)
        print(self.points)
        print(self.faceList)
        print(self.outDirections)
        print(self.invDirections)
        print(self.boundaryFunc)

    def setupBoundary_cpu(self, parallel):
        for itr in range(self.noOfBoundaries):
            self.boundaryFunc.append(getattr(boundaryConditions,
                                             self.boundaryType[itr]))
            self.boundaryFunc[itr] = numba.njit(self.boundaryFunc[itr],
                                                parallel=parallel,
                                                cache=False,
                                                nogil=True)

    def setBoundary(self, fields, lattice, mesh):
        for itr in range(self.noOfBoundaries):
            args = (fields.f, fields.f_new, fields.rho, fields.u,
                    self.faceList[itr], self.outDirections[itr],
                    self.invDirections[itr], self.boundaryVector[itr],
                    self.boundaryScalar[itr], lattice.c, lattice.w,
                    lattice.cs, mesh.Nx, mesh.Ny)
            self.boundaryFunc[itr](*args)

    def setupBoundary_cuda(self):
        self.boundaryScalar_device = cuda.to_device(
            self.boundaryScalar
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
                self.boundaryVector[itr]
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
