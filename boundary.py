import numpy as np
import os
import boundaryConditions
import numba
from numba.typed import List


@numba.njit
def getDirections(i, j, c, Nx, Ny, invDirections):
    elementOut = []
    elementInv = []
    for k in range(1, c.shape[0]):
        i_next = (i + int(c[k, 0]))
        j_next = (j + int(c[k, 1]))
        condEnd = (np.array([i_next, j_next]) == np.array([Nx, Ny]))
        condStart = (np.array([i_next, j_next]) < np.array([0, 0]))
        if (condStart[0] or condStart[1] or
                condEnd[0] or condEnd[1]):
            elementOut.append(k)
            elementInv.append(invDirections[k])
    return np.array(elementOut, dtype=np.int32),\
        np.array(elementInv, dtype=np.int32)


@numba.njit
def initializeBoundaryElements(fields, mesh, lattice,
                               boundaryType, boundaryIndices,
                               boundaryVector, boundaryScalar):
    Nx = mesh.Nx
    Ny = mesh.Ny
    c = lattice.c
    inv = lattice.invList
    cs_2 = 1/(lattice.cs*lattice.cs)
    faceList = []
    invDirections = []
    outDirections = []
    indX_i, indX_f = boundaryIndices[0, 0], boundaryIndices[1, 0]
    indY_i, indY_f = boundaryIndices[0, 1], boundaryIndices[1, 1]
    for ind in range(Nx * Ny):
        for i in range(indX_i, indX_f):
            for j in range(indY_i, indY_f):
                currentId = int(i * Ny + j)
                if currentId == ind:
                    args = (i, j, c, Nx, Ny, inv)
                    tempOut, tempInv = getDirections(*args)
                    if boundaryType == 'fixedU':
                        fields.u[ind, 0] = boundaryVector[0]
                        fields.u[ind, 1] = boundaryVector[1]
                    elif boundaryType == 'fixedPressure':
                        fields.rho[ind] = boundaryScalar * cs_2
                    outDirections.append(tempOut)
                    invDirections.append(tempInv)
                    faceList.append(ind)
    return np.array(faceList, dtype=np.int32),\
        List(outDirections),\
        List(invDirections)


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
                    self.boundaryFunc.append(getattr(boundaryConditions,
                                                     self.boundaryDict[item]
                                                     ['type']))
                    self.boundaryVector.append(vectorValue)
                    self.boundaryScalar.append(scalarValue)
                self.points[item] = tempPoints
                if flag is False:
                    print("ERROR! 'type' keyword not defined")
                    os._exit(0)
        self.noOfBoundaries = len(self.boundaryFunc)

    def initializeBoundary(self, lattice, mesh, elements):
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
            args = (elements, mesh, lattice,
                    self.boundaryType[itr], self.boundaryIndices[itr],
                    self.boundaryVector[itr], self.boundaryScalar[itr])
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

    def setBoundary(self, fields, lattice, mesh):
        for itr in range(self.noOfBoundaries):
            args = (fields, self.faceList[itr], self.outDirections[itr],
                    self.invDirections[itr], self.boundaryVector[itr],
                    self.boundaryScalar[itr], lattice, mesh)
            self.boundaryFunc[itr](*args)
