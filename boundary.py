import numpy as np
import os
import boundaryConditions
import numba
from numba.typed import List


@numba.njit
def getDirections(i, j, c, Nx, Ny, invList):
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
            elementInv.append(invList[k])
    return np.array(elementOut, dtype=np.int32),\
        np.array(elementInv, dtype=np.int32)


@numba.njit
def initializeBoundaryElements(elements, mesh, lattice,
                               boundaryType, boundaryIndices,
                               boundaryVector, boundaryScalar):
    Nx = mesh.Nx
    Ny = mesh.Ny
    c = lattice.c
    invList = lattice.invList
    cs_2 = 1/(lattice.cs*lattice.cs)
    elementList = []
    indX_i, indX_f = boundaryIndices[0, 0], boundaryIndices[1, 0]
    indY_i, indY_f = boundaryIndices[0, 1], boundaryIndices[1, 1]
    tempList = []
    for ind in range(Nx * Ny):
        for i in range(indX_i, indX_f):
            for j in range(indY_i, indY_f):
                if indY_f - indY_i == 1:
                    currentId = int(i * Ny + j)
                elif indX_f - indX_i == 1:
                    currentId = int(i + j * Nx)
                if elements[ind].id == currentId:
                    tempList.append(elements[ind].id)
                    elements[ind].nodeType = 'b'
                    if boundaryType == 'fixedU':
                        elements[ind].u = boundaryVector
                    elif boundaryType == 'fixedPressure':
                        elements[ind].rho = boundaryScalar * cs_2
                    args = (i, j, c, Nx, Ny, invList)
                    tempOut, tempInv = getDirections(*args)
                    args = (tempOut, tempInv)
                    elements[ind].setDirections(*args)
    elementList.append(np.array(tempList))
    return elementList[0]


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
        self.elementList = []

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
            args = (List(elements), mesh, lattice,
                    self.boundaryType[itr], self.boundaryIndices[itr],
                    self.boundaryVector[itr], self.boundaryScalar[itr])
            tempList = initializeBoundaryElements(*args)
            self.elementList.append(tempList)

    def details(self):
        print(self.nameList)
        print(self.boundaryType)
        print(self.boundaryVector)
        print(self.boundaryScalar)
        print(self.boundaryIndices)
        print(self.points)
        print(self.elementList)

    def setBoundary(self, elements, lattice, mesh):
        for itr in range(self.noOfBoundaries):
            args = (elements, self.elementList[itr],
                    self.boundaryVector[itr],
                    self.boundaryVector[itr], lattice, mesh)
            self.boundaryFunc[itr](*args)
