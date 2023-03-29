import numpy as np
import os
import boundaryConditions
import numba


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
                               boundaryValues):
    dtdx = lattice.dtdx
    Nx = mesh.Nx
    Ny = mesh.Ny
    c = lattice.c
    invList = lattice.invList
    cs_2 = 1/(lattice.cs*lattice.cs)
    elementList = []
    for element in elements:
        tempList = []
        for i in range(boundaryIndices[0, 0],
                       boundaryIndices[1, 0]):
            for j in range(boundaryIndices[0, 1],
                           boundaryIndices[1, 1]):
                currentId = int(i * Nx + j)
                if element.id == currentId:
                    tempList.append(element.id)
                    element.nodeType = 'b'
                    if boundaryType == 'fixedU':
                        element.u = boundaryValues[0]
                        element.v = boundaryValues[1]
                    elif boundaryType == 'fixedPressure':
                        element.rho = boundaryValues * cs_2
                    args = (i, j, c * dtdx, Nx, Ny, invList)
                    tempOut, tempInv = getDirections(*args)
                    args = (tempOut, tempInv)
                    element.setDirections(*args)
        elementList.append(np.array(tempList))
    return tempList


class boundary:
    def __init__(self, boundaryDict):
        self.boundaryDict = boundaryDict
        self.boundaryValues = []
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
                            value = np.array(self.boundaryDict[item]['value'],
                                             dtype=np.float64)
                        else:
                            print("ERROR!")
                            print("For 'fixedU' value must be a list of"
                                  + " components: [x1, x2]")
                            os._exit(1)
                    elif self.boundaryDict[item][data] == 'fixedPressure':
                        try:
                            if isinstance(self.boundaryDict[item]['value'],
                                          float):
                                value = np.float64(self.boundaryDict[item]
                                                   ['value'])
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
                        value = np.float64(0.0)
                    elif self.boundaryDict[item][data] == 'periodic':
                        value = np.float64(0.0)
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
                    self.boundaryValues.append(value)
                self.points[item] = tempPoints
                if flag is False:
                    print("ERROR! 'type' keyword not defined")
                    os._exit(0)
        self.noOfBoundaries = len(self.boundaryFunc)

    def initializeBoundary(self, lattice, mesh, elements):
        for name in self.nameList:
            pointArray = np.array(self.points[name])
            for k in range(pointArray.shape[0]):
                tempIndex_i = np.rint(pointArray[k, 0]/lattice.delX).\
                    astype(int)
                tempIndex_f = np.rint(pointArray[k, 1]/lattice.delX).\
                    astype(int) + 1
                self.boundaryIndices.append([tempIndex_i, tempIndex_f])
        self.boundaryIndices = np.array(self.boundaryIndices)
        args = (elements, self.noOfBoundaries, self.boundaryIndices,
                self.boundaryValues, mesh, lattice)
        for itr in range(self.noOfBoundaries):
            args = (elements, mesh, lattice,
                    self.boundaryType[itr], self.boundaryIndices[itr],
                    self.boundaryValues[itr])
            tempList = initializeBoundaryElements(*args)
            self.elementList.append(tempList)

    def details(self):
        print(self.nameList)
        print(self.boundaryType)
        print(self.boundaryValues)
        print(self.boundaryIndices.shape)
        print(self.points)

    def setBoundary(self, elements, lattice, mesh):
        for itr in range(self.noOfBoundaries):
            args = (elements, self.elementList[itr],
                    self.boundaryValues[itr], lattice, mesh)
            self.boundaryFunc[itr](*args)
