import numpy as np
import os
import boundaryConditions


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

    def readBoundaryDict(self):
        for item in self.nameList:
            dataList = list(self.boundaryDict[item].keys())
            tempPoints = []
            flag = False
            for data in dataList:
                if data == 'type':
                    flag = True
                    self.boundaryType.append(self.boundaryDict[item][data])
                    if self.boundaryDict[item][data] == 'fixedVector':
                        try:
                            self.boundaryDict[item]['value']
                        except Exception:
                            print('Key Error!')
                            print("'value' keyword required for type " +
                                  "'fixedVector'")
                            os._exit(1)
                        if isinstance(self.boundaryDict[item]['value'], list):
                            value = np.array(self.boundaryDict[item]['value'])
                        else:
                            print("ERROR!")
                            print("For 'fixedVector' value must be a list of"
                                  + " components: [x1, x2]")
                            os._exit(1)
                    elif self.boundaryDict[item][data] == 'fixedScalar':
                        try:
                            if isinstance(self.boundaryDict[item]['value'],
                                          float):
                                value = self.boundaryDict[item]['value']
                            else:
                                print("ERROR!")
                                print("For 'fixedScalar' value must be a " +
                                      "float")
                                os._exit(0)
                        except Exception:
                            print("ERROR!")
                            print("'value' keyword required for type " +
                                  "'fixedScalar'")
                            os._exit(0)
                    elif self.boundaryDict[item][data] == 'bounceBack':
                        value = np.float(0.0)
                    elif self.boundaryDict[item][data] == 'periodic':
                        value = np.float(0.0)
                    else:
                        print("ERROR! " + self.boundaryDict[item][data] +
                              " is not a valid boundary condition!")
                        print('Please check boundary conditions available')
                        print('Refer to the tutorials and documentation')
                        os._exit(0)
                elif data != 'value':
                    tempPoints.append(self.boundaryDict[item][data])
                    self.boundaryFunc.append(getattr(boundaryConditions,
                                                     self.boundaryDict[item]
                                                     ['type']))
                    self.boundaryValues.append(value)
                self.points[item] = tempPoints
                if flag is False:
                    print("ERROR! 'type' keyword not defined")
                    os._exit(0)
        self.noOfBoundaries = len(self.boundaryFunc)

    def calculateBoundaryIndices(self, delX):
        for name in self.nameList:
            pointArray = np.array(self.points[name])
            for k in range(pointArray.shape[0]):
                tempIndex_i = np.rint(pointArray[k, 0]/delX).astype(int)
                tempIndex_f = np.rint(pointArray[k, 1]/delX).\
                    astype(int) + 1
                self.boundaryIndices.append([tempIndex_i, tempIndex_f])
        self.boundaryIndices = np.array(self.boundaryIndices)

    def details(self):
        print(self.nameList)
        print(self.boundaryType)
        print(self.boundaryValues)
        print(self.boundaryIndices.shape)
        print(self.points)

    def initializeBoundaryElements(self, elements):
        for itr in range(self.noOfBoundaries):
            self.boundaryFunc[itr](elements, self.boundaryValues[itr],
                                   self.boundaryIndices[itr], True)

    def setBoundary(self, elements):
        for itr in range(self.noOfBoundaries):
            self.boundaryFunc[itr](elements, self.boundaryValues[itr],
                                   self.boundaryIndices[itr], False)
