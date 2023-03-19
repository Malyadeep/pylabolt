# import numpy as np
# import numba
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
                        self.boundaryFunc.append(boundaryConditions.
                                                 fixedVector)
                        try:
                            self.boundaryDict[item]['value']
                        except Exception:
                            print('Key Error!')
                            print("'value' keyword required for type " +
                                  "'fixedVector'")
                            os._exit(1)
                        if isinstance(self.boundaryDict[item]['value'], list):
                            self.boundaryValues.append(self.boundaryDict[item]
                                                       ['value'])
                        else:
                            print("ERROR!")
                            print("For 'fixedVector' value must be a list of"
                                  + " components: [x1, x2]")
                            os._exit(1)
                    elif self.boundaryDict[item][data] == 'fixedScalar':
                        self.boundaryFunc.append(boundaryConditions.
                                                 fixedScalar)
                        try:
                            if isinstance(self.boundaryDict[item]['value'],
                                          float):
                                self.boundaryValues.append(self.
                                                           boundaryDict[item]
                                                           ['value'])
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
                        self.boundaryFunc.append(boundaryConditions.bounceBack)
                        self.boundaryValues.append(0.0)
                    elif self.boundaryDict[item][data] == 'periodic':
                        self.boundaryFunc.append(boundaryConditions.periodic)
                        self.boundaryValues.append(0.0)
                    else:
                        print("ERROR! " + self.boundaryDict[item][data] +
                              " is not a valid boundary condition!")
                        print('Please check boundary conditions available')
                        print('Refer to the tutorials and documentation')
                        os._exit(0)
                elif data != 'value':
                    tempPoints.append(self.boundaryDict[item][data])
                self.points[item] = tempPoints
                if flag is False:
                    print("ERROR! 'type' keyword not defined")
                    os._exit(0)

    def initializeBoundary(self):
        pass

    def setBoundary(self):
        pass


if __name__ == '__main__':
    print('Module to read and applying boundary conditions')
