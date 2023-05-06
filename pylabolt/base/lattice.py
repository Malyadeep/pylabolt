import numpy as np


class lattice:
    def __init__(self, lat, precision):
        self.latticeType = lat
        self.cs = precision(1/np.sqrt(3))
        self.deltaX = 1
        self.deltaT = 1
        if lat == 'D2Q9':
            self.c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0],
                               [0, -1], [1, 1], [-1, 1],
                               [-1, -1], [1, -1]], dtype=precision)
            self.w = np.array([4/9, 1/9, 1/9, 1/9, 1/9,
                               1/36, 1/36, 1/36, 1/36], dtype=precision)
            self.invList = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6],
                                    dtype=np.int32)
            self.noOfDirections = np.int32(9)

        elif lat == 'D1Q3':
            self.c = np.array([[0, 0], [1, 0], [-1, 0]], dtype=precision)
            self.w = np.array([2/3, 1/6, 1/6], dtype=precision)
            self.invList = np.array([0, 2, 1], dtype=np.int32)
            self.noOfDirections = np.int32(3)


def createLattice(latticeDict, precision):
    lat = latticeDict['latticeType']
    return lattice(lat, precision)
