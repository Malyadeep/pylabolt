import numpy as np
import numba

spec = [
    ('c', numba.int64[:, :]),
    ('invList', numba.int64),
    ('cs', numba.float64),
    ('latticeType', numba.types.string),
    ('w', numba.float64[:]),
    ('deltaX', numba.float64),
    ('deltaT', numba.float64),
    ('noOfDirections', numba.int64)
]


@numba.experimental.jitclass(spec)
class lattice:
    def __init__(self, lat):
        self.latticeType = lat
        self.cs = 1/np.sqrt(3)
        self.deltaX = 1.0
        self.deltaT = 1.0
        if lat == 'D2Q9':
            self.c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0],
                               [0, -1], [1, 1], [-1, 1],
                               [-1, -1], [1, -1]], dtype=np.int64)
            self.w = np.array([4/9, 1/9, 1/9, 1/9, 1/9,
                               1/36, 1/36, 1/36, 1/36])
            self.invList = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6],
                                    dtype=np.int64)
            self.noOfDirections = 9

        elif lat == 'D1Q3':
            self.c = np.array([[0, 0], [1, 0], [-1, 0]], dtype=np.int64)
            self.w = np.array([2/3, 1/6, 1/6])
            self.invList = np.array([0, 2, 1], dtype=np.int64)
            self.noOfDirections = 3


def createLattice(latticeDict):
    lat = latticeDict['latticeType']
    return lattice(lat)
