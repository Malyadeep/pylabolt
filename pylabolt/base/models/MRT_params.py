import numpy as np
import numba


class MRT_def:
    def __init__(self, precision, S_nu, S_bulk, S_q, S_epsilon):
        self.M = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [-4, -1, -1, -1, -1, 2, 2, 2, 2],
                           [4, -2, -2, -2, -2, 1, 1, 1, 1],
                           [0, 1, 0, -1, 0, 1, -1, -1, 1],
                           [0, -2, 0, 2, 0, 1, -1, -1, 1],
                           [0, 0, 1, 0, -1, 1, 1, -1, -1],
                           [0, 0, -2, 0, 2, 1, 1, -1, -1],
                           [0, 1, -1, 1, -1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, -1, 1, -1]], dtype=precision)
        self.M_1 = np.linalg.inv(self.M)
        self.diagVec = np.array([0, S_bulk, S_epsilon, 0, S_q, 0, S_q,
                                 S_nu, S_nu], dtype=precision)
        self.S = np.diag(self.diagVec, k=0)
        self.preFactorMat = np.matmul(np.matmul(self.M_1, self.S), self.M)

    def setForcingGuo(self):
        diagVecForce = 1 - self.diagVec
        self.S_guo = np.diag(diagVecForce, k=0)
        self.forcingPreFactorMat = np.matmul(np.matmul(self.M_1, self.S_guo),
                                             self.M)
        return self.forcingPreFactorMat


@numba.njit
def computeMRT():
    pass
