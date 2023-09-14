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
def constructMRTOperator(preFactor, forcingPreFactor, nu, nu_bulk, M, M_1,
                         S_q, S_epsilon, cs_2, noOfDirections, precision):
    S_nu = 1. / (nu * cs_2 + 0.5)
    S_bulk = 1. / (cs_2 * (nu / 3 + nu_bulk) + 0.5)
    S = np.array([0, S_bulk, S_epsilon, 0, S_q, 0, S_q,
                 S_nu, S_nu], dtype=precision)
    for k in range(noOfDirections):
        for m in range(noOfDirections):
            colSum = 0
            forceSum = 0
            for n in range(noOfDirections):
                colSum += M_1[k, n] * S[n] * M[n, m]
                forceSum += M_1[k, n] * (1 - 0.5 * S[n]) * M[n, m]
            preFactor[k, m] = colSum
            forcingPreFactor[k, m] = forceSum


@numba.njit
def constructBGKOperator(preFactor, forcingPreFactor, nu, cs_2,
                         noOfDirections):
    preFactorValue = 1. / (nu * cs_2 + 0.5)
    for k in range(noOfDirections):
        preFactor[k, k] = preFactorValue
        forcingPreFactor[k, k] = 1 - 0.5 * preFactorValue
