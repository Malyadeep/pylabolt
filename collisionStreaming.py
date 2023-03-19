import numpy as np
import numba


class collisionStreaming:
    def __init__(self, deltaX, deltaT, c, tau, f):
        self.omega = 1/tau
        self.deltaX = deltaX
        self.deltaT = deltaT
        self.dtdx = deltaT/deltaX
        self.c = c
        self.tau = tau
        self.f_new = np.copy(f)

    @numba.njit
    def BGK(self, f, f_new, f_eq, nodeType):
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                if nodeType[i, j] != 's':
                    for k in range(f.shape[2]):
                        f[i, j, k] = f_new[i, j, k] + self.deltaT * \
                            self.omega * (f_eq[i, j, k] - f[i, j, k])

    @numba.njit
    def stream(self, f, f_new, nodeType):
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                if nodeType[i, j] != 's':
                    for k in range(1, f.shape[2]):
                        i_old = (i - int(self.c[k, 0]*self.dtdx)
                                 + f.shape[0]) % f.shape[0]
                        j_old = (j - int(self.c[k, 1]*self.dtdx)
                                 + f.shape[1]) % f.shape[1]
                        f_new[i, j, k] = f[i_old, j_old, k]


if __name__ == '__main__':
    print('module implementing various collision and streaming steps')
