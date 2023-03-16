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
    def BGK(self, f, f_eq, deltaT, deltaX, c):
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                for k in range(f.shape[2]):
                    i_new = (i + int(self.c[k, 0]*self.dtdx)
                             + f.shape[0]) % f.shape[0]
                    j_new = (j + int(self.c[k, 1]*self.dtdx)
                             + f.shape[1]) % f.shape[1]
                    self.f_new[i_new, j_new, k] = f[i, j, k] + self.deltaT * \
                        self.omega * (f_eq[i, j, k] - f[i, j, k])
        f[:] = self.f_new


if __name__ == '__main__':
    print('module implementing various collision and streaming steps')
