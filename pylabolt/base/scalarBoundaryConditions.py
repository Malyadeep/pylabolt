from numba import prange
import numpy as np


def fixedValue(h_new, u, phi, solid, faceList, nbList, equilibriumFunc,
               equilibriumArgs):
    for itr in prange(faceList.shape[0]):
        ind = faceList[itr]
        if solid[ind, 0] != 1:
            equilibriumFunc(h_new[ind, :], u[ind, :],
                            phi[nbList[itr]], *equilibriumArgs)


def zeroGradient(h, h_new, solid, faceList, outDirections, invDirections,
                 c, Nx, Ny):
    for itr in prange(faceList.shape[0]):
        ind = faceList[itr]
        if solid[ind, 0] != 1:
            i, j = int(ind / Ny), int(ind % Ny)
            for direction in invDirections:
                c_mag = int(np.ceil((c[direction, 0] * c[direction, 0]
                            + c[direction, 1] * c[direction, 1])))
                if c_mag == 1:
                    i_nb = int(i + c[direction, 0])
                    j_nb = int(j + c[direction, 1])
                    ind_nb = int(i_nb * Ny + j_nb)
                    break
            for k in range(h.shape[1]):
                h_new[ind, k] = h_new[ind_nb, k]


def bounceBack(h, h_new, solid, faceList, outDirections, invDirections):
    for itr in prange(faceList.shape[0]):
        ind = faceList[itr]
        if solid[ind, 0] != 1:
            for direction in range(invDirections.shape[0]):
                h_new[ind, invDirections[direction]] = \
                    h[ind, outDirections[direction]]


def periodic():
    pass
