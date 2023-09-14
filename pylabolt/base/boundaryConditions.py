from numba import prange
import numpy as np


def fixedU(f, f_new, rho, u, solid, faceList, outDirections, invDirections,
           c, w, cs, Nx, Ny):
    cs_2 = 1/(cs * cs)
    for itr in prange(faceList.shape[0]):
        ind = faceList[itr]
        if solid[ind, 0] != 1:
            rhoWall = rho[ind]
            for direction in range(invDirections.shape[0]):
                i, j = int(ind / Ny), int(ind % Ny)
                i_nb = int(i + c[outDirections[direction], 0])
                j_nb = int(j + c[outDirections[direction], 1])
                ind_nb = int(i_nb * Ny + j_nb)
                preFactor = 2 * w[outDirections[direction]] * rhoWall *\
                    ((c[outDirections[direction], 0] * u[ind_nb, 0] +
                     c[outDirections[direction], 1] * u[ind_nb, 1])) * cs_2
                f_new[ind, invDirections[direction]] = \
                    f[ind, outDirections[direction]] - preFactor


def variableU(f, f_new, rho, u, solid, faceList, outDirections, invDirections,
              c, w, cs, Nx, Ny):
    cs_2 = 1/(cs * cs)
    for itr in prange(faceList.shape[0]):
        ind = faceList[itr]
        if solid[ind, 0] != 1:
            rhoWall = rho[ind]
            for direction in range(invDirections.shape[0]):
                i_nb = int(int(ind / Ny) + c[outDirections[direction], 0])
                j_nb = int(int(ind % Ny) + c[outDirections[direction], 1])
                ind_nb = int(i_nb * Ny + j_nb)
                preFactor = 2 * w[outDirections[direction]] * rhoWall *\
                    ((c[outDirections[direction], 0] * u[ind_nb, 0] +
                     c[outDirections[direction], 1] * u[ind_nb, 1])) * cs_2
                f_new[ind, invDirections[direction]] = \
                    f[ind, outDirections[direction]] - preFactor


def fixedPressure(f, f_new, rho, u, solid, faceList, outDirections,
                  invDirections, nbList, c, w, cs, Nx, Ny):
    cs_2 = 1/(cs * cs)
    cs_4 = cs_2 * cs_2
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
            u_nb = u[ind, :] + 0.5 * (u[ind, :] - u[ind_nb, :])
            u_nb_2 = u_nb[0] * u_nb[0] + u_nb[1] * u_nb[1]
            for direction in range(invDirections.shape[0]):
                c_dot_u = (c[outDirections[direction], 0] * u_nb[0] +
                           c[outDirections[direction], 1] * u_nb[1])
                preFactor = 2 * w[outDirections[direction]] *\
                    rho[nbList[itr]] * (1. + c_dot_u * c_dot_u * 0.5 * cs_4
                                        - u_nb_2 * 0.5 * cs_2)
                f_new[ind, invDirections[direction]] = \
                    - f[ind, outDirections[direction]] + preFactor


def bounceBack(f, f_new, solid, faceList, outDirections, invDirections):
    for ind in prange(faceList.shape[0]):
        if solid[faceList[ind], 0] != 1:
            for direction in range(invDirections.shape[0]):
                f_new[faceList[ind], invDirections[direction]] = \
                    f[faceList[ind], outDirections[direction]]


def zeroGradient(f, f_new, solid, faceList, outDirections, invDirections,
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
            for k in range(f.shape[1]):
                f_new[ind, k] = f_new[ind_nb, k]


def periodic():
    pass
