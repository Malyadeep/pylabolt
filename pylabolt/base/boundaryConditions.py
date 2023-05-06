from numba import prange
import numpy as np


def fixedU(f, f_new, rho, u, faceList, outDirections, invDirections,
           boundaryVector, boundaryScalar, c, w, cs, Nx, Ny):
    cs_2 = 1/(cs * cs)
    for ind in prange(faceList.shape[0]):
        rhoWall = rho[faceList[ind]]
        for dir in range(invDirections.shape[0]):
            preFactor = 2 * w[outDirections[dir]] * rhoWall *\
                ((c[outDirections[dir], 0] * boundaryVector[0] +
                  c[outDirections[dir], 1] * boundaryVector[1])) * cs_2
            f_new[faceList[ind], invDirections[dir]] = \
                f[faceList[ind], outDirections[dir]] - preFactor


def fixedPressure(f, f_new, rho, u, faceList, outDirections, invDirections,
                  boundaryVector, boundaryScalar, c, w, cs, Nx, Ny):
    cs_2 = 1/(cs * cs)
    cs_4 = cs_2 * cs_2
    for ind in prange(faceList.shape[0]):
        i, j = int(faceList[ind] / Ny), int(faceList[ind] % Ny)
        for direction in invDirections:
            c_mag = int(np.ceil((c[direction, 0] * c[direction, 0]
                        + c[direction, 1] * c[direction, 1])))
            if c_mag == 1:
                i_nb = int(i + c[direction, 0])
                j_nb = int(j + c[direction, 1])
                ind_nb = int(i_nb * Ny + j_nb)
                break
        u_nb = u[faceList[ind], :] + 0.5 * (u[faceList[ind], :] -
                                            u[ind_nb, :])
        u_nb_2 = u_nb[0] * u_nb[0] + u_nb[1] * u_nb[1]
        for dir in range(invDirections.shape[0]):
            c_dot_u = (c[outDirections[dir], 0] * u_nb[0] +
                       c[outDirections[dir], 1] * u_nb[1])
            preFactor = 2 * w[outDirections[dir]] * boundaryScalar * cs_2 *\
                (1. + c_dot_u * c_dot_u * 0.5 * cs_4 - u_nb_2 * 0.5 * cs_2)
            f_new[faceList[ind], invDirections[dir]] = \
                - f[faceList[ind], outDirections[dir]] + preFactor


def bounceBack(f, f_new, rho, u, faceList, outDirections, invDirections,
               boundaryVector, boundaryScalar, c, w, cs, Nx, Ny):
    for ind in prange(faceList.shape[0]):
        for dir in range(invDirections.shape[0]):
            f_new[faceList[ind], invDirections[dir]] = \
                f[faceList[ind], outDirections[dir]]


def zeroGradient(f, f_new, rho, u, faceList, outDirections, invDirections,
                 boundaryVector, boundaryScalar, c, w, cs, Nx, Ny):
    for ind in prange(faceList.shape[0]):
        i, j = int(faceList[ind] / Ny), int(faceList[ind] % Ny)
        for direction in invDirections:
            c_mag = int(np.ceil((c[direction, 0] * c[direction, 0]
                        + c[direction, 1] * c[direction, 1])))
            if c_mag == 1:
                i_nb = int(i + c[direction, 0])
                j_nb = int(j + c[direction, 1])
                ind_nb = int(i_nb * Ny + j_nb)
                break
        for k in range(f.shape[1]):
            f[faceList[ind], k] = f[ind_nb, k]


def periodic(f, f_new, rho, u, faceList, outDirections, invDirections,
             boundaryVector, boundaryScalar, c, w, cs, Nx, Ny):
    pass
