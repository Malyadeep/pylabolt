import numba
import numpy as np


@numba.njit
def fixedU(f, f_new, rho, u, faceList, outDirections, invDirections,
           boundaryVector, boundaryScalar, c, w, cs, Nx, Ny):
    cs_2 = 1/(cs * cs)
    for itr, ind in enumerate(faceList):
        invDir = invDirections[itr]
        outDir = outDirections[itr]
        rhoWall = rho[ind]
        for dir in range(invDir.shape[0]):
            preFactor = 2 * w[outDir[dir]] * rhoWall *\
                ((c[outDir[dir], 0] * boundaryVector[0] +
                  c[outDir[dir], 1] * boundaryVector[1])) * cs_2
            f_new[ind, invDir[dir]] = \
                f[ind, outDir[dir]] - preFactor


@numba.njit
def fixedPressure(f, f_new, rho, u, faceList, outDirections, invDirections,
                  boundaryVector, boundaryScalar, c, w, cs, Nx, Ny):
    cs_2 = 1/(cs * cs)
    cs_4 = cs_2 * cs_2
    for itr, ind in enumerate(faceList):
        i, j = int(ind / Ny), int(ind % Ny)
        # print(ind, i, j)
        invDir = invDirections[itr]
        outDir = outDirections[itr]
        for direction in invDir:
            # print(direction)
            c_mag = int(np.ceil((c[direction, 0] * c[direction, 0]
                        + c[direction, 1] * c[direction, 1])))
            # print(direction, c_mag)
            if c_mag == 1:
                i_nb = int(i + c[direction, 0])
                j_nb = int(j + c[direction, 1])
                # print(i_nb, j_nb)
                ind_nb = int(i_nb * Nx + j_nb)
                # print(i_nb, j_nb, ind_nb)
                break
        u_nb = u[ind, :] + 0.5 * (u[ind, :] -
                                  u[ind_nb, :])
        # print(u_nb)
        u_nb_2 = u_nb[0] * u_nb[0] + u_nb[1] * u_nb[1]
        for dir in range(invDir.shape[0]):
            c_dot_u = (c[outDir[dir], 0] * u_nb[0] +
                       c[outDir[dir], 1] * u_nb[1])
            preFactor = 2 * w[outDir[dir]] * rho[ind] *\
                (1. + c_dot_u * c_dot_u * 0.5 * cs_4 - u_nb_2 * 0.5 * cs_2)
            f_new[ind, invDir[dir]] = \
                - f[ind, outDir[dir]] + preFactor


@numba.njit
def bounceBack(f, f_new, rho, u, faceList, outDirections, invDirections,
               boundaryVector, boundaryScalar, c, w, cs, Nx, Ny):
    for itr, ind in enumerate(faceList):
        invDir = invDirections[itr]
        outDir = outDirections[itr]
        for dir in range(invDir.shape[0]):
            f_new[ind, invDir[dir]] = \
                f[ind, outDir[dir]]


@numba.njit
def periodic(f, f_new, rho, u, faceList, outDirections, invDirections,
             boundaryVector, boundaryScalar, c, w, cs, Nx, Ny):
    pass
