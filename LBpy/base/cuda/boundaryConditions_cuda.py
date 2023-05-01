from numba import cuda
import numpy as np


@cuda.jit
def fixedU(f, f_new, rho, u, faceList, outDirections, invDirections,
           boundaryVector, boundaryScalar, c, w, cs, Nx, Ny):
    cs_2 = 1/(cs * cs)
    ind = cuda.grid(1)
    if ind < Nx * Ny:
        for itr in range(faceList.shape[0]):
            if ind == faceList[itr]:
                rhoWall = rho[ind]
                for dir in range(invDirections.shape[0]):
                    preFactor = 2 * w[outDirections[dir]] * rhoWall *\
                        ((c[outDirections[dir], 0] * boundaryVector[0] +
                         c[outDirections[dir], 1] * boundaryVector[1])) * cs_2
                    f_new[ind, invDirections[dir]] = \
                        f[ind, outDirections[dir]] - preFactor
            else:
                continue
    else:
        return


@cuda.jit
def fixedPressure(f, f_new, rho, u, faceList, outDirections, invDirections,
                  boundaryVector, boundaryScalar, c, w, cs, Nx, Ny):
    cs_2 = 1/(cs * cs)
    cs_4 = cs_2 * cs_2
    ind = cuda.grid(1)
    if ind < Nx * Ny:
        for itr in range(faceList.shape[0]):
            if ind == faceList[itr]:
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
                for dir in range(invDirections.shape[0]):
                    c_dot_u = (c[outDirections[dir], 0] * u_nb[0] +
                               c[outDirections[dir], 1] * u_nb[1])
                    preFactor = 2 * w[outDirections[dir]] * boundaryScalar * \
                        cs_2 * (1. + c_dot_u * c_dot_u * 0.5 * cs_4 - u_nb_2
                                * 0.5 * cs_2)
                    f_new[ind, invDirections[dir]] = \
                        - f[ind, outDirections[dir]] + preFactor
            else:
                continue
    else:
        return


@cuda.jit
def bounceBack(f, f_new, rho, u, faceList, outDirections, invDirections,
               boundaryVector, boundaryScalar, c, w, cs, Nx, Ny):
    ind = cuda.grid(1)
    if ind < Nx * Ny:
        for itr in range(faceList.shape[0]):
            if ind == faceList[itr]:
                for dir in range(invDirections.shape[0]):
                    f_new[ind, invDirections[dir]] = \
                        f[ind, outDirections[dir]]
            else:
                continue
    else:
        return


@cuda.jit
def zeroGradient(f, f_new, rho, u, faceList, outDirections, invDirections,
                 boundaryVector, boundaryScalar, c, w, cs, Nx, Ny):
    ind = cuda.grid(1)
    if ind < Nx * Ny:
        for itr in range(faceList.shape[0]):
            if ind == faceList[itr]:
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
                    f[ind, k] = f[ind_nb, k]
            else:
                continue
    else:
        return


@cuda.jit
def periodic(f, f_new, rho, u, faceList, outDirections, invDirections,
             boundaryVector, boundaryScalar, c, w, cs, Nx, Ny):
    pass
