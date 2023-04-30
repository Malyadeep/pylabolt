from numba import cuda
import numpy as np


<<<<<<< HEAD
@cuda.jit
=======
>>>>>>> 763b1fb88aebd7534ec6dac5ae28097d575b24a7
def fixedU(f, f_new, rho, u, faceList, outDirections, invDirections,
           boundaryVector, boundaryScalar, c, w, cs, Nx, Ny):
    cs_2 = 1/(cs * cs)
    ind = cuda.grid(1)
    if ind < Nx * Ny:
        for itr in range(faceList.shape[0]):
            if ind == faceList[itr]:
<<<<<<< HEAD
                rhoWall = rho[ind]
=======
                rhoWall = rho[faceList[ind]]
>>>>>>> 763b1fb88aebd7534ec6dac5ae28097d575b24a7
                for dir in range(invDirections.shape[0]):
                    preFactor = 2 * w[outDirections[dir]] * rhoWall *\
                        ((c[outDirections[dir], 0] * boundaryVector[0] +
                         c[outDirections[dir], 1] * boundaryVector[1])) * cs_2
<<<<<<< HEAD
                    f_new[ind, invDirections[dir]] = \
                        f[ind, outDirections[dir]] - preFactor
=======
                    f_new[faceList[ind], invDirections[dir]] = \
                        f[faceList[ind], outDirections[dir]] - preFactor
>>>>>>> 763b1fb88aebd7534ec6dac5ae28097d575b24a7
            else:
                continue
    else:
        return


<<<<<<< HEAD
@cuda.jit
=======
>>>>>>> 763b1fb88aebd7534ec6dac5ae28097d575b24a7
def fixedPressure(f, f_new, rho, u, faceList, outDirections, invDirections,
                  boundaryVector, boundaryScalar, c, w, cs, Nx, Ny):
    cs_2 = 1/(cs * cs)
    cs_4 = cs_2 * cs_2
    ind = cuda.grid(1)
    if ind < Nx * Ny:
        for itr in range(faceList.shape[0]):
            if ind == faceList[itr]:
<<<<<<< HEAD
                i, j = int(ind / Ny), int(ind % Ny)
=======
                i, j = int(faceList[ind] / Ny), int(faceList[ind] % Ny)
>>>>>>> 763b1fb88aebd7534ec6dac5ae28097d575b24a7
                for direction in invDirections:
                    c_mag = int(np.ceil((c[direction, 0] * c[direction, 0]
                                + c[direction, 1] * c[direction, 1])))
                    if c_mag == 1:
                        i_nb = int(i + c[direction, 0])
                        j_nb = int(j + c[direction, 1])
                        ind_nb = int(i_nb * Ny + j_nb)
                        break
<<<<<<< HEAD
                u_nb = u[ind, :] + 0.5 * (u[ind, :] - u[ind_nb, :])
=======
                u_nb = u[faceList[ind], :] + 0.5 * (u[faceList[ind], :] -
                                                    u[ind_nb, :])
>>>>>>> 763b1fb88aebd7534ec6dac5ae28097d575b24a7
                u_nb_2 = u_nb[0] * u_nb[0] + u_nb[1] * u_nb[1]
                for dir in range(invDirections.shape[0]):
                    c_dot_u = (c[outDirections[dir], 0] * u_nb[0] +
                               c[outDirections[dir], 1] * u_nb[1])
                    preFactor = 2 * w[outDirections[dir]] * boundaryScalar * \
                        cs_2 * (1. + c_dot_u * c_dot_u * 0.5 * cs_4 - u_nb_2
                                * 0.5 * cs_2)
<<<<<<< HEAD
                    f_new[ind, invDirections[dir]] = \
                        - f[ind, outDirections[dir]] + preFactor
=======
                    f_new[faceList[ind], invDirections[dir]] = \
                        - f[faceList[ind], outDirections[dir]] + preFactor
>>>>>>> 763b1fb88aebd7534ec6dac5ae28097d575b24a7
            else:
                continue
    else:
        return


<<<<<<< HEAD
@cuda.jit
=======
>>>>>>> 763b1fb88aebd7534ec6dac5ae28097d575b24a7
def bounceBack(f, f_new, rho, u, faceList, outDirections, invDirections,
               boundaryVector, boundaryScalar, c, w, cs, Nx, Ny):
    ind = cuda.grid(1)
    if ind < Nx * Ny:
        for itr in range(faceList.shape[0]):
            if ind == faceList[itr]:
                for dir in range(invDirections.shape[0]):
<<<<<<< HEAD
                    f_new[ind, invDirections[dir]] = \
                        f[ind, outDirections[dir]]
=======
                    f_new[faceList[ind], invDirections[dir]] = \
                        f[faceList[ind], outDirections[dir]]
>>>>>>> 763b1fb88aebd7534ec6dac5ae28097d575b24a7
            else:
                continue
    else:
        return


<<<<<<< HEAD
@cuda.jit
=======
>>>>>>> 763b1fb88aebd7534ec6dac5ae28097d575b24a7
def zeroGradient(f, f_new, rho, u, faceList, outDirections, invDirections,
                 boundaryVector, boundaryScalar, c, w, cs, Nx, Ny):
    ind = cuda.grid(1)
    if ind < Nx * Ny:
        for itr in range(faceList.shape[0]):
            if ind == faceList[itr]:
<<<<<<< HEAD
                i, j = int(ind / Ny), int(ind % Ny)
=======
                i, j = int(faceList[ind] / Ny), int(faceList[ind] % Ny)
>>>>>>> 763b1fb88aebd7534ec6dac5ae28097d575b24a7
                for direction in invDirections:
                    c_mag = int(np.ceil((c[direction, 0] * c[direction, 0]
                                + c[direction, 1] * c[direction, 1])))
                    if c_mag == 1:
                        i_nb = int(i + c[direction, 0])
                        j_nb = int(j + c[direction, 1])
                        ind_nb = int(i_nb * Ny + j_nb)
                        break
                for k in range(f.shape[1]):
<<<<<<< HEAD
                    f[ind, k] = f[ind_nb, k]
=======
                    f[faceList[ind], k] = f[ind_nb, k]
>>>>>>> 763b1fb88aebd7534ec6dac5ae28097d575b24a7
            else:
                continue
    else:
        return


<<<<<<< HEAD
@cuda.jit
=======
>>>>>>> 763b1fb88aebd7534ec6dac5ae28097d575b24a7
def periodic(f, f_new, rho, u, faceList, outDirections, invDirections,
             boundaryVector, boundaryScalar, c, w, cs, Nx, Ny):
    pass
