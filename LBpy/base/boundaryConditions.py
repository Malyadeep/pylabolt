import numba
import numpy as np


@numba.njit
def fixedU(fields, faceList, outDirections, invDirections,
           boundaryVector, boundaryScalar, lattice, mesh):
    c = lattice.c
    w = lattice.w
    cs_2 = 1/(lattice.cs * lattice.cs)
    for itr, ind in enumerate(faceList):
        invDir = invDirections[itr]
        outDir = outDirections[itr]
        rhoWall = fields.rho[ind]
        for dir in range(invDir.shape[0]):
            preFactor = 2 * w[outDir[dir]] * rhoWall *\
                ((c[outDir[dir], 0] * boundaryVector[0] +
                  c[outDir[dir], 1] * boundaryVector[1])) * cs_2
            fields.f_new[ind, invDir[dir]] = \
                fields.f[ind, outDir[dir]] - preFactor


@numba.njit
def fixedPressure(fields, faceList, outDirections, invDirections,
                  boundaryVector, boundaryScalar, lattice, mesh):
    c = lattice.c
    Nx = mesh.Nx
    w = lattice.w
    cs_2 = 1/(lattice.cs * lattice.cs)
    cs_4 = cs_2 * cs_2
    for itr, ind in enumerate(faceList):
        i, j = int(ind / mesh.Ny), int(ind % mesh.Ny)
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
        u_nb = fields.u[ind, :] + 0.5 * (fields.u[ind, :] -
                                         fields.u[ind_nb, :])
        # print(u_nb)
        u_nb_2 = u_nb[0] * u_nb[0] + u_nb[1] * u_nb[1]
        for dir in range(invDir.shape[0]):
            c_dot_u = (c[outDir[dir], 0] * u_nb[0] +
                       c[outDir[dir], 1] * u_nb[1])
            preFactor = 2 * w[outDir[dir]] * fields.rho[ind] *\
                (1. + c_dot_u * c_dot_u * 0.5 * cs_4 - u_nb_2 * 0.5 * cs_2)
            fields.f_new[ind, invDir[dir]] = \
                - fields.f[ind, outDir[dir]] + preFactor


@numba.njit
def bounceBack(fields, faceList, outDirections, invDirections,
               boundaryVector, boundaryScalar, lattice, mesh):
    for itr, ind in enumerate(faceList):
        invDir = invDirections[itr]
        outDir = outDirections[itr]
        for dir in range(invDir.shape[0]):
            fields.f_new[ind, invDir[dir]] = \
                fields.f[ind, outDir[dir]]


@numba.njit
def periodic(fields, faceList, outDirections, invDirections,
             boundaryVector, boundaryScalar, lattice, mesh):
    pass
