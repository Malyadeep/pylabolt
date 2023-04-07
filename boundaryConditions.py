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
    for ind in faceList:
        i, j = int(id / mesh.Ny), int(id % mesh.Ny)
        invDir = invDirections[ind]
        outDir = outDirections[ind]
        for dir in range(invDir.shape[0]):
            if int(np.ceil((c[dir, 0] * c[dir, 0]
                   + c[dir, 1] * c[dir, 1]))) == 1:
                i_nb = i + c[dir, 0]
                j_nb = j + c[dir, 1]
                ind_nb = int(i_nb * Nx + j_nb)
                break
        u_wall = fields.u[ind, :] + 0.5 * (fields.u[ind, :] -
                                           fields.u[ind_nb, :])
        u_wall_2 = np.dot(u_wall, u_wall)
        for dir in range(invDir.shape[0]):
            c_dot_u = (c[outDir[dir], 0] * u_wall[0] +
                       c[outDir[dir], 1] * u_wall[1])
            preFactor = 2 * w[outDir[dir]] * boundaryScalar *\
                (1. + c_dot_u * c_dot_u * 0.5 * cs_4 - u_wall_2 * 0.5 * cs_2)
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
