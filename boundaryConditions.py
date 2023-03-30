import numba
import numpy as np


@numba.njit
def fixedU(elements, elementList, boundaryVector,
           boundaryScalar, lattice, mesh):
    c = lattice.c
    w = lattice.w
    cs_2 = 1/(lattice.cs * lattice.cs)
    for ind in elementList:
        invList = elements[ind].invDirections
        outList = elements[ind].outDirections
        rhoWall = np.sum(elements[ind].f)
        for dir in range(invList.shape[0]):
            preFactor = 2 * w[outList[dir]] * rhoWall *\
                ((c[outList[dir], 0] * boundaryVector[0] +
                  c[outList[dir], 1] * boundaryVector[1])) * cs_2
            elements[ind].f_new[invList[dir]] = \
                elements[ind].f[outList[dir]] - preFactor


@numba.njit
def fixedPressure(elements, elementList, boundaryVector,
                  boundaryScalar, lattice, mesh):
    c = lattice.c
    Nx = mesh.Nx
    w = lattice.w
    cs_2 = 1/(lattice.cs * lattice.cs)
    cs_4 = cs_2 * cs_2
    for ind in elementList:
        invList = elements[ind].invDirections
        outList = elements[ind].outDirections
        for dir in range(invList.shape[0]):
            if int(np.ceil((c[dir, 0] * c[dir, 0]
                   + c[dir, 1] * c[dir, 1]))) == 1:
                i_nb = elements[ind].id_x + c[dir, 0]
                j_nb = elements[ind].id_y + c[dir, 1]
                ind_nb = int(i_nb * Nx + j_nb)
                break
        u_wall = elements[ind].u + 0.5 * (elements[ind].u - elements[ind_nb].u)
        u_wall_2 = np.dot(u_wall, u_wall)
        for dir in range(invList.shape[0]):
            c_dot_u = (c[outList[dir], 0] * u_wall[0] +
                       c[outList[dir], 1] * u_wall[1])
            preFactor = 2 * w[outList[dir]] * boundaryScalar *\
                (1. + c_dot_u * c_dot_u * 0.5 * cs_4 - u_wall_2 * 0.5 * cs_2)
            elements[ind].f_new[invList[dir]] = \
                - elements[ind].f[outList[dir]] + preFactor


@numba.njit
def bounceBack(elements, elementList, boundaryVector,
               boundaryScalar, lattice, mesh):
    for ind in elementList:
        invList = elements[ind].invDirections
        outList = elements[ind].outDirections
        for dir in range(invList.shape[0]):
            elements[ind].f_new[invList[dir]] = \
                elements[ind].f[outList[dir]]


@numba.njit
def periodic(elements, elementList, boundaryVector,
             boundaryScalar, lattice, mesh):
    pass
