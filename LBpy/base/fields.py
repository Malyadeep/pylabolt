import numpy as np
import os
import numba


class fields:
    def __init__(self, mesh, lattice, U_initial, rho_initial, precision, size):
        self.f = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
                          dtype=precision)
        self.f_eq = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
                             dtype=precision)
        self.f_new = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
                              dtype=precision)
        self.u = np.zeros((mesh.Nx * mesh.Ny, 2), dtype=precision)
        self.rho = np.full(mesh.Nx * mesh.Ny, fill_value=rho_initial,
                           dtype=precision)
        for ind in range(mesh.Nx * mesh.Ny):
            self.u[ind, 0] = U_initial[0]
            self.u[ind, 1] = U_initial[1]
        self.solid = np.full((mesh.Nx * mesh.Ny), fill_value=0,
                             dtype=np.int32)
        self.procBoundary = np.zeros((mesh.Nx * mesh.Ny), dtype=np.int32)
        if size > 1:
            self.procBoundary = setProcBoundary(mesh.Nx, mesh.Ny)
            self.f_send_topBottom = np.zeros((mesh.Nx + 2, 9), dtype=precision)
            self.f_recv_topBottom = np.zeros((mesh.Nx + 2, 9), dtype=precision)
            self.f_send_leftRight = np.zeros((mesh.Ny + 2, 9), dtype=precision)
            self.f_recv_leftRight = np.zeros((mesh.Ny + 2, 9), dtype=precision)


@numba.njit
def setProcBoundary(Nx, Ny):
    procBoundary = np.ones((Nx * Ny), dtype=np.int32)
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            ind = i * Ny + j
            procBoundary[ind] = 0
    return procBoundary


def circle(center, radius, solid, mesh):
    center_idx = np.int32(np.divide(center, mesh.delX))
    radius_idx = int(radius/mesh.delX)
    for i in range(mesh.Nx_global):
        for j in range(mesh.Ny_global):
            if ((i - center_idx[0])*(i - center_idx[0]) +
                    (j - center_idx[1])*(j - center_idx[1]) <=
                    radius_idx*radius_idx):
                ind = i * mesh.Ny_global + j
                solid[ind] = 1


def rectangle(boundingBox, solid, mesh):
    boundingBox_idx = np.int32(np.divide(boundingBox, mesh.delX))
    for i in range(mesh.Nx_global):
        for j in range(mesh.Ny_global):
            if (i >= boundingBox_idx[0, 0] and i <= boundingBox_idx[1, 0]
                    and j >= boundingBox_idx[0, 1] and
                    j <= boundingBox_idx[1, 1]):
                ind = i * mesh.Ny_global + j
                solid[ind] = 1


def setObstacle(obstacle, mesh):
    try:
        solid = np.full((mesh.Nx_global * mesh.Ny_global), fill_value=0,
                        dtype=np.int32)
        if len(obstacle.keys()) == 0:
            return solid
        obstacleType = obstacle['type']
        if obstacleType == 'circle':
            center = obstacle['center']
            radius = obstacle['radius']
            if isinstance(center, list) and isinstance(radius, float):
                center = np.array(center, dtype=np.float64)
                radius = np.float64(radius)
            else:
                print("ERROR!")
                print("For 'circle' type obstacle center must be a list and"
                      + " radius must be a float")
                os._exit(1)
            circle(center, radius, solid, mesh)
        elif obstacleType == 'rectangle':
            boundingBox = obstacle['boundingBox']
            if isinstance(boundingBox, list):
                boundingBox = np.array(boundingBox, dtype=np.float64)
            else:
                print("ERROR!")
                print("For 'rectangle' type obstacle bounding box must be"
                      + " a list")
                os._exit(1)
            rectangle(boundingBox, solid, mesh)
        else:
            print("ERROR!")
            print("Unsupported obstacle type!")
            os._exit(1)
        return solid
    except KeyError as e:
        print("ERROR!")
        print(str(e) + " keyword missing in 'obstacle' dictionary")
        os._exit(1)
