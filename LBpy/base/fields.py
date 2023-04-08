import numpy as np


class fields:
    def __init__(self, mesh, lattice, U_initial, rho_initial):
        self.f = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
                          dtype=np.float64)
        self.f_eq = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
                             dtype=np.float64)
        self.f_new = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
                              dtype=np.float64)
        self.u = np.zeros((mesh.Nx * mesh.Ny, 2), dtype=np.float64)
        self.rho = np.full(mesh.Nx * mesh.Ny, fill_value=rho_initial,
                           dtype=np.float64)
        self.x = np.zeros(mesh.Nx * mesh.Ny, dtype=np.float64)
        self.y = np.zeros(mesh.Nx * mesh.Ny, dtype=np.float64)
        for ind in range(mesh.Nx * mesh.Ny):
            self.u[ind, 0] = U_initial[0]
            self.u[ind, 1] = U_initial[1]
            self.x[ind] = int(ind / mesh.Ny) * mesh.delX
            self.y[ind] = int(ind % mesh.Ny) * mesh.delX
        self.nodeType = np.full((mesh.Nx * mesh.Ny), fill_value=1,
                                dtype=np.int32)


def setObstacle(obstacle, fields):
    pass
