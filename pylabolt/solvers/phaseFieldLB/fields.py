import numpy as np

import numba


class fields:
    def __init__(self, mesh, lattice, initialFields, precision, size, rank):
        self.fieldList = ['u', 'rho', 'p', 'phi']
        self.f = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
                          dtype=precision)
        self.f_eq = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
                             dtype=precision)
        self.f_new = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
                              dtype=precision)
        self.h = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
                          dtype=precision)
        self.h_eq = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
                             dtype=precision)
        self.h_new = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
                              dtype=precision)
        self.u = np.zeros((mesh.Nx * mesh.Ny, 2), dtype=precision)
        self.u_initialize = np.zeros((mesh.Nx * mesh.Ny, 2), dtype=precision)
        self.rho = np.zeros(mesh.Nx * mesh.Ny, dtype=precision)
        self.p = np.zeros(mesh.Nx * mesh.Ny, dtype=precision)
        self.phi = np.zeros(mesh.Nx * mesh.Ny, dtype=precision)
        self.gradPhi = np.zeros((mesh.Nx * mesh.Ny, 2), dtype=precision)
        self.normalPhi = np.zeros((mesh.Nx * mesh.Ny, 2), dtype=precision)
        self.lapPhi = np.zeros(mesh.Nx * mesh.Ny, dtype=precision)
        self.boundaryNode = np.full((mesh.Nx * mesh.Ny), fill_value=0,
                                    dtype=np.int32)
        self.forceField = np.zeros((mesh.Nx * mesh.Ny, 2), dtype=precision)
        self.source = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
                               dtype=precision)
        self.stressTensor = np.zeros((mesh.Nx * mesh.Ny, 2, 2),
                                     dtype=precision)
        for ind in range(mesh.Nx * mesh.Ny):
            self.u[ind, 0] = initialFields.u[ind, 0]
            self.u[ind, 1] = initialFields.u[ind, 1]
            self.rho[ind] = initialFields.rho[ind]
            self.p[ind] = initialFields.p[ind]
            self.phi[ind] = initialFields.phi[ind]
            self.boundaryNode[ind] = initialFields.boundaryNode[ind]
        self.solid = np.full((mesh.Nx * mesh.Ny, 2), fill_value=0,
                             dtype=np.int32)
        self.procBoundary = np.zeros((mesh.Nx * mesh.Ny), dtype=np.int32)
        if size > 1:
            self.procBoundary = setProcBoundary(mesh.Nx, mesh.Ny)
            self.f_send_topBottom = np.zeros((mesh.Nx,
                                             lattice.noOfDirections),
                                             dtype=precision)
            self.f_recv_topBottom = np.zeros((mesh.Nx,
                                             lattice.noOfDirections),
                                             dtype=precision)
            self.f_send_leftRight = np.zeros((mesh.Ny,
                                             lattice.noOfDirections),
                                             dtype=precision)
            self.f_recv_leftRight = np.zeros((mesh.Ny,
                                             lattice.noOfDirections),
                                             dtype=precision)
            self.h_send_topBottom = np.zeros((mesh.Nx,
                                             lattice.noOfDirections),
                                             dtype=precision)
            self.h_recv_topBottom = np.zeros((mesh.Nx,
                                             lattice.noOfDirections),
                                             dtype=precision)
            self.h_send_leftRight = np.zeros((mesh.Ny,
                                             lattice.noOfDirections),
                                             dtype=precision)
            self.h_recv_leftRight = np.zeros((mesh.Ny,
                                             lattice.noOfDirections),
                                             dtype=precision)
            self.phi_send_topBottom = np.zeros(mesh.Nx, dtype=precision)
            self.phi_recv_topBottom = np.zeros(mesh.Nx, dtype=precision)
            self.phi_send_leftRight = np.zeros(mesh.Ny, dtype=precision)
            self.phi_recv_leftRight = np.zeros(mesh.Ny, dtype=precision)


@numba.njit
def setProcBoundary(Nx, Ny):
    procBoundary = np.ones((Nx * Ny), dtype=np.int32)
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            ind = i * Ny + j
            procBoundary[ind] = 0
    return procBoundary
