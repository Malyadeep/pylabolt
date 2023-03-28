import numpy as np
import numba

spec = [
    ('id', numba.int64),
    ('x', numba.float64),
    ('y', numba.float64),
    ('u', numba.float64[:]),
    ('rho', numba.float64),
    ('f', numba.float64[:]),
    ('f_eq', numba.float64[:]),
    ('f_new', numba.float64[:]),
    ('nodeType', numba.uint8),
    ('outDirections', numba.int32[:]),
    ('invDirections', numba.int32[:])
]


@numba.experimental.jitclass(spec)
class element:
    def __init__(self, mesh, lattice, id, U_initial, rho_initial):
        self.id = id
        self.x = mesh.delX * int(id % mesh.Nx)
        self.y = mesh.delX * int(id / mesh.Nx)
        self.u = U_initial
        self.rho = rho_initial
        self.f = np.zeros(lattice.noOfDirections, dtype=np.float64)
        self.f_eq = np.zeros_like(self.f)
        self.f_new = np.zeros_like(self.f)
        self.nodeType = 'f'

    def computeFields(self, lattice):
        self.rho = np.sum(self.f)
        for k in range(self.f.shape[0]):
            tempU = np.sum(np.dot(lattice.c[:, 0], self.f))
            tempV = np.sum(np.dot(lattice.c[:, 1], self.f))
        self.u[0] = tempU/(self.rho + 1e-9)
        self.u[1] = tempV/(self.rho + 1e-9)

    def setDirections(self, outDirections, invDirections):
        self.outDirections = outDirections
        self.invDirections = invDirections
