import numpy as np
import numba

spec = [
    ('id', numba.int64),
    ('id_x', numba.int64),
    ('id_y', numba.int64),
    ('x', numba.float64),
    ('y', numba.float64),
    ('u', numba.float64[:]),
    ('rho', numba.float64),
    ('f', numba.float64[:]),
    ('f_eq', numba.float64[:]),
    ('f_new', numba.float64[:]),
    ('nodeType', numba.types.string),
    ('outDirections', numba.int32[:]),
    ('invDirections', numba.int32[:])
]


@numba.experimental.jitclass(spec)
class element:
    def __init__(self, mesh, lattice, id, U_initial, rho_initial):
        self.id = id
        self.id_x, self.id_y = int(id / mesh.Ny), int(id % mesh.Ny)
        self.x = mesh.delX * self.id_x
        self.y = mesh.delX * self.id_y
        self.u = np.zeros(2, dtype=np.float64)
        self.u = U_initial
        self.rho = rho_initial
        self.f = np.zeros(lattice.noOfDirections, dtype=np.float64)
        self.f_eq = np.zeros_like(self.f)
        self.f_new = np.zeros_like(self.f)
        self.nodeType = 'f'

    def computeFields(self, lattice):
        self.rho = np.sum(self.f)
        tempU, tempV = 0, 0
        for k in range(self.f.shape[0]):
            tempU += lattice.c[k, 0] * self.f[k]
            tempV += lattice.c[k, 1] * self.f[k]
        self.u[0] = tempU/(self.rho + 1e-9)
        self.u[1] = tempV/(self.rho + 1e-9)

    def setDirections(self, outDirections, invDirections):
        self.outDirections = outDirections
        self.invDirections = invDirections
