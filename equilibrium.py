import numba
import numpy as np

spec = [
    ('cs_2', numba.float64),
    ('cs_4', numba.float64),
    ('Nx', numba.float64),
    ('Ny', numba.float64),
    ('w', numba.float64[:]),
    ('c', numba.float64[:, :])
]


@numba.experimental.jitclass(spec)
class equilibrium:
    def __init__(self, mesh, lattice):
        self.cs_2 = 1/(lattice.cs*lattice.cs)
        self.cs_4 = self.cs_2/(lattice.cs*lattice.cs)
        self.Nx = mesh.Nx
        self.Ny = mesh.Ny
        self.c = lattice.c
        self.w = lattice.w

    def secondOrder(self, f_eq, u, rho):
        t2 = np.dot(u, u)
        for k in range(f_eq.shape[0]):
            t1 = np.dot(self.c[k, :], u)
            f_eq[k] = self.w[k] * rho * (1 + t1 * self.cs_2 +
                                         0.5 * t1 * t1 * self.cs_4 -
                                         0.5 * t2 * self.cs_2)

    def firstOrder(self, f_eq, u, rho):
        for k in range(f_eq.shape[0]):
            t1 = np.dot(self.c[k, :], u)
            f_eq[k] = self.w[k] * rho * (1 + t1 * self.cs_2)
