import numpy as np
from numba import cuda


class deviceData:
    def __init__(self, mesh, lattice, precision):
        # Grid Data
        self.Nx = np.zeros(1, dtype=precision)
        self.Ny = np.zeros(1, dtype=precision)

        # Lattice data
        self.noOfDirections = 0
        self.w = np.zeros((lattice.noOfDirections), dtype=precision)
        self.c = np.zeros((lattice.noOfDirections, 2), dtype=precision)
        self.invList = np.zeros((lattice.noOfDirections), dtype=np.int32)
        self.cs = np.zeros(1, dtype=precision)
        self.cs_2 = np.zeros(1, dtype=precision)
        self.cs_4 = np.zeros(1, dtype=precision)

        # Scheme data
        self.preFactor = np.zeros(1, dtype=precision)
        self.equilibriumArgs = ()
        self.equilibriumType = np.zeros(1, dtype=np.int32)
        self.collisionType = np.zeros(1, dtype=np.int32)
        self.forcingType = np.zeros(1, dtype=np.int32)
        self.source = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
                               dtype=precision)
        self.F = np.zeros(2, dtype=precision)
        self.A = np.zeros(1, dtype=precision)
        self.forcingPreFactor = np.zeros((lattice.noOfDirections,
                                          lattice.noOfDirections),
                                         dtype=precision)

        # Fields data
        self.f = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
                          dtype=precision)
        self.f_eq = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
                             dtype=precision)
        self.f_new = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
                              dtype=precision)
        self.u = np.zeros((mesh.Nx * mesh.Ny, 2), dtype=precision)
        self.rho = np.zeros(mesh.Nx * mesh.Ny, dtype=precision)
        self.solid = np.zeros((mesh.Nx * mesh.Ny, 2), dtype=np.int32)

        # Function arguments
        self.collisionArgs = ()
        self.streamArgs = ()
        self.computeFieldsArgs = ()

    def copyFields(self, simulation):
        self.f = cuda.to_device(simulation.fields.f)
        self.f_eq = cuda.to_device(simulation.fields.f_eq)
        self.f_new = cuda.to_device(simulation.fields.f_new)
        self.u = cuda.to_device(simulation.fields.u)
        self.rho = cuda.to_device(simulation.fields.rho)
        self.solid = cuda.to_device(simulation.fields.solid)

    def setFuncArgs(self, simulation):
        self.collisionArgs = (
            self.Nx[0], self.Ny[0], self.f_eq, self.f, self.f_new,
            self.u, self.rho, self.solid, self.preFactor,
            self.rho_0[0], self.U_0, self.cs_2[0], self.cs_4[0],
            self.c, self.w, self.source, self.noOfDirections[0], self.F,
            self.equilibriumType[0], self.collisionType[0],
            self.forcingType[0], self.forcingPreFactor
        )
        self.streamArgs = (
            self.Nx[0], self.Ny[0], self.f, self.f_new, self.solid,
            self.rho, self.u, self.c, self.w, self.noOfDirections[0],
            self.cs_2[0], self.invList
        )
        self.computeFieldsArgs = (
            self.Nx[0], self.Ny[0], self.f_new, self.u, self.rho,
            self.solid, self.c, self.noOfDirections[0], self.forcingType[0],
            self.F, self.A[0]
        )
