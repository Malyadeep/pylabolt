import numpy as np


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

        # Fields data
        self.f = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
                          dtype=precision)
        self.f_eq = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
                             dtype=precision)
        self.f_new = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
                              dtype=precision)
        self.u = np.zeros((mesh.Nx * mesh.Ny, 2), dtype=precision)
        self.rho = np.zeros(mesh.Nx * mesh.Ny, dtype=precision)
        self.solid = np.zeros((mesh.Nx * mesh.Ny), dtype=np.int32)

        # Function arguments
        self.collisionArgs = ()
        self.streamArgs = ()
