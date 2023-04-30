import numpy as np


class deviceData:
<<<<<<< HEAD
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
=======
    def __init__(self, mesh, lattice):
        # Grid Data
        self.Nx = np.zeros(1, dtype=np.float32)
        self.Ny = np.zeros(1, dtype=np.float32)

        # Lattice data
        self.noOfDirections = 0
        self.w = np.zeros((lattice.noOfDirections), dtype=np.float32)
        self.c = np.zeros((lattice.noOfDirections, 2), dtype=np.float32)
        self.invList = np.zeros((lattice.noOfDirections), dtype=np.int32)
        self.cs = np.zeros(1, dtype=np.float32)
        self.cs_2 = np.zeros(1, dtype=np.float32)
        self.cs_4 = np.zeros(1, dtype=np.float32)

        # Scheme data
        self.preFactor = np.zeros(1, dtype=np.float32)
>>>>>>> 763b1fb88aebd7534ec6dac5ae28097d575b24a7
        self.equilibriumArgs = ()
        self.equilibriumType = np.zeros(1, dtype=np.int32)
        self.collisionType = np.zeros(1, dtype=np.int32)

        # Fields data
        self.f = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
<<<<<<< HEAD
                          dtype=precision)
        self.f_eq = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
                             dtype=precision)
        self.f_new = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
                              dtype=precision)
        self.u = np.zeros((mesh.Nx * mesh.Ny, 2), dtype=precision)
        self.rho = np.zeros(mesh.Nx * mesh.Ny, dtype=precision)
        self.solid = np.zeros((mesh.Nx * mesh.Ny), dtype=np.int32)
=======
                          dtype=np.float32)
        self.f_eq = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
                             dtype=np.float32)
        self.f_new = np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections),
                              dtype=np.float32)
        self.u = np.zeros((mesh.Nx * mesh.Ny, 2), dtype=np.float32)
        self.rho = np.zeros(mesh.Nx * mesh.Ny, dtype=np.float32)
        self.solid = np.zeros((mesh.Nx * mesh.Ny), fill_value=0,
                              dtype=np.int32)
>>>>>>> 763b1fb88aebd7534ec6dac5ae28097d575b24a7

        # Function arguments
        self.collisionArgs = ()
        self.streamArgs = ()