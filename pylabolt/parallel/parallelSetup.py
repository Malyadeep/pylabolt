import numba
import numpy as np
from numba import cuda


class parallelSetup:
    def __init__(self, parallelization, n_threads):
        self.mode = parallelization
        self.n_threads = n_threads
        self.blocks = 0
        self.device = None

    def setupCuda(self, simulation, device):
        self.device = device
        self.blocks = int(np.ceil(simulation.mesh.Nx * simulation.mesh.Ny /
                                  self.n_threads))

        # Copy grid data
        self.device.Nx = cuda.to_device(np.array([simulation.mesh.Nx],
                                        dtype=np.int32))
        self.device.Ny = cuda.to_device(np.array([simulation.mesh.Ny],
                                        dtype=np.int32))

        # Copy lattice data
        self.device.noOfDirections = cuda.to_device(np.array([simulation.
                                                    lattice.
                                                    noOfDirections],
                                                    dtype=np.int32))
        self.device.c = cuda.to_device(simulation.lattice.c)
        self.device.w = cuda.to_device(simulation.lattice.w)
        self.device.invList = cuda.to_device(simulation.lattice.invList)
        self.device.cs = cuda.to_device(np.array([simulation.lattice.
                                                 cs], dtype=simulation.
                                                 precision))
        self.device.cs_2 = cuda.to_device(np.array([simulation.collisionScheme.
                                                   cs_2], dtype=simulation.
                                                   precision))
        self.device.cs_4 = cuda.to_device(np.array([simulation.collisionScheme.
                                                   cs_4], dtype=simulation.
                                                   precision))

        # Copy scheme data
        self.device.tau_1 = cuda.to_device(np.array([simulation.
                                                    collisionScheme.
                                                    tau_1],
                                                    dtype=simulation.
                                                    precision))
        self.device.collisionType = cuda.to_device(np.array([simulation.
                                                   collisionScheme.
                                                   collisionType],
                                                   dtype=np.int32))
        self.device.equilibriumType = cuda.to_device(np.array([simulation.
                                                     collisionScheme.
                                                     equilibriumType],
                                                     dtype=np.int32))
        if simulation.collisionScheme.equilibriumType == 1:
            self.device.rho_0 = cuda.to_device(np.array([simulation.
                                               collisionScheme.rho_0],
                                               dtype=simulation.precision))
            self.device.U_0 = cuda.to_device(np.array([0, 0],
                                             dtype=simulation.precision))
        elif simulation.collisionScheme.equilibriumType == 2:
            self.device.rho_0 = cuda.to_device(np.array([0],
                                               dtype=simulation.precision))
            self.device.U_0 = cuda.to_device(np.array([0, 0],
                                             dtype=simulation.precision))
        elif simulation.collisionScheme.equilibriumType == 3:
            self.device.rho_0 = cuda.to_device(np.array([simulation.
                                               collisionScheme.rho_0],
                                               dtype=simulation.precision))
            self.device.U_0 = cuda.to_device(np.array([0, 0],
                                             dtype=simulation.precision))
        elif simulation.collisionScheme.equilibriumType == 4:
            self.device.rho_0 = cuda.to_device(np.array([simulation.
                                               collisionScheme.rho_0],
                                               dtype=simulation.precision))
            self.device.U_0 = cuda.to_device(simulation.
                                             collisionScheme.U_0)

        self.device.source = cuda.to_device(self.device.source)
        if simulation.forcingScheme.forcingFlag != 0:
            self.device.forcingType = cuda.to_device(np.array([simulation.
                                                     forcingScheme.
                                                     forcingType],
                                                     dtype=np.int32))
            self.device.F = cuda.to_device(np.array(simulation.forcingScheme.F,
                                           dtype=simulation.precision))
            self.device.A = cuda.to_device(np.array([simulation.
                                           forcingScheme.A],
                                           dtype=simulation.
                                           precision))

        # Copy fields data
        self.device.copyFields(simulation)

        # Create function arguments
        self.device.setFuncArgs(simulation)

        simulation.boundary.setupBoundary_cuda()

        if (simulation.options.computeForces is True
                or simulation.options.computeTorque is True):
            simulation.options.setupForcesParallel_cuda()

    def setupParallel(self, baseAlgorithm, simulation, parallel):
        if parallel is True:
            numba.set_num_threads(self.n_threads)
        baseAlgorithm.equilibriumRelaxation = numba.njit(baseAlgorithm.
                                                         equilibriumRelaxation,
                                                         parallel=parallel,
                                                         cache=False,
                                                         nogil=True)
        baseAlgorithm.stream = numba.njit(baseAlgorithm.stream,
                                          parallel=parallel,
                                          cache=False,
                                          nogil=True)
        baseAlgorithm.computeFields = numba.njit(baseAlgorithm.computeFields,
                                                 parallel=parallel,
                                                 cache=False,
                                                 nogil=True)
        simulation.boundary.setupBoundary_cpu(parallel)
        if (simulation.options.computeForces is True or
                simulation.options.computeTorque is True):
            simulation.options.setupForcesParallel_cpu(parallel)
