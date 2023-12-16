import numba
import numpy as np
from numba import cuda

from pylabolt.parallel.cudaReduce import setBlockSize


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
        self.blockSize = setBlockSize(self.blocks)

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
        self.device.collisionType = cuda.to_device(np.array([simulation.
                                                   collisionScheme.
                                                   collisionType],
                                                   dtype=np.int32))
        if simulation.collisionScheme.collisionType == 1:
            simulation.collisionScheme.preFactor = np.diag(simulation.
                                                           collisionScheme.
                                                           preFactor,
                                                           k=0)
            self.device.preFactor = cuda.to_device(simulation.collisionScheme.
                                                   preFactor)
        elif simulation.collisionScheme.collisionType == 2:
            self.device.preFactor = cuda.to_device(simulation.collisionScheme.
                                                   preFactor)
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
            self.device.forcingPreFactor = \
                cuda.to_device(simulation.forcingScheme.forcingPreFactor)
        else:
            self.device.forcingType = cuda.to_device(self.device.forcingType)
            self.device.F = cuda.to_device(self.device.F)
            self.device.A = cuda.to_device(self.device.A)
            self.device.forcingPreFactor = \
                cuda.to_device(self.device.forcingPreFactor)

        # Copy fields data
        self.device.copyFields(simulation)

        # Create function arguments
        self.device.setFuncArgs(simulation)

        simulation.boundary.setupBoundary_cuda()

        if (simulation.options.computeForces is True
                or simulation.options.computeTorque is True):
            simulation.options.setupForcesParallel_cuda()
        if (simulation.obstacle.obsModifiable is True):
            simulation.obstacle.setupModifyObstacle_cuda(simulation.precision)

    def setupParallel(self, baseAlgorithm, simulation, parallel):
        if parallel is True:
            numba.set_num_threads(self.n_threads)
        baseAlgorithm.setupBase_cpu(parallel)
        simulation.setupParallel_cpu(parallel)
        if (simulation.options.computeForces is True or
                simulation.options.computeTorque is True):
            simulation.options.setupForcesParallel_cpu(parallel)
        if (simulation.obstacle.displaySolidMass is True):
            simulation.obstacle.solidMassSetup_cpu(parallel)
        if (simulation.obstacle.obsModifiable is True):
            simulation.obstacle.setupModifyObstacle_cpu(parallel)
