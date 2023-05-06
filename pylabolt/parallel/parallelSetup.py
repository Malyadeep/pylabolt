import numba
import numpy as np
from numba import cuda

from pylabolt.base.cuda.deviceFields import deviceData


class parallelSetup:
    def __init__(self, parallelization, n_threads, baseAlgorithm, simulation):
        self.mode = parallelization
        self.n_threads = n_threads
        self.blocks = 0
        if self.mode is None:
            self.setupParallel(baseAlgorithm, simulation, parallel=False)
        elif self.mode == 'openmp':
            self.setupParallel(baseAlgorithm, simulation, parallel=True)
        elif self.mode == 'cuda':
            self.device = self.setupCuda(simulation)

    def setupCuda(self, simulation):
        self.blocks = int(np.ceil(simulation.mesh.Nx * simulation.mesh.Ny /
                                  self.n_threads))
        device = deviceData(simulation.mesh, simulation.lattice,
                            simulation.precision)

        # Copy grid data
        device.Nx = cuda.to_device(np.array([simulation.mesh.Nx],
                                   dtype=np.int32))
        device.Ny = cuda.to_device(np.array([simulation.mesh.Ny],
                                   dtype=np.int32))

        # Copy lattice data
        device.noOfDirections = cuda.to_device(np.array([simulation.lattice.
                                               noOfDirections],
                                               dtype=np.int32))
        device.c = cuda.to_device(simulation.lattice.c)
        device.w = cuda.to_device(simulation.lattice.w)
        device.invList = cuda.to_device(simulation.lattice.invList)
        device.cs = cuda.to_device(np.array([simulation.lattice.
                                            cs], dtype=simulation.precision))
        device.cs_2 = cuda.to_device(np.array([simulation.collisionScheme.
                                               cs_2], dtype=simulation.
                                     precision))
        device.cs_4 = cuda.to_device(np.array([simulation.collisionScheme.
                                               cs_4], dtype=simulation.
                                     precision))

        # Copy scheme data
        device.preFactor = cuda.to_device(np.array([simulation.collisionScheme.
                                          preFactor], dtype=simulation.
                                          precision))
        device.collisionType = cuda.to_device(np.array([simulation.
                                              collisionScheme.collisionType],
                                              dtype=np.int32))
        device.equilibriumType = cuda.to_device(np.array([simulation.
                                                collisionScheme.
                                                equilibriumType],
                                                dtype=np.int32))
        if simulation.collisionScheme.equilibriumType == 1:
            device.equilibriumArgs = (device.cs_2[0], device.c, device.w)
        elif simulation.collisionScheme.equilibriumType == 2:
            device.equilibriumArgs = (device.cs_2[0], device.cs_4[0],
                                      device.c, device.w)

        # Copy fields data
        device.f = cuda.to_device(simulation.fields.f)
        device.f_eq = cuda.to_device(simulation.fields.f_eq)
        device.f_new = cuda.to_device(simulation.fields.f_new)
        device.u = cuda.to_device(simulation.fields.u)
        device.rho = cuda.to_device(simulation.fields.rho)
        device.solid = cuda.to_device(simulation.fields.solid)

        # Create function arguments
        device.collisionArgs = (
            device.Nx[0], device.Ny[0], device.f_eq, device.f, device.f_new,
            device.u, device.rho, device.solid, device.preFactor[0],
            device.cs_2[0], device.cs_4[0], device.c, device.w,
            device.equilibriumType[0], device.collisionType[0]
        )
        device.streamArgs = (
            device.Nx[0], device.Ny[0], device.f, device.f_new, device.c,
            device.noOfDirections[0], device.invList, device.solid
        )
        device.computeFieldsArgs = (
            device.Nx[0], device.Ny[0], device.f_new, device.u, device.rho,
            device.solid, device.c, device.noOfDirections[0]
        )
        simulation.boundary.setupBoundary_cuda()
        return device

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
