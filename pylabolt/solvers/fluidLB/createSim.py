import os
import sys
import numpy as np
import numba
from copy import deepcopy


from pylabolt.base import (mesh, lattice, boundary, obstacle, transport)
from pylabolt.solvers.fluidLB import (initFields, fields)
from pylabolt.parallel.MPI_decompose import (decompose, distributeSolid_mpi,
                                             distributeInitialFields_mpi,
                                             distributeBoundaries_mpi,
                                             distributeForceNodes_mpi)
from pylabolt.utils.options import options
from pylabolt.solvers.fluidLB import schemeLB


@numba.njit
def initializePopulations(Nx, Ny, f_eq, f, f_new, u, rho,
                          noOfDirections, equilibriumFunc,
                          equilibriumArgs, procBoundary, boundaryNode):
    for ind in range(Nx * Ny):
        if procBoundary[ind] != 1 and boundaryNode[ind] != 1:
            equilibriumFunc(f_eq[ind, :], u[ind, :], rho[ind],
                            *equilibriumArgs)
            for k in range(noOfDirections):
                f[ind, k] = f_eq[ind, k]
                f_new[ind, k] = f_eq[ind, k]


class simulation:
    def __init__(self, parallelization, rank, size, comm):
        self.rank = rank
        self.size = size
        if rank == 0:
            print('Reading simulation parameters...\n', flush=True)
        try:
            workingDir = os.getcwd()
            sys.path.append(workingDir)
            from simulation import (controlDict, boundaryDict, collisionDict,
                                    latticeDict, meshDict, internalFields,
                                    transportDict)
        except ImportError as e:
            print('FATAL ERROR!')
            print(str(e))
            print('Aborting....')
            os._exit(1)

        if rank == 0:
            print('Setting control parameters...', flush=True)
        try:
            self.startTime = controlDict['startTime']
            self.endTime = controlDict['endTime']
            self.stdOutputInterval = controlDict['stdOutputInterval']
            self.saveInterval = controlDict['saveInterval']
            self.saveStateInterval = controlDict['saveStateInterval']
            self.relTolU = controlDict['relTolU']
            self.relTolV = controlDict['relTolV']
            self.relTolRho = controlDict['relTolRho']
            self.precisionType = controlDict['precision']
            if self.precisionType == 'single':
                self.precision = np.float32
            elif self.precisionType == 'double':
                self.precision = np.float64
            else:
                raise RuntimeError("Incorrect precision specified!")
        except KeyError as e:
            if rank == 0:
                print('ERROR! Keyword ' + str(e) + ' missing in controlDict')
            os._exit(1)
        if rank == 0:
            print('Setting control parameters done!\n', flush=True)

        if rank == 0:
            print('Reading mesh info and creating mesh...', flush=True)
        self.mesh = mesh.createMesh(meshDict, self.precision, self.rank)
        if rank == 0:
            print('Reading mesh info and creating mesh done!\n', flush=True)

        if rank == 0:
            print('Setting lattice structure...', flush=True)
        self.lattice = lattice.createLattice(latticeDict, self.precision)
        if rank == 0:
            print('Setting lattice structure done!\n', flush=True)
            print('Setting transport parameters...', flush=True)
        self.transport = transport.transportDef()
        self.transport.readFluidTransportDict(transportDict, self.precision,
                                              self.rank)
        if rank == 0:
            print('Setting transport parameters done!\n', flush=True)
            print('Setting collision scheme and equilibrium model...',
                  flush=True)
        self.collisionScheme = schemeLB.collisionScheme(self.lattice,
                                                        self.mesh,
                                                        collisionDict,
                                                        parallelization,
                                                        self.transport,
                                                        self.rank,
                                                        self.precision,
                                                        comm)
        if rank == 0:
            print('Setting collision scheme and equilibrium model done!\n',
                  flush=True)
            print('Setting forcing scheme...', flush=True)
        self.forcingScheme = schemeLB.forcingScheme(self.precision,
                                                    self.lattice)
        self.forcingScheme.setForcingScheme(self.lattice,
                                            self.collisionScheme,
                                            self.rank,
                                            self.precision)

        if rank == 0:
            print('Setting forcing scheme done!\n', flush=True)

        if size > 1:
            if rank == 0:
                print('Decomposing domain...', flush=True)
            self.mpiParams = decompose(self.mesh, self.rank, self.size, comm)
            if rank == 0:
                print('Domain decomposition done!\n', flush=True)

        if rank == 0:
            print('Reading options...', flush=True)
        self.options = options(self.rank, self.precision, self.mesh)

        if rank == 0:
            print('Reading options done!\n', flush=True)

        if rank == 0:
            print('Initializing fields...', flush=True)

        initialFields_temp = initFields.initFields(internalFields, self.mesh,
                                                   self.precision, self.rank,
                                                   comm)
        self.initialFields = initFields.initialFields(self.mesh.Nx,
                                                      self.mesh.Ny,
                                                      self.precision)
        self.obstacle = obstacle.obstacleSetup(size, self.mesh, self.precision)
        solid = self.obstacle.setObstacle(self.mesh, self.precision,
                                          self.options, initialFields_temp,
                                          self.rank, size, comm)
        if rank == 0:
            print('Initializing fields done!\n', flush=True)
            print('Reading boundary conditions...', flush=True)
        self.boundary = boundary.boundary(boundaryDict)
        if rank == 0:
            self.boundary.readBoundaryDict(initialFields_temp, self.lattice,
                                           self.mesh, self.rank,
                                           self.precision)
            self.boundary.initializeBoundary(self.lattice, self.mesh,
                                             initialFields_temp,
                                             solid, self.precision, rank)
        if rank == 0:
            print('Reading boundary conditions done...\n', flush=True)
        if (self.options.computeForces is True or self.options.computeTorque
                is True and rank == 0):
            if size > 1:
                self.obstacle.\
                    computeFluidSolidNb(solid, self.mesh, self.lattice,
                                        initialFields_temp, self.size,
                                        mpiParams=self.mpiParams)
            else:
                self.obstacle.\
                    computeFluidSolidNb(solid, self.mesh, self.lattice,
                                        initialFields_temp, self.size)

        if size > 1:
            distributeInitialFields_mpi(initialFields_temp, self.initialFields,
                                        self.mpiParams, self.mesh, self.rank,
                                        self.size, comm, self.precision)
        else:
            self.initialFields = initialFields_temp
        self.fields = fields.fields(self.mesh, self.lattice,
                                    self.initialFields,
                                    self.precision, size, rank)

        obstacleTemp = deepcopy(self.obstacle)
        if size == 1 and solid is not None:
            self.fields.solid = solid
        elif size > 1 and solid is not None:
            distributeSolid_mpi(solid, self.obstacle,
                                self.fields, self.mpiParams, self.mesh,
                                self.precision, self.rank, self.size, comm)

        if (self.options.computeForces is True or self.options.computeTorque
                is True):
            if rank == 0:
                self.options.gatherObstacleNodes(obstacleTemp)
                self.options.gatherBoundaryNodes(self.boundary)
                self.options.initializeForceReconstruction(self.precision)

        if size > 1:
            distributeBoundaries_mpi(self.boundary, self.mpiParams, self.mesh,
                                     rank, size, self.precision, comm)
            if (self.options.computeForces is True or
                    self.options.computeTorque is True):
                distributeForceNodes_mpi(self.options, self.mpiParams,
                                         self.mesh, rank, size,
                                         self.precision, comm)
        # if rank == 0:
        #     self.options.details(rank)
        del obstacleTemp, solid
        # initialize functions
        self.equilibriumFunc = self.collisionScheme.equilibriumFunc
        self.equilibriumArgs = self.collisionScheme.equilibriumArgs
        self.collisionFunc = self.collisionScheme.collisionFunc
        self.setBoundaryFunc = self.boundary.setBoundary

        # Prepare function arguments
        self.collisionArgs = (
            self.mesh.Nx, self.mesh.Ny, self.fields.f_eq,
            self.fields.f, self.fields.f_new, self.fields.u, self.fields.rho,
            self.fields.solid, self.fields.boundaryNode,
            self.fields.source, self.forcingScheme.gravity, self.collisionFunc,
            self.equilibriumFunc, self.collisionScheme.preFactor,
            self.collisionScheme.equilibriumArgs, self.fields.procBoundary,
            self.forcingScheme.forceFunc_force,
            self.forcingScheme.forceArgs_force,
            self.forcingScheme.forcingPreFactor,
            self.lattice.noOfDirections, self.precision
        )

        if size == 1:
            nx, ny = 0, 0
            nProc_x, nProc_y = 1, 1
        else:
            nx, ny = self.mpiParams.nx, self.mpiParams.ny
            nProc_x, nProc_y = self.mpiParams.nProc_x, self.mpiParams.nProc_y
        self.streamArgs = (
            self.mesh.Nx, self.mesh.Ny, self.fields.f, self.fields.f_new,
            self.fields.solid, self.fields.rho, self.fields.u,
            self.fields.boundaryNode,
            self.fields.procBoundary, self.lattice.c, self.lattice.w,
            self.lattice.noOfDirections, self.collisionScheme.cs_2,
            self.lattice.invList, nx, ny, nProc_x, nProc_y, self.size
        )

        self.computeFieldsArgs = (
            self.mesh.Nx, self.mesh.Ny, self.fields.f_new,
            self.fields.u, self.fields.rho, self.fields.solid,
            self.fields.boundaryNode, self.forcingScheme.gravity,
            self.lattice.c, self.lattice.noOfDirections,
            self.fields.procBoundary, self.size,
            self.forcingScheme.forceFunc_vel,
            self.forcingScheme.forceCoeffVel,
            self.fields.f_eq, self.precision
        )

        if size > 1:
            self.gatherArgs = (self.fields.u, self.fields.rho,
                               self.fields.solid, self.rank,
                               self.mpiParams.nProc_x,
                               self.mpiParams.nProc_y,
                               self.mesh.Nx_global, self.mesh.Ny_global,
                               self.mesh.Nx, self.mesh.Ny, self.precision)

            self.proc_boundaryArgs = (self.mesh.Nx, self.mesh.Ny,
                                      self.fields.f,
                                      self.fields.f_send_topBottom,
                                      self.fields.f_recv_topBottom,
                                      self.fields.f_send_leftRight,
                                      self.fields.f_recv_leftRight,
                                      self.mpiParams.nx, self.mpiParams.ny,
                                      self.mpiParams.nProc_x,
                                      self.mpiParams.nProc_y)
            self.proc_copyArgs = (self.mesh.Nx, self.mesh.Ny,
                                  self.lattice.c, self.fields.f_new)

        self.boundary.setBoundaryArgs(self.lattice, self.mesh, self.fields,
                                      self.collisionScheme)

        initializePopulations(
            self.mesh.Nx, self.mesh.Ny, self.fields.f_eq, self.fields.f,
            self.fields.f_new, self.fields.u, self.fields.rho,
            self.lattice.noOfDirections, self.equilibriumFunc,
            self.equilibriumArgs,
            self.fields.procBoundary, self.fields.boundaryNode
        )

    def setupParallel_cpu(self, parallel):
        self.boundary.setupBoundary_cpu(parallel)
