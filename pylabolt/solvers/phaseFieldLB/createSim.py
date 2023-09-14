import os
import sys
import numpy as np
import numba
from copy import deepcopy


from pylabolt.base import (mesh, lattice, boundary, transport,
                           obstacle)
from pylabolt.solvers.phaseFieldLB import (initFields, fields)
from pylabolt.parallel.MPI_decompose import (decompose, distributeSolid_mpi,
                                             distributeInitialFields_mpi,
                                             distributeBoundaries_mpi,
                                             distributeForceNodes_mpi)
from pylabolt.utils.options import options
from pylabolt.solvers.phaseFieldLB import phaseField, schemeLB


@numba.njit
def initializePopulations(Nx, Ny, pop_eq, pop, pop_new, u, scalar,
                          noOfDirections, equilibriumFunc,
                          equilibriumArgs, procBoundary, boundaryNode):
    for ind in range(Nx * Ny):
        if procBoundary[ind] != 1 and boundaryNode[ind] != 1:
            equilibriumFunc(pop_eq[ind, :], u[ind, :], scalar[ind],
                            *equilibriumArgs)
            for k in range(noOfDirections):
                pop[ind, k] = pop_eq[ind, k]
                pop_new[ind, k] = pop_eq[ind, k]


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
                                    phaseDict, transportDict)
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
            self.relTolPhi = controlDict['relTolPhi']
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
        self.transport.readPhaseTransportDict(transportDict, self.precision,
                                              self.rank)
        if rank == 0:
            print('Setting transport parameters done!\n', flush=True)
            print('Setting Phase field parameters...',
                  flush=True)
        self.phaseField = phaseField.phaseFieldDef(phaseDict, self.lattice,
                                                   self.transport, self.rank,
                                                   self.precision)
        if rank == 0:
            print('Setting Phase field parameters done!\n',
                  flush=True)
            print('Setting collision scheme and equilibrium model...',
                  flush=True)
        self.collisionScheme = schemeLB.collisionScheme(self.lattice,
                                                        self.mesh,
                                                        collisionDict,
                                                        parallelization,
                                                        self.transport,
                                                        self.phaseField,
                                                        self.rank,
                                                        self.precision)
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
        if self.phaseField.contactAngle is not None:
            wettingFlag = True
        else:
            wettingFlag = False
        self.options = options(self.rank, self.precision, self.mesh,
                               wetting=wettingFlag, phase=True)

        if rank == 0:
            print('Reading options done!\n', flush=True)

        if rank == 0:
            print('Initializing fields...', flush=True)

        initialFields_temp = initFields.initFields(internalFields, self.mesh,
                                                   self.lattice,
                                                   self.precision, self.rank,
                                                   comm)
        self.initialFields = initFields.initialFields(self.mesh.Nx,
                                                      self.mesh.Ny,
                                                      self.precision)
        self.obstacle = obstacle.obstacleSetup(size, self.mesh, self.precision,
                                               phase=True)
        solid = self.obstacle.setObstacle(self.mesh, self.precision,
                                          self.options, initialFields_temp,
                                          self.rank, size)
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
                                             solid, self.precision)
        if rank == 0:
            print('Reading boundary conditions done...\n', flush=True)

        if (self.options.computeForces is True or self.options.computeTorque
                is True or self.phaseField.contactAngle is not None
                and rank == 0):
            self.obstacle.computeFluidSolidNb(solid, self.mesh, self.lattice,
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
        if size == 1:
            self.fields.solid = solid
        elif size > 1:
            distributeSolid_mpi(solid, self.obstacle,
                                self.fields, self.mpiParams, self.mesh,
                                self.precision, self.rank, self.size, comm)

        if (self.options.computeForces is True or self.options.computeTorque
                is True or self.phaseField.contactAngle is not None):
            if rank == 0:
                self.options.gatherObstacleNodes(obstacleTemp)
                self.options.gatherBoundaryNodes(self.boundary)
        if size > 1:
            distributeBoundaries_mpi(self.boundary, self.mpiParams, self.mesh,
                                     rank, size, self.precision, comm)
            if (self.options.computeForces is True or
                    self.options.computeTorque is True or
                    self.phaseField.contactAngle is not None):
                distributeForceNodes_mpi(self.options, self.mpiParams,
                                         self.mesh, rank, size,
                                         self.precision, comm)
        if self.phaseField.contactAngle is not None:
            self.options.computeSolidNormals(self.fields, self.lattice,
                                             self.mesh, size, self.precision)
        del obstacleTemp, solid
        # initialize functions
        self.equilibriumFuncFluid = self.collisionScheme.equilibriumFuncFluid
        self.equilibriumArgsFluid = self.collisionScheme.equilibriumArgsFluid
        self.collisionFuncFluid = self.collisionScheme.collisionFuncFluid
        self.equilibriumFuncPhase = self.collisionScheme.equilibriumFuncPhase
        self.equilibriumArgsPhase = self.collisionScheme.equilibriumArgsPhase
        self.collisionFuncPhase = self.collisionScheme.collisionFuncPhase
        self.setBoundaryFunc = self.boundary.setBoundary

        # Prepare function arguments
        self.collisionArgsFluid = [
            self.mesh.Nx, self.mesh.Ny, self.fields.f_eq,
            self.fields.f, self.fields.f_new, self.fields.u, self.fields.p,
            self.fields.rho, self.fields.solid, self.fields.forceField,
            self.fields.source, self.collisionFuncFluid,
            self.equilibriumFuncFluid, self.collisionScheme.preFactorFluid,
            self.equilibriumArgsFluid, self.fields.boundaryNode,
            self.fields.procBoundary,
            self.forcingScheme.forceFunc_force,
            self.forcingScheme.forceArgs_force,
            self.forcingScheme.forcingPreFactorFluid, self.lattice.cs_2]

        self.collisionArgsPhase = [
            self.mesh.Nx, self.mesh.Ny, self.fields.h_eq,
            self.fields.h, self.fields.h_new, self.fields.u, self.fields.phi,
            self.fields.gradPhi, self.fields.solid, self.fields.boundaryNode,
            self.phaseField.interfaceWidth, self.lattice.c, self.lattice.w,
            self.collisionFuncPhase, self.equilibriumFuncPhase,
            self.collisionScheme.preFactorPhase,
            self.collisionScheme.equilibriumArgsPhase,
            self.fields.procBoundary, self.forcingScheme.forcingPreFactorPhase,
            self.lattice.noOfDirections, self.precision
        ]

        self.streamArgsFluid = [
            self.mesh.Nx, self.mesh.Ny, self.fields.f, self.fields.f_new,
            self.fields.solid, self.fields.rho, self.fields.u,
            self.fields.procBoundary, self.fields.boundaryNode, self.lattice.c,
            self.lattice.w, self.lattice.noOfDirections,
            self.collisionScheme.cs_2, self.lattice.invList, self.size
        ]

        self.streamArgsPhase = [
            self.mesh.Nx, self.mesh.Ny, self.fields.h, self.fields.h_new,
            self.fields.solid, self.fields.procBoundary,
            self.fields.boundaryNode, self.lattice.c,
            self.lattice.noOfDirections, self.lattice.invList,
            self.size
        ]

        self.computeFieldsArgsFluid = [
            self.mesh.Nx, self.mesh.Ny, self.fields.f_new,
            self.fields.u, self.fields.p, self.fields.rho,
            self.fields.phi, self.fields.forceField, self.fields.solid,
            self.lattice.c, self.lattice.cs, self.lattice.noOfDirections,
            self.fields.boundaryNode, self.fields.procBoundary, self.size,
            self.forcingScheme.forceFunc_vel,
            self.forcingScheme.forceCoeffVel, self.transport.rho_l,
            self.transport.rho_g, self.transport.phi_g,
            self.precision
        ]

        self.computeFieldsArgsPhase = [
            self.mesh.Nx, self.mesh.Ny, self.fields.h_new,
            self.fields.phi, self.fields.solid, self.fields.procBoundary,
            self.fields.boundaryNode, self.lattice.noOfDirections,
            self.precision
        ]

        self.computeGradLapPhiArgs = [
            self.mesh.Nx, self.mesh.Ny, self.fields.phi, self.fields.gradPhi,
            self.fields.normalPhi, self.fields.lapPhi, self.fields.solid,
            self.fields.procBoundary, self.fields.boundaryNode,
            self.lattice.cs_2, self.lattice.c, self.lattice.w,
            self.lattice.noOfDirections, self.size
        ]

        self.forceFluidArgs = [self.fields.f_new, self.fields.f_eq,
                               self.fields.forceField, self.fields.rho,
                               self.fields.p,  self.fields.phi,
                               self.fields.solid, self.fields.lapPhi,
                               self.fields.gradPhi, self.fields.stressTensor,
                               self.fields.procBoundary, self.fields.
                               boundaryNode, self.forcingScheme.gravity,
                               self.collisionScheme.preFactorFluid,
                               self.forcingScheme.forcingPreFactorFluid,
                               self.collisionScheme.constructOperatorFunc,
                               self.collisionScheme.collisionOperatorArgs,
                               self.collisionScheme.constructStressTensorFunc,
                               self.phaseField.computeViscFunc,
                               self.phaseField.beta, self.phaseField.kappa,
                               self.transport.mu_l, self.transport.mu_g,
                               self.transport.rho_l, self.transport.rho_g,
                               self.transport.phi_l, self.transport.phi_g,
                               self.lattice.cs, self.lattice.noOfDirections,
                               self.lattice.c, self.lattice.cs_2,
                               self.mesh.Nx, self.mesh.Ny
                               ]

        if size > 1:
            self.gatherArgs = [self.fields.u, self.fields.rho,
                               self.fields.solid, self.rank,
                               self.mpiParams.nProc_x,
                               self.mpiParams.nProc_y,
                               self.mesh.Nx_global, self.mesh.Ny_global,
                               self.mesh.Nx, self.mesh.Ny, self.precision]

            self.procBoundaryArgsPhase = [self.mesh.Nx, self.mesh.Ny,
                                          self.fields.h,
                                          self.fields.h_send_topBottom,
                                          self.fields.h_recv_topBottom,
                                          self.fields.h_send_leftRight,
                                          self.fields.h_recv_leftRight,
                                          self.mpiParams.nx, self.mpiParams.ny,
                                          self.mpiParams.nProc_x,
                                          self.mpiParams.nProc_y]
            self.procBoundaryArgsFluid = [self.mesh.Nx, self.mesh.Ny,
                                          self.fields.f,
                                          self.fields.f_send_topBottom,
                                          self.fields.f_recv_topBottom,
                                          self.fields.f_send_leftRight,
                                          self.fields.f_recv_leftRight,
                                          self.mpiParams.nx, self.mpiParams.ny,
                                          self.mpiParams.nProc_x,
                                          self.mpiParams.nProc_y]
            self.commPhiArgs = [self.mesh.Nx, self.mesh.Ny,
                                self.fields.phi,
                                self.fields.phi_send_topBottom,
                                self.fields.phi_recv_topBottom,
                                self.fields.phi_send_leftRight,
                                self.fields.phi_recv_leftRight,
                                self.mpiParams.nx, self.mpiParams.ny,
                                self.mpiParams.nProc_x,
                                self.mpiParams.nProc_y]

        self.boundary.setBoundaryArgs(self.lattice, self.mesh, self.fields,
                                      self.collisionScheme)
        self.initializePopulations(u_initial='u')

    def setupParallel_cpu(self, parallel):
        self.boundary.setupBoundary_cpu(parallel)
        self.phaseField.setupParallel_cpu(parallel)

    def initializePopulations(self, u_initial='u_initial'):
        if u_initial == 'u':
            u = self.fields.u
        elif u_initial == 'u_initial':
            u = self.fields.u_initialize
        initializePopulations(
            self.mesh.Nx, self.mesh.Ny, self.fields.h_eq, self.fields.h,
            self.fields.h_new, u, self.fields.phi,
            self.lattice.noOfDirections, self.equilibriumFuncPhase,
            self.equilibriumArgsPhase,
            self.fields.procBoundary, self.fields.boundaryNode
        )
        initializePopulations(
            self.mesh.Nx, self.mesh.Ny, self.fields.f_eq, self.fields.f,
            self.fields.f_new, u, self.fields.p,
            self.lattice.noOfDirections, self.equilibriumFuncFluid,
            self.equilibriumArgsFluid,
            self.fields.procBoundary, self.fields.boundaryNode
        )
