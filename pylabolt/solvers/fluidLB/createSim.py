import os
import sys
import numpy as np
import numba
from copy import deepcopy


from pylabolt.base import (mesh, lattice, boundary, schemeLB,
                           obstacle)
from pylabolt.solvers.fluidLB import (initFields, fields)
from pylabolt.parallel.MPI_decompose import (decompose, distributeSolid_mpi,
                                             distributeInitialFields_mpi)
from pylabolt.utils.options import options


@numba.njit
def initializePopulations(Nx, Ny, f_eq, f, f_new, u, rho,
                          noOfDirections, equilibriumFunc,
                          equilibriumArgs, procBoundary):
    for ind in range(Nx * Ny):
        if procBoundary[ind] != 1:
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
                                    latticeDict, meshDict, internalFields)
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
            self.writeControlLog()
            print('Setting control parameters done!\n', flush=True)

        if rank == 0:
            print('Reading mesh info and creating mesh...', flush=True)
        self.mesh = mesh.createMesh(meshDict, self.precision)
        if rank == 0:
            print('Reading mesh info and creating mesh done!\n', flush=True)

        if rank == 0:
            print('Setting lattice structure...', flush=True)
        self.lattice = lattice.createLattice(latticeDict, self.precision)
        if rank == 0:
            print('Setting lattice structure done!\n', flush=True)
            print('Setting collision scheme and equilibrium model...',
                  flush=True)
        self.collisionScheme = schemeLB.collisionScheme(self.lattice,
                                                        collisionDict,
                                                        parallelization,
                                                        self.rank,
                                                        self.precision)
        if rank == 0:
            print('Setting collision scheme and equilibrium model done!\n',
                  flush=True)

        if rank == 0:
            print('Setting forcing scheme...', flush=True)
        self.forcingScheme = schemeLB.forcingScheme(self.precision,
                                                    self.lattice)
        self.forcingScheme.setForcingScheme(self.lattice,
                                            self.collisionScheme,
                                            self.rank,
                                            self.precision)
        if rank == 0:
            self.schemeLog()
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
            print('Reading option done!\n', flush=True)

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
                                          self.rank, size)
        comm.Barrier()
        if size > 1:
            distributeInitialFields_mpi(initialFields_temp, self.initialFields,
                                        self.mpiParams, self.mesh, self.rank,
                                        self.size, comm, self.precision)
        else:
            self.initialFields = initialFields_temp
        self.fields = fields.fields(self.mesh, self.lattice,
                                    self.initialFields,
                                    self.precision, size, rank)
        if (self.options.computeForces is True or self.options.computeTorque
                is True and rank == 0):
            self.obstacle.computeFluidNb(solid, self.mesh, self.lattice,
                                         self.size)
        # if rank == 0:
        #     self.obstacle.details()
        obstacleTemp = deepcopy(self.obstacle)
        if size == 1 and solid is not None:
            self.fields.solid = solid
        elif size > 1 and solid is not None:
            distributeSolid_mpi(solid, self.obstacle,
                                self.fields, self.mpiParams, self.mesh,
                                self.precision, self.rank, self.size, comm)
        if rank == 0:
            print('Initializing fields done!\n', flush=True)

        if rank == 0:
            print('Reading boundary conditions...', flush=True)
        self.boundary = boundary.boundary(boundaryDict)
        if rank == 0:
            self.boundary.readBoundaryDict(self.rank)
            self.boundary.initializeBoundary(self.lattice, self.mesh,
                                             self.fields, self.precision)
            # self.boundary.details()
        if rank == 0:
            self.writeDomainLog(meshDict)
            print('Reading boundary conditions done...\n', flush=True)
        if (self.options.computeForces is True or self.options.computeTorque
                is True):
            if rank == 0:
                self.options.gatherObstacleNodes(obstacleTemp)
                self.options.gatherBoundaryNodes(self.boundary)
                # self.options.details(self.rank, self.mesh, self.fields.solid,
                #                      self.fields.u, flag='all')
        # if rank == 3:
        #     self.obstacle.details()
        #     obstacleTemp.details()
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
            self.fields.solid, self.collisionFunc, self.equilibriumFunc,
            self.collisionScheme.preFactor, self.collisionScheme.
            equilibriumArgs, self.fields.procBoundary, self.forcingScheme.
            forceFunc_force, self.forcingScheme.forceArgs_force,
            self.forcingScheme.forcingPreFactor,
            self.lattice.noOfDirections, self.precision
        )

        self.streamArgs = (
            self.mesh.Nx, self.mesh.Ny, self.fields.f, self.fields.f_new,
            self.fields.solid, self.fields.rho, self.fields.u,
            self.fields.procBoundary, self.lattice.c, self.lattice.w,
            self.lattice.noOfDirections, self.collisionScheme.cs_2,
            self.lattice.invList, self.size
        )

        self.computeFieldsArgs = (
            self.mesh.Nx, self.mesh.Ny, self.fields.f_new,
            self.fields.u, self.fields.rho, self.fields.solid,
            self.lattice.c, self.lattice.noOfDirections,
            self.fields.procBoundary, self.size,
            self.forcingScheme.forceFunc_vel,
            self.forcingScheme.forceArgs_vel,
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

        initializePopulations(
            self.mesh.Nx, self.mesh.Ny, self.fields.f_eq, self.fields.f,
            self.fields.f_new, self.fields.u, self.fields.rho,
            self.lattice.noOfDirections, self.equilibriumFunc,
            self.equilibriumArgs,
            self.fields.procBoundary
        )

    def writeControlLog(self):
        controlFile = open('log_control', 'w')
        controlFile.write('Control parameters...\n')
        controlFile.write('\tstartTime = ' + str(self.startTime) + '\n')
        controlFile.write('\tendTime = ' + str(self.endTime) + '\n')
        controlFile.write('\tstdOutputInterval = ' +
                          str(self.stdOutputInterval) + '\n')
        controlFile.write('saveInterval = ' +
                          str(self.saveInterval) + '\n')
        controlFile.write('\tsaveStateInterval = ' +
                          str(self.saveStateInterval) + '\n')
        controlFile.write('\trelTolU = ' + str(self.relTolU) + '\n')
        controlFile.write('\trelTolV = ' + str(self.relTolV) + '\n')
        controlFile.write('\trelTolRho = ' + str(self.relTolRho) + '\n')
        controlFile.close()

    def writeDomainLog(self, meshDict):
        meshFile = open('log_domain', 'w')
        meshFile.write('Domain Information...\n')
        meshFile.write('\tBounding box : ' + str(meshDict['boundingBox'])
                       + '\n')
        meshFile.write('\tGrid points in x-direction : ' +
                       str(self.mesh.Nx_global) + '\n')
        meshFile.write('\tGrid points in y-direction : ' +
                       str(self.mesh.Ny_global) + '\n')
        meshFile.write('\tdelX : ' +
                       str(self.mesh.delX) + '\n')
        meshFile.write('\nBoundary information...\n')
        for itr, name in enumerate(self.boundary.nameList):
            meshFile.write('\n\tboundary name : ' + str(name) + '\n')
            meshFile.write('\tboundary type : ' +
                           str(self.boundary.boundaryType[itr]) + '\n')
            pointArray = np.array(self.boundary.points[name])
            for k in range(pointArray.shape[0]):
                meshFile.write('\tpoint ' + str(k) + ': ' +
                               str(pointArray[k, 0]) + ', ' +
                               str(pointArray[k, 1]) + '\n')
                temp = self.boundary.boundaryIndices[k + itr, 1] -\
                    self.boundary.boundaryIndices[k + itr, 0]
                meshFile.write('\tno.of fields : ' +
                               str(temp[0] * temp[1]) + '\n')
        meshFile.close()

    def schemeLog(self):
        schemeFile = open('log_scheme', 'w')
        schemeFile.write('Scheme Information...\n')
        schemeFile.write('\tcollision scheme : ' +
                         str(self.collisionScheme.collisionModel) + '\n')
        schemeFile.write('\trelaxation time : ' +
                         str(self.collisionScheme.nu) + '\n')
        schemeFile.write('\tcollision scheme : ' +
                         str(self.collisionScheme.equilibriumModel) + '\n')
        schemeFile.write('\nLattice Information...\n')
        schemeFile.write('\tLattice : ' +
                         str(self.lattice.latticeType) + '\n')
        schemeFile.write('\tdeltaX in lattice units : ' +
                         str(self.lattice.deltaX) + '\n')
        schemeFile.write('\tdeltaT in lattice units : ' +
                         str(self.lattice.deltaT) + '\n')
        schemeFile.write('\nForcing Information...\n')
        try:
            schemeFile.write('\tModel : ' + self.forcingScheme.
                             forcingModel + '\n')
            schemeFile.write('\tValue : ' + str(self.forcingScheme.
                             forcingValue) + '\n')
        except AttributeError:
            schemeFile.write('\tNo Forcing scheme selected\n')
        schemeFile.close()
