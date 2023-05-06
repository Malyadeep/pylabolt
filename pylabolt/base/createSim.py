import os
import sys
import numpy as np
import numba


from pylabolt.base import mesh, lattice, boundary, schemeLB, fields
from pylabolt.parallel.MPI_decompose import decompose, distributeSolid_mpi


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
                                    latticeDict, meshDict, obstacle,
                                    internalFields)
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
        try:
            self.u_initial = internalFields['u']
            self.v_initial = internalFields['v']
            self.U_initial = np.array([self.u_initial, self.v_initial],
                                      dtype=self.precision)
            self.rho_initial = np.float64(internalFields['rho'])
        except KeyError as e:
            if rank == 0:
                print('ERROR! Keyword ' + str(e) +
                      ' missing in internalFields')

        if rank == 0:
            print('Reading mesh info and creating mesh...', flush=True)
        self.mesh = mesh.createMesh(meshDict, self.precision)
        if rank == 0:
            print('Reading mesh info and creating mesh done!\n', flush=True)
        if size > 1:
            self.mpiParams = decompose(self.mesh, self.rank, self.size, comm)
        if rank == 0:
            print('Setting lattice structure...', flush=True)
        self.lattice = lattice.createLattice(latticeDict, self.precision)
        if rank == 0:
            print('Setting lattice structure done!\n', flush=True)
            print('Setting collision scheme and equilibrium model...',
                  flush=True)
        self.collisionScheme = schemeLB.collisionScheme(self.lattice,
                                                        collisionDict,
                                                        parallelization)
        if rank == 0:
            self.schemeLog()
            print('Setting collision scheme and equilibrium model done!\n',
                  flush=True)
            print('Initializing fields...', flush=True)
        self.fields = fields.fields(self.mesh, self.lattice,
                                    self.U_initial, self.rho_initial,
                                    self.precision, size)
        solid = fields.setObstacle(obstacle, self.mesh)
        if size == 1:
            fields.solid = solid
        else:
            distributeSolid_mpi(solid, self.fields, self.mpiParams, self.mesh,
                                self.rank, self.size, comm)
        if rank == 0:
            print('Initializing fields done!\n')
            print('Reading boundary conditions...')
        self.boundary = boundary.boundary(boundaryDict)
        if rank == 0:
            self.boundary.readBoundaryDict()
            self.boundary.initializeBoundary(self.lattice, self.mesh,
                                             self.fields)
            # self.boundary.details()
        if rank == 0:
            self.writeDomainLog(meshDict)
            print('Reading boundary conditions done...\n')

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
            self.collisionScheme.preFactor,
            self.collisionScheme.equilibriumArgs,
            self.fields.procBoundary
        )

        self.streamArgs = (
            self.mesh.Nx, self.mesh.Ny, self.fields.f, self.fields.f_new,
            self.lattice.c, self.lattice.noOfDirections,
            self.lattice.invList, self.fields.solid, self.size
        )

        self.computeFieldsArgs = (
            self.mesh.Nx, self.mesh.Ny, self.fields.f_new,
            self.fields.u, self.fields.rho, self.fields.solid,
            self.lattice.c, self.lattice.noOfDirections,
            self.fields.procBoundary, self.size
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
                         str(self.collisionScheme.tau) + '\n')
        schemeFile.write('\tcollision scheme : ' +
                         str(self.collisionScheme.equilibriumModel) + '\n')
        schemeFile.write('\nLattice Information...\n')
        schemeFile.write('\tLattice : ' +
                         str(self.lattice.latticeType) + '\n')
        schemeFile.write('\tdeltaX in lattice units : ' +
                         str(self.lattice.deltaX) + '\n')
        schemeFile.write('\tdeltaT in lattice units : ' +
                         str(self.lattice.deltaT) + '\n')
        schemeFile.close()
