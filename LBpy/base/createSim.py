import os
import sys
import numpy as np
import numba


from LBpy.base import mesh, lattice, boundary, schemeLB, fields


@numba.njit
def initializePopulations(Nx, Ny, f_eq, f, f_new, u, rho,
                          noOfDirections, equilibriumFunc,
                          equilibriumArgs):
    for ind in range(Nx * Ny):
        equilibriumFunc(f_eq[ind, :], u[ind, :], rho[ind],
                        *equilibriumArgs)
        for k in range(noOfDirections):
            f[ind, k] = f_eq[ind, k]
            f_new[ind, k] = f_eq[ind, k]


class simulation:
    def __init__(self):
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
        except KeyError as e:
            print('ERROR! Keyword ' + str(e) + ' missing in controlDict')
        self.writeControlLog()
        print('Setting control parameters done!\n', flush=True)
        try:
            self.u_initial = internalFields['u']
            self.v_initial = internalFields['v']
            self.U_initial = np.array([self.u_initial, self.v_initial],
                                      dtype=np.float64)
            self.rho_initial = np.float64(internalFields['rho'])
        except KeyError as e:
            print('ERROR! Keyword ' + str(e) + ' missing in internalFields')

        print('Reading mesh info and creating mesh...', flush=True)
        self.mesh = mesh.createMesh(meshDict)
        print('Reading mesh info and creating mesh done!\n', flush=True)
        print('Setting lattice structure...', flush=True)
        self.lattice = lattice.createLattice(latticeDict)
        print('Setting lattice structure done!\n', flush=True)
        print('Setting collision scheme and equilibrium model...',
              flush=True)
        self.collisionScheme = schemeLB.collisionScheme(self.lattice,
                                                        collisionDict)
        self.schemeLog()
        print('Setting collision scheme and equilibrium model done!\n',
              flush=True)
        print('Initializing fields...', flush=True)
        self.fields = fields.fields(self.mesh, self.lattice,
                                    self.U_initial, self.rho_initial)
        fields.setObstacle(obstacle, fields)
        print('Initializing fields done!\n')
        print('Reading boundary conditions...')
        self.boundary = boundary.boundary(boundaryDict)
        self.boundary.readBoundaryDict()
        self.boundary.initializeBoundary(self.lattice, self.mesh,
                                         self.fields)
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
            self.collisionFunc, self.equilibriumFunc,
            self.collisionScheme.preFactor,
            self.collisionScheme.equilibriumArgs
        )

        self.streamArgs = (
            self.mesh.Nx, self.mesh.Ny, self.fields.f, self.fields.f_new,
            self.lattice.c, self.lattice.noOfDirections, self.fields.nodeType
        )

        self.computeFieldsArgs = (
            self.mesh.Nx, self.mesh.Ny, self.fields.f_new,
            self.fields.u, self.fields.rho,
            self.lattice.c, self.lattice.noOfDirections
        )

        initializePopulations(
            self.mesh.Nx, self.mesh.Ny, self.fields.f_eq, self.fields.f,
            self.fields.f_new, self.fields.u, self.fields.rho,
            self.lattice.noOfDirections, self.equilibriumFunc,
            self.equilibriumArgs
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
        meshFile.write('\tGrid points in x-direction ' + str(self.mesh.Nx)
                       + '\n')
        meshFile.write('\tGrid points in y-direction ' + str(self.mesh.Ny)
                       + '\n')
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
