import os
import numpy as np
import numba


import mesh
import lattice
import boundary
import equilibrium
import schemeLB


class simulation:
    def __init__(self):
        print('Reading simulation parameters...\n', flush=True)
        try:
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
        self.mesh = mesh.createMesh(meshDict, obstacle)
        print('Reading mesh info and creating mesh done!\n', flush=True)
        print('Setting lattice structure...', flush=True)
        self.lattice = lattice.createLattice(latticeDict)
        print('Setting lattice structure done!\n', flush=True)
        print('Setting collision scheme and equilibrium model...',
              flush=True)
        self.equilibrium = equilibrium.equilibrium(self.mesh, self.lattice)
        self.collisionScheme = schemeLB.collisionScheme(self.lattice,
                                                        collisionDict,
                                                        self.equilibrium)
        self.schemeLog()
        print('Setting collision scheme and equilibrium model done!\n',
              flush=True)
        print('Initializing elements...', flush=True)
        self.elements = mesh.createElements(self.lattice, self.mesh,
                                            self.U_initial, self.rho_initial)
        print('Initializing elements done!\n')

        # Initialize boundary object
        print('Reading boundary conditions...')
        self.boundary = boundary.boundary(boundaryDict)
        self.boundary.readBoundaryDict()
        self.boundary.calculateBoundaryIndices(self.mesh.delX)
        self.boundary.initializeBoundaryElements(self.elements)
        self.writeDomainLog()
        print('Reading boundary conditions done...\n')

        # initialize functions
        self.equilibriumFunc = self.collisionScheme.equilibriumFunc
        self.collisionFunc = self.collisionScheme.collisionFunc
        self.setBoundaryFunc = self.boundary.setBoundary

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

    def writeDomainLog(self):
        meshFile = open('log_domain', 'w')
        meshFile.write('Domain Information...\n')
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
                meshFile.write('\tno.of elements : ' +
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


@numba.njit
def initializePopulations(elements, mesh, equilibriumFunc):
    for ind in range(mesh.Nx * mesh.Ny):
        elements[ind].computeEquilibrium(equilibriumFunc)
        elements[ind].f[:] = elements.f_eq[:]
        elements[ind].f[:] = elements.f_new[:]
