import os


import mesh
import lattice
import boundary
import equilibrium
import schemeConfig


class simulation:
    def __init__(self):
        try:
            from simulation import (controlDict, boundaryDict, collisionDict,
                                    latticeDict, meshDict, obstacle)
        except ImportError as e:
            print('FATAL ERROR!')
            print(e)
            os._exit(1)

        try:
            self.startTime = controlDict['startTime']
            self.endTime = controlDict['endTime']
        except KeyError as e:
            print('ERROR! Keyword ' + str(e) + ' missing in controlDict')

        self.mesh = mesh.mesh(meshDict, obstacle)
        self.lattice = lattice.lattice(latticeDict)
        self.equilibrium = equilibrium.equilibrium(self.mesh, self.lattice)
        self.collisionScheme = schemeConfig.collisionScheme(self.lattice,
                                                            collisionDict)
        self.elements = self.mesh.createElements()

        # Initialize boundary object
        self.boundary = boundary.boundary(boundaryDict)
        self.boundary.readBoundaryDict()
        self.boundary.calculateBoundaryIndices(self.mesh.delX)

        # initialize functions
        self.equilibriumFunc = self.collisionScheme.equilibriumFunc
        self.collisionFunc = self.collisionScheme.collisionFunc
        self.setBoundaryFunc = self.boundary.setBoundary
        self.propagation = self.schemeConfig.stream
