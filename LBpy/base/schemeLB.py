import numba
import os
from LBpy.base.models import collisionModels, equilibriumModels


class collisionScheme:
    def __init__(self, lattice, collisionDict):
        try:
            self.deltaX = lattice.deltaX
            self.deltaT = lattice.deltaT
            self.cs_2 = 1/(lattice.cs*lattice.cs)
            self.cs_4 = self.cs_2/(lattice.cs*lattice.cs)
            self.c = lattice.c
            self.w = lattice.w
            if collisionDict['model'] == 'BGK':
                self.collisionFunc = collisionModels.BGK
            else:
                print("ERROR! Unsupported collision model : " +
                      collisionDict['model'])
            self.collisionModel = collisionDict['model']
            if collisionDict['equilibrium'] == 'firstOrder':
                self.equilibriumFunc = equilibriumModels.firstOrder
                self.equilibriumArgs = (self.cs_2, self.c, self.w)
            elif collisionDict['equilibrium'] == 'secondOrder':
                self.equilibriumFunc = equilibriumModels.secondOrder
                self.equilibriumArgs = (self.cs_2, self.cs_4,
                                        self.c, self.w)
            else:
                print("ERROR! Unsupported equilibrium model : " +
                      collisionDict['equilibrium'])
                os._exit(1)
            self.equilibriumModel = collisionDict['equilibrium']
        except KeyError as e:
            print("ERROR! Keyword: " + str(e) + " missing in 'latticeDict'")
            os._exit(1)
        try:
            self.tau = collisionDict['tau']
        except KeyError as e:
            print("ERROR! Keyword: " + str(e) + " missing in 'collisionDict'")
            os._exit(1)
        self.preFactor = self.deltaT/self.tau


@numba.njit
def stream(fields, lattice, mesh):
    Nx = mesh.Nx
    Ny = mesh.Ny
    for i in range(Nx):
        for j in range(Ny):
            ind = int(i * Ny + j)
            for k in range(lattice.noOfDirections):
                i_old = (i - int(lattice.c[k, 0])
                         + Nx) % Nx
                j_old = (j - int(lattice.c[k, 1])
                         + Ny) % Ny
                if fields.nodeType[i_old * Ny + j_old] != 3:
                    fields.f_new[ind, k] = fields.f[i_old * Ny + j_old, k]
