import numba
import os
import collisionModels
import equilibriumModels


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
            if collisionDict['equilibrium'] == 'secondOrder':
                self.equilibriumFunc = equilibriumModels.secondOrder
                self.equilibriumArgs = (self.cs_2, self.cs_4,
                                        self.c, self.w)
            else:
                print("ERROR! Unsupported equilibrium model : " +
                      collisionDict['equilibrium'])
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
def stream(elements, lattice, mesh):
    Nx = mesh.Nx
    Ny = mesh.Ny
    for i in range(Nx):
        for j in range(Ny):
            ind = int(i * Ny + j)
            if elements[ind].nodeType != 'o':
                for k in range(1, lattice.c.shape[0]):
                    i_old = (i - int(lattice.c[k, 0])
                             + Nx) % Nx
                    j_old = (j - int(lattice.c[k, 1])
                             + Ny) % Ny
                    elements[ind].f_new[k] = elements[i_old * Ny + j_old].f[k]
