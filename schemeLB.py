import numba
import os
import collisionModels


class collisionScheme:
    def __init__(self, lattice, collisionDict, equilibrium):
        try:
            self.deltaX = lattice.deltaX
            self.deltaT = lattice.deltaT
            if collisionDict['model'] == 'BGK':
                self.collisionFunc = collisionModels.BGK
            else:
                print("ERROR! Unsupported collision model : " +
                      collisionDict['model'])
            self.collisionModel = collisionDict['model']
            if collisionDict['equilibrium'] == 'firstOrder':
                self.equilibriumFunc = equilibrium.firstOrder
            if collisionDict['equilibrium'] == 'secondOrder':
                self.equilibriumFunc = equilibrium.secondOrder
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
            ind = int(i * Nx + j)
            if elements[ind].nodeType != 'o':
                for k in range(1, lattice.c.shape[0]):
                    i_old = (i - int(lattice.c[k, 0])
                             + Nx) % Nx
                    j_old = (j - int(lattice.c[k, 1])
                             + Ny) % Ny
                    elements[ind].f_new[k] = elements[i_old * Nx + j_old].f[k]
