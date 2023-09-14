import numpy as np
import os


class mesh:
    def __init__(self, Nx, Ny, precision):
        self.Nx = Nx
        self.Ny = Ny
        self.delX = 1
        self.delX_ghost = 1
        self.delY_ghost = 1
        self.Nx_global = Nx
        self.Ny_global = Ny
        self.x = np.zeros(self.Nx_global * self.Ny_global, dtype=precision)
        self.y = np.zeros(self.Nx_global * self.Ny_global, dtype=precision)


def createMesh(meshDict, precision, rank):
    [Nx, Ny] = meshDict['grid']
    boundingBox = np.array(meshDict['boundingBox'])
    if Nx > 1 and Ny > 1:
        if (Nx - 1 != (boundingBox[1, 0] - boundingBox[0, 0]) and
                Ny - 1 != (boundingBox[1, 1] - boundingBox[0, 1])):
            if rank == 0:
                print("ERROR! Inconsistent grid and domain definition!")
            os._exit(1)
        else:
            meshObj = mesh(Nx + 2, Ny + 2, precision)
    elif Nx == 1 and Ny > 1:
        if (Ny - 1 != (boundingBox[1, 1] - boundingBox[0, 1])):
            if rank == 0:
                print("ERROR! Inconsistent grid and domain definition!")
            os._exit(1)
        else:
            meshObj = mesh(Nx, Ny + 2, precision)
    elif Ny == 1 and Nx > 1:
        if (Nx - 1 != (boundingBox[1, 0] - boundingBox[0, 0])):
            if rank == 0:
                print("ERROR! Inconsistent grid and domain definition!")
            os._exit(1)
        else:
            meshObj = mesh(Nx + 2, Ny, precision)
    elif Nx == 1 and Ny == 1:
        if rank == 0:
            print("ERROR! prescribed domain is a point!!")
        os._exit(1)
    for ind in range(meshObj.Nx * meshObj.Ny):
        meshObj.x[ind] = int(ind / meshObj.Ny) * meshObj.delX
        meshObj.y[ind] = int(ind % meshObj.Ny) * meshObj.delX
    return meshObj
