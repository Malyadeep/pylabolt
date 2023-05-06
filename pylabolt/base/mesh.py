import numpy as np
import os


class mesh:
    def __init__(self, Nx, Ny, delX, precision):
        self.Nx = Nx
        self.Ny = Ny
        self.delX = delX
        self.Nx_global = Nx
        self.Ny_global = Ny
        self.x = np.zeros(self.Nx_global * self.Ny_global, dtype=precision)
        self.y = np.zeros(self.Nx_global * self.Ny_global, dtype=precision)


def createMesh(meshDict, precision):
    [Nx, Ny] = meshDict['grid']
    boundingBox = np.array(meshDict['boundingBox'])
    delX1 = (boundingBox[1, 0] - boundingBox[0, 0])/(Nx - 1)
    delX2 = (boundingBox[1, 1] - boundingBox[0, 1])/(Ny - 1)
    if delX1 == delX2:
        meshObj = mesh(Nx, Ny, delX1, precision)
    else:
        print("Error: Only Square lattices are supported")
        os._exit(1)
    for ind in range(meshObj.Nx * meshObj.Ny):
        meshObj.x[ind] = int(ind / meshObj.Ny) * meshObj.delX
        meshObj.y[ind] = int(ind % meshObj.Ny) * meshObj.delX
    return meshObj
