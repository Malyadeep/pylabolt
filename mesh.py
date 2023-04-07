import numpy as np
import numba
import os


spec = [
    ('delX', numba.float64),
    ('Nx', numba.int64),
    ('Ny', numba.int64),
]


@numba.experimental.jitclass(spec)
class mesh:
    def __init__(self, Nx, Ny, delX):
        self.Nx = Nx
        self.Ny = Ny
        self.delX = delX


def createMesh(meshDict, obstacle):
    [Nx, Ny] = meshDict['grid']
    boundingBox = np.array(meshDict['boundingBox'])
    delX1 = (boundingBox[1, 0] - boundingBox[0, 0])/(Nx - 1)
    delX2 = (boundingBox[1, 1] - boundingBox[0, 1])/(Ny - 1)
    if delX1 == delX2:
        meshObj = mesh(Nx, Ny, delX1)
    else:
        print("Error: Only Square lattices are supported")
        os._exit(1)
    return meshObj
