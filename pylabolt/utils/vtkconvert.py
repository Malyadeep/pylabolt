import numpy as np
import vtk
import os
import sys


from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkIntArray
from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid


def toVTK(options, time):
    currentDir = os.getcwd()
    sys.path.append(currentDir)
    if options == 'last':
        try:
            from simulation import controlDict
            if int(controlDict['endTime']) < int(controlDict['saveInterval']):
                raise Warning("'endTime' in controlDict is greater than" +
                              " 'saveInterval'! Execution may fail!")
            time = int(controlDict['endTime']/controlDict['saveInterval']) * \
                controlDict['saveInterval']
            inputPath = "output/" + str(time) + "/"
            if not os.path.isdir('output/VTK'):
                os.makedirs('output/VTK')
            outputPath = "output/VTK/"
            vtkConverter(inputPath, outputPath, time)
        except ImportError as e:
            print('FATAL ERROR!')
            print(str(e))
            print('Aborting....')
            os._exit(1)
        except KeyError as e:
            print('FATAL ERROR!')
            print(str(e))
            print('Aborting....')
            os._exit(1)
        except FileNotFoundError as e:
            print('FATAL ERROR!')
            print(str(e))
            print('Aborting....')
            os._exit(1)
    elif options == 'all':
        try:
            if not os.path.isdir('output/VTK'):
                os.makedirs('output/VTK')
            for (root, dirs, files) in os.walk('output/'):
                if str(root) == 'output/':
                    ind = dirs.index('VTK')
                    dirs.pop(ind)
                    dirs = np.array(list(map(int, dirs)), dtype=np.int64)
                    dirs = np.sort(dirs)
                    for currentDir in dirs:
                        inputPath = "output/" + str(currentDir) + "/"
                        outputPath = "output/VTK/"
                        vtkConverter(inputPath, outputPath, currentDir)
        except ImportError as e:
            print('FATAL ERROR!')
            print(str(e))
            print('Aborting....')
            os._exit(1)
        except KeyError as e:
            print('FATAL ERROR!')
            print(str(e))
            print('Aborting....')
            os._exit(1)
        except FileNotFoundError as e:
            print('FATAL ERROR!')
            print(str(e))
            print('Aborting....')
            os._exit(1)
    elif options == 'time':
        try:
            inputPath = "output/" + str(time) + "/"
            if not os.path.isdir('output/VTK'):
                os.makedirs('output/VTK')
            outputPath = "output/VTK/"
            vtkConverter(inputPath, outputPath, time)
        except FileNotFoundError as e:
            print('FATAL ERROR!')
            print(str(e))
            print('Aborting....')
            os._exit(1)


def vtkConverter(inputPath, outputPath, time):
    u, rho = False, False
    p = False
    phi, T = False, False
    print('----------------------------------------------')
    print('timeStep = ' + str(time) + '\n')
    try:
        pointsData = np.loadtxt(inputPath + "points.dat")
        print("Reading points field")
    except FileNotFoundError:
        print("ERROR! points information not present!")
        os._exit(1)
    try:
        uData = np.loadtxt(inputPath + "u.dat")
        print("Reading velocity field")
        u = True
        velArray = vtkDoubleArray()
        velArray.SetName('U')
        velArray.SetNumberOfComponents(3)
    except FileNotFoundError:
        pass
    try:
        rhoData = np.loadtxt(inputPath + "rho.dat")
        print("Reading density field")
        rho = True
        rhoArray = vtkDoubleArray()
        rhoArray.SetName('rho')
    except FileNotFoundError:
        pass
    try:
        pData = np.loadtxt(inputPath + "p.dat")
        print("Reading pressure field")
        p = True
        pArray = vtkDoubleArray()
        pArray.SetName('p')
    except FileNotFoundError:
        pass
    try:
        phiData = np.loadtxt(inputPath + "phi.dat")
        print("Reading phase field")
        phi = True
        phiArray = vtkDoubleArray()
        phiArray.SetName('phi')
    except FileNotFoundError:
        pass
    try:
        TData = np.loadtxt(inputPath + "T.dat")
        print("Reading temperature field")
        T = True
        TArray = vtkDoubleArray()
        TArray.SetName('T')
    except FileNotFoundError:
        pass
    try:
        solidData = np.loadtxt(inputPath + "solid.dat")
        print("Reading obstacle field")
        solArray = vtkIntArray()
        solArray.SetName('obstacle')
    except FileNotFoundError:
        pass
    from simulation import meshDict
    Nx = int(meshDict['grid'][0])
    Ny = int(meshDict['grid'][1])
    if (Nx * Ny) != pointsData.shape[0]:
        print("ERROR! Grid size in 'meshDict' doesn't match with output!")
        os._exit(1)

    # Create a grid
    grid = vtkRectilinearGrid()
    grid.SetDimensions(Nx, Ny, 1)

    # Write out points
    pointArray = vtkIntArray()
    pointArray.SetName('Point_ID')
    xArray = vtkDoubleArray()
    yArray = vtkDoubleArray()
    zArray = vtkDoubleArray()

    for i in range(Nx):
        xArray.InsertNextValue(pointsData[i * Ny, 1])

    for i in range(Ny):
        yArray.InsertNextValue(pointsData[i, 2])

    zArray = vtkDoubleArray()
    zArray.InsertNextValue(0.0)

    for j in range(Ny):
        for i in range(Nx):
            ind = i * Ny + j
            xArray.InsertNextValue(pointsData[ind, 1])
            yArray.InsertNextValue(pointsData[ind, 2])
            zArray.InsertNextValue(0.0)
            pointArray.InsertNextValue(int(pointsData[ind, 0]))
    grid.SetXCoordinates(xArray)
    grid.SetYCoordinates(yArray)
    grid.SetZCoordinates(zArray)
    grid.GetPointData().AddArray(pointArray)

    # Write Fields
    for j in range(Ny):
        for i in range(Nx):
            ind = i * Ny + j
            if u is True:
                velArray.InsertNextTuple3(uData[ind, 0], uData[ind, 1], 0)
            if rho is True:
                rhoArray.InsertNextValue(rhoData[ind])
            if p is True:
                pArray.InsertNextValue(pData[ind])
            if phi is True:
                phiArray.InsertNextValue(phiData[ind])
            if T is True:
                TArray.InsertNextValue(TData[ind])
            solArray.InsertNextValue(np.int32(solidData[ind]))
    if u is True:
        grid.GetPointData().AddArray(velArray)
    if rho is True:
        grid.GetPointData().AddArray(rhoArray)
    if p is True:
        grid.GetPointData().AddArray(pArray)
    if phi is True:
        grid.GetPointData().AddArray(phiArray)
    if T is True:
        grid.GetPointData().AddArray(TArray)
    grid.GetPointData().AddArray(solArray)

    print('\nThere are', grid.GetNumberOfPoints(), 'points.')
    print('There are', grid.GetNumberOfCells(), 'cells.')
    print('----------------------------------------------\n\n')

    writer = vtk.vtkRectilinearGridWriter()
    writer.SetFileVersion(42)
    writer.SetFileName(outputPath + "output_" + str(time) + ".vtk")
    writer.SetInputData(grid)
    writer.Write()


if __name__ == '__main__':
    vtkConverter()
