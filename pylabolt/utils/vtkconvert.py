import numpy as np
import vtk
import h5py
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
            inputFile = h5py.File("output.h5", "r")
            if not os.path.isdir('VTK'):
                os.makedirs('VTK')
            outputPath = "VTK/"
            vtkConverter(inputFile, outputPath, time)
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
            if not os.path.isdir('VTK'):
                os.makedirs('VTK')
            outputPath = "VTK/"
            inputFile = h5py.File("output.h5", "r")
            endTime = inputFile["parameters"].attrs["endTime"]
            saveInterval = inputFile["parameters"].attrs["saveInterval"]
            times = np.linspace(0, endTime, endTime//saveInterval + 1)
            for t, time in enumerate(times):
                vtkConverter(inputFile, outputPath, int(time))
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
            inputFile = h5py.File("output.h5", "r")
            if not os.path.isdir('VTK'):
                os.makedirs('VTK')
            outputPath = "VTK/"
            vtkConverter(inputFile, outputPath, time)
        except FileNotFoundError as e:
            print('FATAL ERROR!')
            print(str(e))
            print('Aborting....')
            os._exit(1)


def vtkConverter(inputFile, outputPath, time):
    u, rho = False, False
    p = False
    phi, T = False, False
    print('----------------------------------------------')
    print('timeStep = ' + str(time) + '\n')
    try:
        time_group = inputFile["fields/t_" + str(time)]
    except KeyError:
        print("No field data found for t = " + str(time))
        print('----------------------------------------------\n\n')
        return
    try:
        uData = time_group["u"]
        print("Reading velocity field")
        u = True
        velArray = vtkDoubleArray()
        velArray.SetName('U')
        velArray.SetNumberOfComponents(3)
    except KeyError:
        print("No velocity field found")
    try:
        rhoData = time_group["rho"]
        print("Reading density field")
        rho = True
        rhoArray = vtkDoubleArray()
        rhoArray.SetName('rho')
    except KeyError:
        print("No density field found")
    try:
        pData = time_group["p"]
        print("Reading pressure field")
        p = True
        pArray = vtkDoubleArray()
        pArray.SetName('p')
    except KeyError:
        print("No pressure field found")
    try:
        phiData = time_group["phi"]
        print("Reading phase field")
        phi = True
        phiArray = vtkDoubleArray()
        phiArray.SetName('phi')
    except KeyError:
        print("No phase field found")
    try:
        TData = time_group["T"]
        print("Reading temperature field")
        T = True
        TArray = vtkDoubleArray()
        TArray.SetName('T')
    except KeyError:
        print("No temperature field found")
    try:
        solidData = time_group["obstacle"]
        print("Reading obstacle field")
        solArray = vtkIntArray()
        solArray.SetName('obstacle')
    except KeyError:
        print("No obstacle field found")
    from simulation import meshDict
    Nx = int(meshDict['grid'][0])
    Ny = int(meshDict['grid'][1])
    Nx_out = inputFile["parameters"].attrs["Nx"]
    Ny_out = inputFile["parameters"].attrs["Ny"]

    if (Nx * Ny) != (Nx_out * Ny_out):
        print("ERROR! Grid size in 'meshDict' doesn't match with output!")
        os._exit(1)

    """ Create a grid """
    grid = vtkRectilinearGrid()
    grid.SetDimensions(Nx, Ny, 1)

    """ Write out points """
    pointArray = vtkIntArray()
    pointArray.SetName('Point_ID')
    xArray = vtkDoubleArray()
    yArray = vtkDoubleArray()
    zArray = vtkDoubleArray()

    for i in range(Nx):
        xArray.InsertNextValue(i)

    for j in range(Ny):
        yArray.InsertNextValue(j)

    zArray = vtkDoubleArray()
    zArray.InsertNextValue(0.0)

    for j in range(Ny):
        for i in range(Nx):
            ind = i * Ny + j
            xArray.InsertNextValue(i)
            yArray.InsertNextValue(j)
            zArray.InsertNextValue(0.0)
            pointArray.InsertNextValue(ind)
    grid.SetXCoordinates(xArray)
    grid.SetYCoordinates(yArray)
    grid.SetZCoordinates(zArray)
    grid.GetPointData().AddArray(pointArray)

    """ Write Fields """
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
            solArray.InsertNextValue(np.int32(solidData[ind, 0]))
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
