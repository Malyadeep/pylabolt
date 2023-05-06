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
    data = np.loadtxt(inputPath + "fields.dat")
    nx = len(np.unique(data[:, 1]))
    ny = len(np.unique(data[:, 2]))
    point_id = np.array(data[:, 0], dtype=np.int64)

    # Create a grid
    grid = vtkRectilinearGrid()
    grid.SetDimensions(nx, ny, 1)

    xArray = vtkDoubleArray()
    for i in range(nx):
        xArray.InsertNextValue(data[i*ny, 1])

    yArray = vtkDoubleArray()
    for i in range(ny):
        yArray.InsertNextValue(data[i, 2])

    zArray = vtkDoubleArray()
    zArray.InsertNextValue(0.0)

    # Create vector/scalar arrays
    pointArray = vtkIntArray()
    pointArray.SetName('Points_ID')
    for ind in range(nx * ny):
        pointArray.InsertNextValue(point_id[ind])

    velArray = vtkDoubleArray()
    velArray.SetName('Velocity')
    velArray.SetNumberOfComponents(3)
    for j in range(ny):
        for i in range(nx):
            velArray.InsertNextTuple3(data[j+i*ny, 4], data[j+i*ny, 5], 0)

    rhoArray = vtkDoubleArray()
    rhoArray.SetName('Density')
    for j in range(ny):
        for i in range(nx):
            rhoArray.InsertNextValue(data[j+i*ny, 3])

    solArray = vtkDoubleArray()
    solArray.SetName('obstacle')
    for j in range(ny):
        for i in range(nx):
            solArray.InsertNextValue(data[j+i*ny, 6])

    grid.SetXCoordinates(xArray)
    grid.SetYCoordinates(yArray)
    grid.SetZCoordinates(zArray)
    grid.GetPointData().AddArray(pointArray)
    grid.GetPointData().AddArray(velArray)
    grid.GetPointData().AddArray(rhoArray)
    grid.GetPointData().AddArray(solArray)
    print('timeStep = ' + str(time))
    print('There are', grid.GetNumberOfPoints(), 'points.')
    print('There are', grid.GetNumberOfCells(), 'cells. \n')

    writer = vtk.vtkRectilinearGridWriter()
    writer.SetFileVersion(42)
    writer.SetFileName(outputPath + "output_" + str(time) + ".vtk")
    writer.SetInputData(grid)
    writer.Write()


if __name__ == '__main__':
    vtkConverter()
