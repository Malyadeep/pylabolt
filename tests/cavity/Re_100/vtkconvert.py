import numpy as np
import vtk


from vtkmodules.vtkCommonCore import vtkDoubleArray
from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid


def vtkConverter():
    data = np.loadtxt('fields.dat')
    nx = len(np.unique(data[:,1]))
    ny = len(np.unique(data[:,2]))
    point_id = data[:,0]
    nz = 1   

    # Create a grid
    grid = vtkRectilinearGrid()
    grid.SetDimensions(nx, ny, 1)

    xArray = vtkDoubleArray()
    for i in range(nx):
    	xArray.InsertNextValue(data[i*ny,1])
    

    yArray = vtkDoubleArray()
    for i in range(ny):
    	yArray.InsertNextValue(data[i,2])
    
    
    zArray = vtkDoubleArray()
    zArray.InsertNextValue(0.0)
    

    velArray = vtkDoubleArray()
    velArray.SetName('Velocity')
    velArray.SetNumberOfComponents(3)
    for j in range(ny):
        for i in range(nx):
    	    velArray.InsertNextTuple3(data[j+i*ny,4],data[j+i*ny,5],0)



    rhoArray = vtkDoubleArray()
    rhoArray.SetName('Density')
    for j in range(ny):
        for i in range(nx):
    	    rhoArray.InsertNextValue(data[j+i*ny,3])

    solArray = vtkDoubleArray()
    solArray.SetName('obs')
    for j in range(ny):
        for i in range(nx):
    	    solArray.InsertNextValue(data[j+i*ny,6])

    grid.SetXCoordinates(xArray)
    grid.SetYCoordinates(yArray)
    grid.SetZCoordinates(zArray)
    grid.GetPointData().AddArray(velArray)
    grid.GetPointData().AddArray(rhoArray)
    grid.GetPointData().AddArray(solArray)
    print('There are', grid.GetNumberOfPoints(), 'points.')
    print('There are', grid.GetNumberOfCells(), 'cells.')


    writer = vtk.vtkRectilinearGridWriter()
    writer.SetFileName('output.vtk')
    writer.SetInputData(grid)
    #writer.SetFileVersion(4)
    writer.Write()

if __name__ == '__main__':
    vtkConverter()

