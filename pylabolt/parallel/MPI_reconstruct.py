import numpy as np
import os
import sys
import numba
from mpi4py import MPI
from pylabolt.base.mesh import createMesh


def reconstructMesh(comm, precision):
    try:
        from simulation import meshDict
    except ImportError as e:
        print('FATAL ERROR!')
        print(str(e))
        print('Aborting....')
        comm.Abort(1)
    meshObj = createMesh(meshDict, precision)
    Nx_global = meshObj.Nx_global
    Ny_global = meshObj.Ny_global
    x = meshObj.x
    y = meshObj.y
    return Nx_global, Ny_global, x, y


def readDecomposeLog(rank, comm):
    try:
        decomposeFile = open('procs/proc_' + str(rank) + '/' +
                             '/log_decompose', 'r')
    except FileNotFoundError as e:
        print('FATAL ERROR!')
        print(str(e))
        print('Aborting....')
        comm.Abort(1)
    lineList = decomposeFile.readlines()
    nProc_x = int(lineList[2].split()[-1])
    nProc_y = int(lineList[3].split()[-1])
    Nx = int(lineList[4].split()[-1])
    Ny = int(lineList[5].split()[-1])
    return Nx, Ny, nProc_x, nProc_y


def readFields(fieldsFile, Nx, Ny, precision):
    u = np.zeros((Nx * Ny, 2), dtype=precision)
    rho = np.ones((Nx * Ny), dtype=precision)
    solid = np.zeros((Nx * Ny), dtype=np.int32)
    for line in fieldsFile:
        arr = line.split()
        ind = int(arr[0])
        rho[ind] = precision(arr[1])
        u[ind, 0] = precision(arr[2])
        u[ind, 1] = precision(arr[3])
        solid[ind] = np.int32(arr[4])
    return u, rho, solid


def writeData(time, x, y, u, rho, solid):
    if not os.path.isdir('output'):
        os.makedirs('output')
    if not os.path.isdir('output/' + str(time)):
        os.makedirs('output/' + str(time))
    writeFile = open('output/' + str(time) + '/fields.dat', 'w')
    for ind in range(u.shape[0]):
        writeFile.write(str(round(ind, 10)).ljust(12) + '\t' +
                        str(round(x[ind], 10)).ljust(12) + '\t' +
                        str(round(y[ind], 10)).ljust(12) + '\t' +
                        str(round(rho[ind], 10)).ljust(12) + '\t' +
                        str(round(u[ind, 0], 10)).ljust(12) + '\t' +
                        str(round(u[ind, 1], 10)).ljust(12) + '\t' +
                        str(round(solid[ind], 10)).ljust(12) + '\n')
    writeFile.close()


@numba.njit
def gather_copy(u_all, rho_all, solid_all, u_temp, rho_temp,
                solid_temp, N_sub, nx, ny, Nx_global, Ny_global,
                nProc_x, nProc_y):
    i_read, j_read = 0, 0
    Nx_i, Nx_f = int(nx * (N_sub[0] - 2)), int((nx + 1) * (N_sub[0] - 2))
    Ny_i, Ny_f = int(ny * (N_sub[1] - 2)), int((ny + 1) * (N_sub[1] - 2))
    if nx == nProc_x - 1:
        Nx_i = nx * int(np.ceil(Nx_global/nProc_x))
        Nx_f = int(Nx_i + N_sub[0] - 2)
    if ny == nProc_y - 1:
        Ny_i = ny * int(np.ceil(Ny_global/nProc_y))
        Ny_f = int(Ny_i + N_sub[1] - 2)
    for i in range(Nx_i, Nx_f):
        for j in range(Ny_i, Ny_f):
            ind = int(i * Ny_global + j)
            ind_read = int((i_read + 1) * N_sub[1] + (j_read + 1))
            u_all[ind, 0] = u_temp[ind_read, 0]
            u_all[ind, 1] = u_temp[ind_read, 1]
            rho_all[ind] = rho_temp[ind_read]
            solid_all[ind] = solid_temp[ind_read]
            j_read += 1
        j_read = 0
        i_read += 1


def gather(rank, timeStep, comm):
    current_dir = os.getcwd()
    sys.path.append(current_dir)
    try:
        fieldsFile = open('procs/proc_' + str(rank) + '/' +
                          str(timeStep) + '/fields.dat', 'r')
        from simulation import controlDict
        precisionType = controlDict['precision']
        if precisionType == 'single':
            precision = np.float32
        elif precisionType == 'double':
            precision = np.float64
        else:
            raise RuntimeError("Incorrect precision specified!")
    except ImportError as e:
        print('FATAL ERROR!')
        print(str(e))
        print('Aborting....')
        os._exit(1)
    except KeyError as e:
        if rank == 0:
            print('ERROR! Keyword ' + str(e) + ' missing in controlDict')
        os._exit(1)
    except FileNotFoundError:
        raise RuntimeError('FATAL ERROR! No output data' +
                           ' found to reconstruct - ' + str(rank))
        os._exit(1)
    Nx, Ny, nProc_x, nProc_y = readDecomposeLog(rank, comm)
    u, rho, solid = readFields(fieldsFile, Nx, Ny, precision)
    if rank == 0:
        Nx_global, Ny_global, x, y = reconstructMesh(comm, precision)
        u_all = np.zeros((Nx_global * Ny_global, 2), dtype=precision)
        rho_all = np.ones((Nx_global * Ny_global), dtype=precision)
        solid_all = np.zeros((Nx_global * Ny_global), dtype=np.int32)
        fieldsToGather = (u_all, rho_all, solid_all)
        for nx in range(nProc_x):
            for ny in range(nProc_y):
                current_rank = int(nx * nProc_y + ny)
                if current_rank == 0:
                    N_sub = np.array([Nx, Ny], dtype=np.int32)
                    gather_copy(*fieldsToGather, u, rho, solid, N_sub, nx,
                                ny, Nx_global, Ny_global, nProc_x, nProc_y)
                else:
                    N_sub = np.zeros(2, dtype=np.int32)
                    comm.Recv(N_sub, source=current_rank,
                              tag=1*current_rank)
                    u_temp = np.zeros((N_sub[0] * N_sub[1], 2),
                                      dtype=precision)
                    rho_temp = np.zeros((N_sub[0] * N_sub[1]),
                                        dtype=precision)
                    solid_temp = np.zeros((N_sub[0] * N_sub[1]),
                                          dtype=np.int32)
                    comm.Recv(u_temp, source=current_rank,
                              tag=2*current_rank)
                    comm.Recv(rho_temp, source=current_rank,
                              tag=3*current_rank)
                    comm.Recv(solid_temp, source=current_rank,
                              tag=4*current_rank)
                    gather_copy(*fieldsToGather, u_temp, rho_temp, solid_temp,
                                N_sub, nx, ny, Nx_global, Ny_global, nProc_x,
                                nProc_y)
        return x, y, u_all, rho_all, solid_all
    else:
        comm.Send(np.array([Nx, Ny], dtype=np.int32),
                  dest=0, tag=1*rank)
        comm.Send(u, dest=0, tag=2*rank)
        comm.Send(rho, dest=0, tag=3*rank)
        comm.Send(solid, dest=0, tag=4*rank)
        return 0, 0, 0, 0, 0


def reconstruct(options, time):
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print('\n\nDomain reconstruction initialized with size ' + str(size) +
              ' and option ' + options + '\n')
    if size == 1:
        print('Nothing to reconstruct -- size is 1')
        print('help --> check the number of processors used to run the case')
        print('Exiting...')
        os._exit(1)

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
            gather(rank, time, comm)
        except ImportError as e:
            print('FATAL ERROR!')
            print(str(e))
            print('Aborting....')
            comm.Abort(1)
        except KeyError as e:
            print('FATAL ERROR!')
            print(str(e))
            print('Aborting....')
            comm.Abort(1)
        except FileNotFoundError as e:
            print('FATAL ERROR!')
            print(str(e))
            print('Aborting....')
            comm.Abort(1)
        if rank == 0:
            print('Reconstructing fields --> time = ' + str(time))
        x, y, u, rho, solid = gather(rank, time, comm)
        if rank == 0:
            writeData(time, x, y, u, rho, solid)
    elif options == 'all':
        try:
            from simulation import controlDict
            if int(controlDict['endTime']) < int(controlDict['saveInterval']):
                raise Warning("'endTime' in controlDict is greater than" +
                              " 'saveInterval'! Execution may fail or" +
                              " produce no output at all!")
            startTime = int(controlDict['startTime'])
            endTime = int(controlDict['endTime'])
            interval = int(controlDict['saveInterval'])
        except ImportError as e:
            print('FATAL ERROR!')
            print(str(e))
            print('Aborting....')
            comm.Abort(1)
        except KeyError as e:
            print('FATAL ERROR!')
            print(str(e))
            print('Aborting....')
            comm.Abort(1)
        except FileNotFoundError as e:
            print('FATAL ERROR!')
            print(str(e))
            print('Aborting....')
            comm.Abort(1)
        for time in range(startTime, endTime + 1, interval):
            if rank == 0:
                print('Reconstructing fields --> time = ' + str(time))
            x, y, u, rho, solid = gather(rank, time, comm)
            if rank == 0:
                writeData(time, x, y, u, rho, solid)
    elif options == 'time':
        if rank == 0:
            print('Reconstructing fields --> time = ' + str(time))
        x, y, u, rho, solid = gather(rank, time, comm)
        if rank == 0:
            writeData(time, x, y, u, rho, solid)
    if rank == 0:
        print('\nReconstruction of Fields done!\n')
    MPI.Finalize()
