import numpy as np
import os
import sys
import numba
from mpi4py import MPI
from time import perf_counter

from pylabolt.base.mesh import createMesh


def reconstructMesh(comm, precision, rank):
    try:
        from simulation import meshDict
    except ImportError as e:
        print('FATAL ERROR!')
        print(str(e))
        print('Aborting....')
        comm.Abort(1)
    meshObj = createMesh(meshDict, precision, rank)
    return meshObj


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


def readFields(Nx, Ny, decomposedFields, precision):
    for itr in range(Nx * Ny):
        if decomposedFields.u_flag is True:
            arr = decomposedFields.uFileLines[itr].split()
            decomposedFields.u[itr, 0] = precision(arr[0])
            decomposedFields.u[itr, 1] = precision(arr[1])
        if decomposedFields.rho_flag is True:
            arr = decomposedFields.rhoFileLines[itr].split()
            decomposedFields.rho[itr] = precision(arr[0])
        if decomposedFields.p_flag is True:
            arr = decomposedFields.pFileLines[itr].split()
            decomposedFields.p[itr] = precision(arr[0])
        if decomposedFields.phi_flag is True:
            arr = decomposedFields.phiFileLines[itr].split()
            decomposedFields.phi[itr] = precision(arr[0])
        if decomposedFields.T_flag is True:
            arr = decomposedFields.TFileLines[itr].split()
            decomposedFields.T[itr] = precision(arr[0])
        arr = decomposedFields.solidFileLines[itr].split()
        decomposedFields.solid[itr] = np.int32(arr[0])
        arr = decomposedFields.boundaryNodeFileLines[itr].split()
        decomposedFields.boundaryNode[itr] = np.int32(arr[0])


def writeData(timeStep, mesh, fields):
    if not os.path.isdir('output'):
        os.makedirs('output')
    if not os.path.isdir('output/' + str(timeStep)):
        os.makedirs('output/' + str(timeStep))
    pointsFile = open('output/' + str(timeStep) + '/points.dat', 'w')
    solidFile = open('output/' + str(timeStep) + '/solid.dat', 'w')
    if fields.u_flag is True:
        uFile = open('output/' + str(timeStep) + '/u.dat', 'w')
    if fields.rho_flag is True:
        rhoFile = open('output/' + str(timeStep) + '/rho.dat', 'w')
    if fields.p_flag is True:
        pFile = open('output/' + str(timeStep) + '/p.dat', 'w')
    if fields.phi_flag is True:
        phiFile = open('output/' + str(timeStep) + '/phi.dat', 'w')
    if fields.T_flag is True:
        TFile = open('output/' + str(timeStep) + '/T.dat', 'w')
    for ind in range(mesh.Nx * mesh.Ny):
        if fields.boundaryNode[ind] != 1:
            pointsFile.write(str(np.round(ind, 10)).ljust(12) + '\t' +
                             str(np.round(mesh.x[ind], 10)).ljust(12) + '\t' +
                             str(np.round(mesh.y[ind], 10)).ljust(12) + '\n')
            if fields.u_flag is True:
                uFile.\
                    write(str(np.round(fields.u[ind, 0], 10)).ljust(12) + '\t'
                          + str(np.round(fields.u[ind, 1], 10)).ljust(12) +
                          '\n')
            if fields.rho_flag is True:
                rhoFile.write(str(np.round(fields.rho[ind], 10)).ljust(12) +
                              '\n')
            if fields.p_flag is True:
                pFile.write(str(np.round(fields.p[ind], 10)).ljust(12) +
                            '\n')
            if fields.phi_flag is True:
                phiFile.write(str(np.round(fields.phi[ind], 10)).ljust(12) +
                              '\n')
            if fields.T_flag is True:
                TFile.write(str(np.round(fields.T[ind], 10)).ljust(12) + '\n')
            solidFile.write(str(np.round(fields.solid[ind], 10)).ljust(12)
                            + '\n')
    pointsFile.close()
    solidFile.close()
    if fields.u_flag is True:
        uFile.close()
    if fields.rho_flag is True:
        rhoFile.close()
    if fields.p_flag is True:
        pFile.close()
    if fields.phi_flag is True:
        phiFile.close()
    if fields.T_flag is True:
        TFile.close()


@numba.njit
def gather_copy_vector(u, u_temp, N_sub, nx, ny, Nx_global, Ny_global,
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
            u[ind, 0] = u_temp[ind_read, 0]
            u[ind, 1] = u_temp[ind_read, 1]
            j_read += 1
        j_read = 0
        i_read += 1


@numba.njit
def gather_copy_scalar(scalarField, scalarField_temp, N_sub, nx, ny,
                       Nx_global, Ny_global, nProc_x, nProc_y):
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
            scalarField[ind] = scalarField_temp[ind_read]
            j_read += 1
        j_read = 0
        i_read += 1


class fields:
    def __init__(self, Nx, Ny, fileList, precision, timeStep,
                 rank, decomposed=False):
        self.u_flag = False
        self.rho_flag = False
        self.p_flag = False
        self.phi_flag = False
        self.T_flag = False
        for fileName in fileList:
            if fileName == 'u.dat':
                self.u_flag = True
                self.u = np.zeros((Nx * Ny, 2), dtype=precision)
                if decomposed is True:
                    self.uFile = open('procs/proc_' + str(rank) + '/' +
                                      str(timeStep) + '/u.dat', 'r')
                    self.uFileLines = self.uFile.readlines()
            if fileName == 'rho.dat':
                self.rho_flag = True
                self.rho = np.zeros(Nx * Ny, dtype=precision)
                if decomposed is True:
                    self.rhoFile = open('procs/proc_' + str(rank) + '/' +
                                        str(timeStep) + '/rho.dat', 'r')
                    self.rhoFileLines = self.rhoFile.readlines()
            if fileName == 'p.dat':
                self.p_flag = True
                self.p = np.zeros(Nx * Ny, dtype=precision)
                if decomposed is True:
                    self.pFile = open('procs/proc_' + str(rank) + '/' +
                                      str(timeStep) + '/p.dat', 'r')
                    self.pFileLines = self.pFile.readlines()
            if fileName == 'phi.dat':
                self.phi_flag = True
                self.phi = np.zeros(Nx * Ny, dtype=precision)
                if decomposed is True:
                    self.phiFile = open('procs/proc_' + str(rank) + '/' +
                                        str(timeStep) + '/phi.dat', 'r')
                    self.phiFileLines = self.phiFile.readlines()
            if fileName == 'T.dat':
                self.T_flag = True
                self.T = np.zeros(Nx * Ny, dtype=precision)
                if decomposed is True:
                    self.TFile = open('procs/proc_' + str(rank) + '/' +
                                      str(timeStep) + '/T.dat', 'r')
                    self.TFileLines = self.TFile.readlines()
        self.solid = np.zeros(Nx * Ny, dtype=np.int32)
        self.solidFile = open('procs/proc_' + str(rank) + '/' +
                              str(timeStep) + '/solid.dat', 'r')
        self.solidFileLines = self.solidFile.readlines()
        self.boundaryNode = np.zeros(Nx * Ny, dtype=np.int32)
        self.boundaryNodeFile = open('procs/proc_' + str(rank) + '/' +
                                     str(timeStep) + '/boundaryNode.dat', 'r')
        self.boundaryNodeFileLines = self.boundaryNodeFile.readlines()


def gather(rank, timeStep, comm):
    current_dir = os.getcwd()
    sys.path.append(current_dir)
    Nx, Ny, nProc_x, nProc_y = readDecomposeLog(rank, comm)
    try:
        from simulation import controlDict
        precisionType = controlDict['precision']
        if precisionType == 'single':
            precision = np.float32
        elif precisionType == 'double':
            precision = np.float64
        else:
            raise RuntimeError("Incorrect precision specified!")
        fileList = os.listdir('procs/proc_' + str(rank) + '/' +
                              str(timeStep) + '/')
        decomposedFields = fields(Nx, Ny, fileList, precision, timeStep, rank,
                                  decomposed=True)
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
    readFields(Nx, Ny, decomposedFields, precision)
    if rank == 0:
        mesh = reconstructMesh(comm, precision, rank)
        reconstructedFields = fields(mesh.Nx_global, mesh.Ny_global, fileList,
                                     precision, timeStep, rank)
        # print(nProc_x, nProc_y)
        for nx in range(nProc_x):
            for ny in range(nProc_y):
                current_rank = int(nx * nProc_y + ny)
                if current_rank == 0:
                    N_sub = np.array([Nx, Ny], dtype=np.int32)
                    if reconstructedFields.u_flag is True:
                        gather_copy_vector(reconstructedFields.u,
                                           decomposedFields.u, N_sub,
                                           nx, ny, mesh.Nx_global,
                                           mesh.Ny_global, nProc_x, nProc_y)
                    if reconstructedFields.rho_flag is True:
                        gather_copy_scalar(reconstructedFields.rho,
                                           decomposedFields.rho, N_sub,
                                           nx, ny, mesh.Nx_global,
                                           mesh.Ny_global, nProc_x, nProc_y)
                    if reconstructedFields.p_flag is True:
                        gather_copy_scalar(reconstructedFields.p,
                                           decomposedFields.p, N_sub,
                                           nx, ny, mesh.Nx_global,
                                           mesh.Ny_global, nProc_x, nProc_y)
                    if reconstructedFields.phi_flag is True:
                        gather_copy_scalar(reconstructedFields.phi,
                                           decomposedFields.phi, N_sub,
                                           nx, ny, mesh.Nx_global,
                                           mesh.Ny_global, nProc_x, nProc_y)
                    if reconstructedFields.T_flag is True:
                        gather_copy_scalar(reconstructedFields.T,
                                           decomposedFields.T, N_sub,
                                           nx, ny, mesh.Nx_global,
                                           mesh.Ny_global, nProc_x, nProc_y)
                    gather_copy_scalar(reconstructedFields.solid,
                                       decomposedFields.solid, N_sub,
                                       nx, ny, mesh.Nx_global, mesh.Ny_global,
                                       nProc_x, nProc_y)
                    gather_copy_scalar(reconstructedFields.boundaryNode,
                                       decomposedFields.boundaryNode, N_sub,
                                       nx, ny, mesh.Nx_global, mesh.Ny_global,
                                       nProc_x, nProc_y)
                else:
                    N_sub = np.zeros(2, dtype=np.int32)
                    comm.Recv(N_sub, source=current_rank,
                              tag=1*current_rank)
                    if decomposedFields.u_flag is True:
                        u_temp = np.zeros((N_sub[0] * N_sub[1], 2),
                                          dtype=precision)
                        comm.Recv(u_temp, source=current_rank,
                                  tag=2*current_rank)
                        gather_copy_vector(reconstructedFields.u,
                                           u_temp, N_sub,
                                           nx, ny, mesh.Nx_global,
                                           mesh.Ny_global, nProc_x, nProc_y)
                    if decomposedFields.rho_flag is True:
                        rho_temp = np.zeros((N_sub[0] * N_sub[1]),
                                            dtype=precision)
                        comm.Recv(rho_temp, source=current_rank,
                                  tag=3*current_rank)
                        gather_copy_scalar(reconstructedFields.rho,
                                           rho_temp, N_sub,
                                           nx, ny, mesh.Nx_global,
                                           mesh.Ny_global, nProc_x, nProc_y)
                    if decomposedFields.p_flag is True:
                        p_temp = np.zeros((N_sub[0] * N_sub[1]),
                                          dtype=precision)
                        comm.Recv(p_temp, source=current_rank,
                                  tag=4*current_rank)
                        gather_copy_scalar(reconstructedFields.p,
                                           p_temp, N_sub,
                                           nx, ny, mesh.Nx_global,
                                           mesh.Ny_global, nProc_x, nProc_y)
                    if decomposedFields.phi_flag is True:
                        phi_temp = np.zeros((N_sub[0] * N_sub[1]),
                                            dtype=precision)
                        comm.Recv(phi_temp, source=current_rank,
                                  tag=5*current_rank)
                        gather_copy_scalar(reconstructedFields.phi,
                                           phi_temp, N_sub,
                                           nx, ny, mesh.Nx_global,
                                           mesh.Ny_global, nProc_x, nProc_y)
                    if decomposedFields.T_flag is True:
                        T_temp = np.zeros((N_sub[0] * N_sub[1]),
                                          dtype=precision)
                        comm.Recv(T_temp, source=current_rank,
                                  tag=6*current_rank)
                        gather_copy_scalar(reconstructedFields.T,
                                           T_temp, N_sub,
                                           nx, ny, mesh.Nx_global,
                                           mesh.Ny_global, nProc_x, nProc_y)
                    solid_temp = np.zeros((N_sub[0] * N_sub[1]),
                                          dtype=np.int32)
                    comm.Recv(solid_temp, source=current_rank,
                              tag=7*current_rank)
                    gather_copy_scalar(reconstructedFields.solid,
                                       solid_temp, N_sub,
                                       nx, ny, mesh.Nx_global, mesh.Ny_global,
                                       nProc_x, nProc_y)
                    boundaryNode_temp = np.zeros((N_sub[0] * N_sub[1]),
                                                 dtype=np.int32)
                    comm.Recv(boundaryNode_temp, source=current_rank,
                              tag=8*current_rank)
                    gather_copy_scalar(reconstructedFields.boundaryNode,
                                       boundaryNode_temp, N_sub,
                                       nx, ny, mesh.Nx_global, mesh.Ny_global,
                                       nProc_x, nProc_y)
        return mesh, reconstructedFields
    else:
        comm.Send(np.array([Nx, Ny], dtype=np.int32),
                  dest=0, tag=1*rank)
        # print('sent', rank)
        if decomposedFields.u_flag is True:
            comm.Send(decomposedFields.u, dest=0, tag=2*rank)
        if decomposedFields.rho_flag is True:
            comm.Send(decomposedFields.rho, dest=0, tag=3*rank)
        if decomposedFields.p_flag is True:
            comm.Send(decomposedFields.p, dest=0, tag=4*rank)
        if decomposedFields.phi_flag is True:
            comm.Send(decomposedFields.phi, dest=0, tag=5*rank)
        if decomposedFields.T_flag is True:
            comm.Send(decomposedFields.T, dest=0, tag=6*rank)
        comm.Send(decomposedFields.solid, dest=0, tag=7*rank)
        comm.Send(decomposedFields.boundaryNode, dest=0, tag=8*rank)
        return None, None


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
        start = perf_counter()
        if rank == 0:
            print('Reconstructing fields --> time = ' + str(time))
        mesh, reconstructedFields = gather(rank, time, comm)
        if rank == 0:
            writeData(time, mesh, reconstructedFields)
        runTime = perf_counter() - start
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
        start = perf_counter()
        for time in range(startTime, endTime + 1, interval):
            if rank == 0:
                print('Reconstructing fields --> time = ' + str(time))
            mesh, reconstructedFields = gather(rank, time, comm)
            if rank == 0:
                writeData(time, mesh, reconstructedFields)
        runTime = perf_counter() - start
    elif options == 'time':
        start = perf_counter()
        if rank == 0:
            print('Reconstructing fields --> time = ' + str(time))
        mesh, reconstructedFields = gather(rank, time, comm)
        if rank == 0:
            writeData(time, mesh, reconstructedFields)
        runTime = perf_counter() - start
    if rank == 0:
        print('\nReconstruction of Fields done!')
        print('\nRun time = ' + str(runTime) + ' s \n')
    MPI.Finalize()
