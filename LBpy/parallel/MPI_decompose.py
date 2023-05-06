import numpy as np
import os
import sys


def decompose(mesh, rank, size, comm):
    try:
        workingDir = os.getcwd()
        sys.path.append(workingDir)
        from simulation import (decomposeDict)
    except ImportError as e:
        print('FATAL ERROR!')
        print(str(e))
        print('Aborting....')
        os._exit(1)
    try:
        nProc_x = decomposeDict['nx']
        nProc_y = decomposeDict['ny']
    except KeyError as e:
        print('FATAL ERROR!')
        print(str(e))
        print('Aborting....')
        os._exit(1)
    mpiParams = decomposeParams(nProc_x, nProc_y)
    if mpiParams.nProc_x * mpiParams.nProc_y != size:
        if rank == 0:
            print("FATAL ERROR in domain decompsition!")
            print("nx * ny is not equal to the total no.of MPI processes")
        comm.Barrier()
        comm.Abort(0)

    if rank == 0:
        if not os.path.isdir('procs'):
            os.makedirs('procs')
    comm.Barrier()
    if not os.path.isdir('procs/proc_' + str(rank)):
        os.makedirs('procs/proc_' + str(rank))
    decomposeFile = open('procs/proc_' + str(rank) + '/' +
                         '/log_decompose', 'w')
    decomposeFile.write('Domain decomposition information...\n')
    decomposeFile.write('\trank : ' + str(rank) + '\n')
    for i in range(nProc_x):
        for j in range(nProc_y):
            if rank == nProc_y * i + j:
                Nx_local = int(np.ceil(mesh.Nx/nProc_x))
                Ny_local = int(np.ceil(mesh.Ny/nProc_y))
                if i == nProc_x - 1:
                    Nx_local = mesh.Nx - \
                        i * int(np.ceil(mesh.Nx/nProc_x))
                if j == nProc_y - 1:
                    Ny_local = mesh.Ny - \
                        j * int(np.ceil(mesh.Ny/nProc_y))
    mesh.Nx_global = mesh.Nx
    mesh.Ny_global = mesh.Ny
    mesh.Nx = Nx_local + 2
    mesh.Ny = Ny_local + 2
    mpiParams.nx = int(rank / nProc_y)
    mpiParams.ny = int(rank % nProc_y)
    decomposeFile.write('\tn_Procs x : ' + str(nProc_x) + '\n')
    decomposeFile.write('\tn_Procs y : ' + str(nProc_y) + '\n')
    decomposeFile.write('\tNo. of grid points in x-direction : ' +
                        str(mesh.Nx) + '\n')
    decomposeFile.write('\tNo. of grid points in y-direction : ' +
                        str(mesh.Ny) + '\n')
    decomposeFile.close()
    return mpiParams


class decomposeParams:
    def __init__(self, nProc_x, nProc_y):
        self.nProc_x = nProc_x
        self.nProc_y = nProc_y
        self.nx = 0
        self.ny = 0


def solidCopy(solid, fields, mpiParams, mesh):
    i_write, j_write = 0, 0
    for i in range(int(mpiParams.nx * (mesh.Nx - 2)),
                   int((mpiParams.nx + 1) * (mesh.Nx - 2))):
        for j in range(int(mpiParams.ny * (mesh.Ny - 2)),
                       int((mpiParams.ny + 1) * (mesh.Ny - 2))):
            ind = int(i * mesh.Ny_global + j)
            ind_write = int((i_write + 1) * mesh.Ny + (j_write + 1))
            fields.solid[ind_write] = solid[ind]
            j_write += 1
        j_write = 0
        i_write += 1


def distributeSolid_mpi(solid, fields, mpiParams, mesh, rank, size, comm):
    if rank == 0:
        print('MPI option selected')
        print('Distributing obstacle to sub-domains...')
    if rank == 0:
        for nx in range(mpiParams.nProc_x):
            for ny in range(mpiParams.nProc_y):
                rank_send = int(nx * mpiParams.nProc_y + ny)
                if rank_send == 0:
                    solidCopy(solid, fields, mpiParams, mesh)
                else:
                    comm.Send(solid, dest=rank_send, tag=rank_send)
    else:
        solid_temp = np.zeros((mesh.Nx_global * mesh.Ny_global),
                              dtype=np.int32)
        comm.Recv(solid_temp, source=0, tag=rank)
        solidCopy(solid, fields, mpiParams, mesh)
    comm.Barrier()
    if rank == 0:
        print('done distributing obstacle')


def distributeBoundaries_mpi(boundary, mpiParams, mesh, rank, size, comm):
    if rank == 0:
        print('MPI option selected')
        print('Distributing boundaries to sub-domains...')
    if rank == 0:
        N_local = np.zeros((mpiParams.nProc_x, mpiParams.nProc_y, 2),
                           dtype=np.int64)
        for nx in range(mpiParams.nProc_x - 1, -1, -1):
            for ny in range(mpiParams.nProc_y - 1, -1, -1):
                N_local[nx, ny, 0] = int(np.ceil(mesh.Nx_global /
                                                 mpiParams.nProc_x))
                N_local[nx, ny, 1] = int(np.ceil(mesh.Ny_global /
                                                 mpiParams.nProc_y))
                if nx == mpiParams.nProc_x - 1:
                    N_local[nx, ny, 0] = mesh.Nx_global - \
                        nx * int(np.ceil(mesh.Nx_global/mpiParams.nProc_x))
                if ny == mpiParams.nProc_y - 1:
                    N_local[nx, ny, 1] = mesh.Ny_global - \
                        ny * int(np.ceil(mesh.Ny_global/mpiParams.nProc_y))
        for nx in range(mpiParams.nProc_x - 1, -1, -1):
            for ny in range(mpiParams.nProc_y - 1, -1, -1):
                Nx_local = N_local[nx, ny, 0]
                Ny_local = N_local[nx, ny, 1]
                noOfBoundaries = 0
                faceList = []
                boundaryVector = []
                boundaryScalar = []
                outDirections = []
                invDirections = []
                boundaryFunc = []
                for itr in range(boundary.noOfBoundaries):
                    flag = 0
                    tempFaces = []
                    for ind in boundary.faceList[itr]:
                        i = int(ind / mesh.Ny_global)
                        j = int(ind % mesh.Ny_global)
                        if nx == mpiParams.nProc_x - 1:
                            Nx_local = N_local[nx - 1, ny, 0]
                        if ny == mpiParams.nProc_y - 1:
                            Ny_local = N_local[nx, ny - 1, 1]
                        if (i >= nx * Nx_local and i < (nx + 1) * Nx_local
                                and j >= ny * Ny_local and j < (ny + 1) *
                                Ny_local):
                            flag = 1
                            i_local = int(i % Nx_local)
                            j_local = int(j % Ny_local)
                            if nx == mpiParams.nProc_x - 1:
                                Nx_local = N_local[nx, ny, 0]
                            if ny == mpiParams.nProc_y - 1:
                                Ny_local = N_local[nx, ny, 1]
                            tempFaces.append((i_local + 1) * (Ny_local + 2)
                                             + (j_local + 1))
                    if flag != 0:
                        noOfBoundaries += 1
                        faceList.append(np.array(tempFaces, dtype=np.int64))
                        boundaryVector.append(boundary.boundaryVector[itr])
                        boundaryScalar.append(boundary.boundaryScalar[itr])
                        outDirections.append(boundary.outDirections[itr])
                        invDirections.append(boundary.invDirections[itr])
                        boundaryFunc.append(boundary.boundaryFunc[itr])
                dataToProc = (noOfBoundaries, faceList, boundaryVector,
                              boundaryScalar, outDirections, invDirections,
                              boundaryFunc)
                rank_send = nx * mpiParams.nProc_y + ny
                if rank_send == 0:
                    boundary.noOfBoundaries = dataToProc[0]
                    boundary.faceList = dataToProc[1]
                    boundary.boundaryVector = dataToProc[2]
                    boundary.boundaryScalar = dataToProc[3]
                    boundary.outDirections = dataToProc[4]
                    boundary.invDirections = dataToProc[5]
                    boundary.boundaryFunc = dataToProc[6]
                else:
                    comm.send(dataToProc, dest=rank_send, tag=rank_send)
    else:
        dataFromRoot = comm.recv(source=0, tag=rank)
        boundary.noOfBoundaries = dataFromRoot[0]
        boundary.faceList = dataFromRoot[1]
        boundary.boundaryVector = dataFromRoot[2]
        boundary.boundaryScalar = dataFromRoot[3]
        boundary.outDirections = dataFromRoot[4]
        boundary.invDirections = dataFromRoot[5]
        boundary.boundaryFunc = dataFromRoot[6]
    comm.Barrier()
    if rank == 0:
        print('done distributing boundaries')
