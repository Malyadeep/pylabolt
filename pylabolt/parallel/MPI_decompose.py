import numpy as np
import os
import sys

from pylabolt.parallel.MPI_comm import proc_boundary


def computeLocalSize(mpiParams, mesh):
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
    return N_local


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
    mpiParams.N_local = computeLocalSize(mpiParams, mesh)
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
        self.N_local = np.zeros((2, 2, 0), dtype=np.int64)


def fieldCopy(fields_temp, fields, mpiParams, Nx_local, Ny_local, mesh):
    i_write, j_write = 0, 0
    Nx_final, Ny_final = Nx_local, Ny_local
    if mpiParams.nx == mpiParams.nProc_x - 1:
        Nx_final = mesh.Nx - 2
    if mpiParams.ny == mpiParams.nProc_y - 1:
        Ny_final = mesh.Ny - 2
    for i in range(int(mpiParams.nx * Nx_local),
                   int(mpiParams.nx * Nx_local + Nx_final)):
        for j in range(int(mpiParams.ny * Ny_local),
                       int(mpiParams.ny * Ny_local + Ny_final)):
            ind = int(i * mesh.Ny_global + j)
            ind_write = int((i_write + 1) * mesh.Ny + (j_write + 1))
            fields.u[ind_write, 0] = fields_temp.u[ind, 0]
            fields.u[ind_write, 1] = fields_temp.u[ind, 1]
            fields.rho[ind_write] = fields_temp.rho[ind]
            j_write += 1
        j_write = 0
        i_write += 1


def distributeInitialFields_mpi(fields_temp, fields, mpiParams, mesh,
                                rank, size, comm, precision):
    if rank == 0:
        print('MPI option selected', flush=True)
        print('Distributing fields to sub-domains...', flush=True)
    N_local = mpiParams.N_local
    nx = int(rank / mpiParams.nProc_y)
    ny = int(rank % mpiParams.nProc_y)
    Nx_local = N_local[nx, ny, 0]
    Ny_local = N_local[nx, ny, 1]
    if nx == mpiParams.nProc_x - 1:
        Nx_local = N_local[nx - 1, ny, 0]
    if ny == mpiParams.nProc_y - 1:
        Ny_local = N_local[nx, ny - 1, 1]
    if rank == 0:
        for nx in range(mpiParams.nProc_x):
            for ny in range(mpiParams.nProc_y):
                rank_send = int(nx * mpiParams.nProc_y + ny)
                if rank_send == 0:
                    fieldCopy(fields_temp, fields, mpiParams, Nx_local,
                              Ny_local, mesh)
                else:
                    comm.send(fields_temp, dest=rank_send, tag=rank_send)
    else:
        fields_temp = comm.recv(source=0, tag=rank)
        fieldCopy(fields_temp, fields, mpiParams, Nx_local, Ny_local, mesh)
    comm.Barrier()
    initial_send_topBottom = np.zeros((mesh.Nx + 2, 2),
                                      dtype=precision)
    initial_recv_topBottom = np.zeros((mesh.Nx + 2, 2),
                                      dtype=precision)
    initial_send_leftRight = np.zeros((mesh.Ny + 2, 2),
                                      dtype=precision)
    initial_recv_leftRight = np.zeros((mesh.Ny + 2, 2),
                                      dtype=precision)
    args = (mesh.Nx, mesh.Ny,
            fields.u, initial_send_topBottom,
            initial_recv_topBottom, initial_send_leftRight,
            initial_recv_leftRight, mpiParams.nx,
            mpiParams.ny, mpiParams.nProc_x,
            mpiParams.nProc_y, comm)
    proc_boundary(*args)
    comm.Barrier()
    initial_send_topBottom = np.zeros((mesh.Nx + 2),
                                      dtype=precision)
    initial_recv_topBottom = np.zeros((mesh.Nx + 2),
                                      dtype=precision)
    initial_send_leftRight = np.zeros((mesh.Ny + 2),
                                      dtype=precision)
    initial_recv_leftRight = np.zeros((mesh.Ny + 2),
                                      dtype=precision)
    args = (mesh.Nx, mesh.Ny,
            fields.rho, initial_send_topBottom,
            initial_recv_topBottom, initial_send_leftRight,
            initial_recv_leftRight, mpiParams.nx,
            mpiParams.ny, mpiParams.nProc_x,
            mpiParams.nProc_y, comm)
    proc_boundary(*args)
    comm.Barrier()
    if rank == 0:
        print('Done distributing fields!', flush=True)


def solidCopy(solid, fields, mpiParams, Nx_local, Ny_local, mesh):
    i_write, j_write = 0, 0
    Nx_final, Ny_final = Nx_local, Ny_local
    if mpiParams.nx == mpiParams.nProc_x - 1:
        Nx_final = mesh.Nx - 2
    if mpiParams.ny == mpiParams.nProc_y - 1:
        Ny_final = mesh.Ny - 2
    for i in range(int(mpiParams.nx * Nx_local),
                   int(mpiParams.nx * Nx_local + Nx_final)):
        for j in range(int(mpiParams.ny * Ny_local),
                       int(mpiParams.ny * Ny_local + Ny_final)):
            ind = int(i * mesh.Ny_global + j)
            ind_write = int((i_write + 1) * mesh.Ny + (j_write + 1))
            fields.solid[ind_write, :] = solid[ind, :]
            j_write += 1
        j_write = 0
        i_write += 1


def distributeSolid_mpi(solid, obstacle, fields, mpiParams, mesh,
                        precision, rank, size, comm):
    if rank == 0:
        print('MPI option selected', flush=True)
        print('Distributing obstacle to sub-domains...', flush=True)
    N_local = mpiParams.N_local
    nx = int(rank / mpiParams.nProc_y)
    ny = int(rank % mpiParams.nProc_y)
    Nx_local = N_local[nx, ny, 0]
    Ny_local = N_local[nx, ny, 1]
    if nx == mpiParams.nProc_x - 1:
        Nx_local = N_local[nx - 1, ny, 0]
    if ny == mpiParams.nProc_y - 1:
        Ny_local = N_local[nx, ny - 1, 1]
    if rank == 0:
        for nx in range(mpiParams.nProc_x):
            for ny in range(mpiParams.nProc_y):
                rank_send = int(nx * mpiParams.nProc_y + ny)
                if rank_send == 0:
                    solidCopy(solid, fields, mpiParams, Nx_local, Ny_local,
                              mesh)
                else:
                    comm.Send(solid, dest=rank_send, tag=rank_send)
    else:
        solid_temp = np.zeros((mesh.Nx_global * mesh.Ny_global, 2),
                              dtype=np.int32)
        comm.Recv(solid_temp, source=0, tag=rank)
        solidCopy(solid_temp, fields, mpiParams, Nx_local, Ny_local, mesh)
    comm.Barrier()
    solid_send_topBottom = np.zeros((mesh.Nx + 2, 2),
                                    dtype=np.int32)
    solid_recv_topBottom = np.zeros((mesh.Nx + 2, 2),
                                    dtype=np.int32)
    solid_send_leftRight = np.zeros((mesh.Ny + 2, 2),
                                    dtype=np.int32)
    solid_recv_leftRight = np.zeros((mesh.Ny + 2, 2),
                                    dtype=np.int32)
    args = (mesh.Nx, mesh.Ny,
            fields.solid, solid_send_topBottom,
            solid_recv_topBottom, solid_send_leftRight,
            solid_recv_leftRight, mpiParams.nx,
            mpiParams.ny, mpiParams.nProc_x,
            mpiParams.nProc_y, comm)
    proc_boundary(*args)
    comm.Barrier()
    # transfer solid nodes with tag
    if rank == 0:
        for nx in range(mpiParams.nProc_x - 1, -1, -1):
            for ny in range(mpiParams.nProc_y - 1, -1, -1):
                Nx_local = N_local[nx, ny, 0]
                Ny_local = N_local[nx, ny, 1]
                obsNodes = []
                obstacles = []
                noOfObstacles = 0
                modifyObsFunc = []
                obsModifiable = obstacle.obsModifiable
                momentOfInertia = []
                obsU = []
                obsU_old = []
                obsOmega = []
                obsOmega_old = []
                obsOrigin = []
                writeInterval = obstacle.writeInterval
                writeProperties = obstacle.writeProperties
                allStatic = obstacle.allStatic
                for itr in range(obstacle.noOfObstacles):
                    flag = 0
                    tempNodes = []
                    for numNode, ind in enumerate(obstacle.obsNodes[itr]):
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
                            tempNodes.append((i_local + 1) * (Ny_local + 2)
                                             + (j_local + 1))
                    if flag != 0:
                        noOfObstacles += 1
                        obsNodes.append(np.array(tempNodes, dtype=np.int64))
                        obstacles.append(obstacle.obstacles[itr])
                        modifyObsFunc.append(obstacle.modifyObsFunc[itr])
                        momentOfInertia.append(obstacle.momentOfInertia[itr])
                        obsU.append(obstacle.obsU[itr])
                        obsU_old.append(obstacle.obsU_old[itr])
                        obsOmega.append(obstacle.obsOmega[itr])
                        obsOmega_old.append(obstacle.obsOmega_old[itr])
                        obsOrigin.append(obstacle.obsOrigin[itr])
                obsOmega = np.array(obsOmega, dtype=precision)
                obsOmega_old = np.array(obsOmega_old, dtype=precision)
                dataToProc = (noOfObstacles, obsNodes, obstacles,
                              modifyObsFunc, obsModifiable, momentOfInertia,
                              obsU, obsU_old, obsOmega, obsOmega_old,
                              obsOrigin, writeProperties, writeInterval,
                              allStatic)
                rank_send = nx * mpiParams.nProc_y + ny
                if rank_send == 0:
                    obstacle.noOfObstacles = dataToProc[0]
                    obstacle.obsNodes = dataToProc[1]
                    obstacle.obstacles = dataToProc[2]
                    obstacle.modifyObsFunc = dataToProc[3]
                    obstacle.obsModifiable = dataToProc[4]
                    obstacle.momentOfInertia = dataToProc[5]
                    obstacle.obsU = dataToProc[6]
                    obstacle.obsU_old = dataToProc[7]
                    obstacle.obsOmega = dataToProc[8]
                    obstacle.obsOmega_old = dataToProc[9]
                    obstacle.obsOrigin = dataToProc[10]
                    obstacle.writeProperties = dataToProc[11]
                    obstacle.writeInterval = dataToProc[12]
                    obstacle.allStatic = allStatic
                else:
                    comm.send(dataToProc, dest=rank_send, tag=rank_send)
    else:
        dataFromRoot = comm.recv(source=0, tag=rank)
        obstacle.noOfObstacles = dataFromRoot[0]
        obstacle.obsNodes = dataFromRoot[1]
        obstacle.obstacles = dataFromRoot[2]
        obstacle.modifyObsFunc = dataFromRoot[3]
        obstacle.obsModifiable = dataFromRoot[4]
        obstacle.momentOfInertia = dataFromRoot[5]
        obstacle.obsU = dataFromRoot[6]
        obstacle.obsU_old = dataFromRoot[7]
        obstacle.obsOmega = dataFromRoot[8]
        obstacle.obsOmega_old = dataFromRoot[9]
        obstacle.obsOrigin = dataFromRoot[10]
        obstacle.writeProperties = dataFromRoot[11]
        obstacle.writeInterval = dataFromRoot[12]
        obstacle.allStatic = dataFromRoot[13]
    comm.Barrier()
    if rank == 0:
        print('Done distributing obstacle!', flush=True)


def distributeBoundaries_mpi(boundary, mpiParams, mesh, rank, size,
                             precision, comm):
    if rank == 0:
        print('MPI option selected', flush=True)
        print('Distributing boundaries to sub-domains...', flush=True)
    if rank == 0:
        N_local = mpiParams.N_local
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
                    tempVector = []
                    for numFace, ind in enumerate(boundary.faceList[itr]):
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
                            if boundary.boundaryType[itr] == 'variableU':
                                tempVector.append(boundary.boundaryVector
                                                  [itr][numFace])
                    if flag != 0:
                        noOfBoundaries += 1
                        faceList.append(np.array(tempFaces, dtype=np.int64))
                        if boundary.boundaryType[itr] == 'variableU':
                            boundaryVector.append(np.array(tempVector,
                                                           dtype=precision))
                        else:
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
        print('Distributing boundaries done!', flush=True)


def distributeForceNodes_mpi(simulation, rank, size, comm):
    if rank == 0:
        print('\nMPI option with force computation selected', flush=True)
        print('Distributing nodes to sub-domains...', flush=True)
    N_local = simulation.mpiParams.N_local
    if rank == 0:
        for nx in range(simulation.mpiParams.nProc_x - 1, -1, -1):
            for ny in range(simulation.mpiParams.nProc_y - 1, -1, -1):
                Nx_local = N_local[nx, ny, 0]
                Ny_local = N_local[nx, ny, 1]
                noOfSurfaces = 0
                surfaceNames = []
                surfaceNodes = []
                surfaceInvList = []
                surfaceOutList = []
                obstacleFlag = []
                for itr in range(simulation.options.noOfSurfaces):
                    flag = 0
                    tempFaces = []
                    for ind in simulation.options.surfaceNodes[itr]:
                        i = int(ind / simulation.mesh.Ny_global)
                        j = int(ind % simulation.mesh.Ny_global)
                        if nx == simulation.mpiParams.nProc_x - 1:
                            Nx_local = N_local[nx - 1, ny, 0]
                        if ny == simulation.mpiParams.nProc_y - 1:
                            Ny_local = N_local[nx, ny - 1, 1]
                        if (i >= nx * Nx_local and i < (nx + 1) * Nx_local
                                and j >= ny * Ny_local and j < (ny + 1) *
                                Ny_local):
                            flag = 1
                            i_local = int(i % Nx_local)
                            j_local = int(j % Ny_local)
                            if nx == simulation.mpiParams.nProc_x - 1:
                                Nx_local = N_local[nx, ny, 0]
                            if ny == simulation.mpiParams.nProc_y - 1:
                                Ny_local = N_local[nx, ny, 1]
                            tempFaces.append((i_local + 1) * (Ny_local + 2)
                                             + (j_local + 1))
                    if flag != 0:
                        noOfSurfaces += 1
                        surfaceNodes.append(np.array(tempFaces,
                                            dtype=np.int64))
                        surfaceNames.append(simulation.options.
                                            surfaceNames[itr])
                        surfaceInvList.append(simulation.options.
                                              surfaceInvList[itr])
                        surfaceOutList.append(simulation.options.
                                              surfaceOutList[itr])
                        obstacleFlag.append(simulation.options.
                                            obstacleFlag[itr])
                dataToProc = (noOfSurfaces, surfaceNames, surfaceNodes,
                              surfaceInvList, surfaceOutList, obstacleFlag)
                rank_send = nx * simulation.mpiParams.nProc_y + ny
                if rank_send == 0:
                    simulation.options.noOfSurfaces = dataToProc[0]
                    simulation.options.surfaceNames = dataToProc[1]
                    simulation.options.surfaceNodes = dataToProc[2]
                    simulation.options.surfaceInvList = dataToProc[3]
                    simulation.options.surfaceOutList = dataToProc[4]
                    simulation.options.obstacleFlag = dataToProc[5]
                else:
                    comm.send(dataToProc, dest=rank_send, tag=rank_send)
    else:
        dataFromRoot = comm.recv(source=0, tag=rank)
        simulation.options.noOfSurfaces = dataFromRoot[0]
        simulation.options.surfaceNames = dataFromRoot[1]
        simulation.options.surfaceNodes = dataFromRoot[2]
        simulation.options.surfaceInvList = dataFromRoot[3]
        simulation.options.surfaceOutList = dataFromRoot[4]
        simulation.options.obstacleFlag = dataFromRoot[5]
    comm.Barrier()
    if rank == 0:
        print('Distributing nodes to sub-domains done!', flush=True)
