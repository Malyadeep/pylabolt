import numpy as np
import os
import sys

from pylabolt.parallel.MPI_comm import proc_boundary, proc_boundaryScalars


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
            for itr, field in enumerate(fields.fieldList):
                if field == 'u':
                    fields.u[ind_write, 0] = fields_temp.u[ind, 0]
                    fields.u[ind_write, 1] = fields_temp.u[ind, 1]
                if field == 'rho':
                    fields.rho[ind_write] = fields_temp.rho[ind]
                if field == 'phi':
                    fields.phi[ind_write] = fields_temp.phi[ind]
                fields.boundaryNode[ind_write] = fields_temp.boundaryNode[ind]
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
    for fieldName in fields.fieldList:
        if fieldName == 'u':
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
        if (fieldName == 'rho' or fieldName == 'phi' or fieldName == 'T' or
                fieldName == 'boundaryNode'):
            initial_send_topBottom = np.zeros((mesh.Nx + 2),
                                              dtype=precision)
            initial_recv_topBottom = np.zeros((mesh.Nx + 2),
                                              dtype=precision)
            initial_send_leftRight = np.zeros((mesh.Ny + 2),
                                              dtype=precision)
            initial_recv_leftRight = np.zeros((mesh.Ny + 2),
                                              dtype=precision)
            if fieldName == 'rho':
                field = fields.rho
            elif fieldName == 'phi':
                field = fields.phi
            elif fieldName == 'T':
                field = fields.T
            elif fieldName == 'boundaryNode':
                field = fields.boundaryNode
            args = (mesh.Nx, mesh.Ny,
                    field, initial_send_topBottom,
                    initial_recv_topBottom, initial_send_leftRight,
                    initial_recv_leftRight, mpiParams.nx,
                    mpiParams.ny, mpiParams.nProc_x,
                    mpiParams.nProc_y, comm)
            if fieldName == 'phi':
                proc_boundaryScalars(*args, inner=True)
            else:
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
                nbList = []
                outDirections = []
                invDirections = []
                boundaryFuncFluid = []
                boundaryFuncPhase = []
                boundaryFuncT = []
                boundaryTypeFluid = []
                boundaryTypePhase = []
                boundaryTypeT = []
                fluid, phase = False, False
                T = False
                for itr in range(boundary.noOfBoundaries):
                    flag = 0
                    tempFaces = []
                    tempNb = []
                    tempVector = []
                    for numFace in range(boundary.faceList[itr].shape[0]):
                        ind = boundary.faceList[itr][numFace]
                        ind_nb = boundary.nbList[itr][numFace]
                        i = int(ind / mesh.Ny_global)
                        j = int(ind % mesh.Ny_global)
                        i_nb = int(ind_nb / mesh.Ny_global)
                        j_nb = int(ind_nb % mesh.Ny_global)
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
                            i_nb_local = int(i_nb % Nx_local)
                            j_nb_local = int(j_nb % Ny_local)
                            if nx == mpiParams.nProc_x - 1:
                                Nx_local = N_local[nx, ny, 0]
                            if ny == mpiParams.nProc_y - 1:
                                Ny_local = N_local[nx, ny, 1]
                            tempFaces.append((i_local + 1) * (Ny_local + 2)
                                             + (j_local + 1))
                            tempNb.append((i_nb_local + 1) * (Ny_local + 2)
                                          + (j_nb_local + 1))
                            if boundary.boundaryTypeFluid[itr] == 'variableU':
                                tempVector.append(boundary.boundaryVector
                                                  [itr][numFace])
                    if flag != 0:
                        noOfBoundaries += 1
                        faceList.append(np.array(tempFaces, dtype=np.int64))
                        nbList.append(np.array(tempNb, dtype=np.int64))
                        if boundary.fluid is True:
                            fluid = True
                            boundaryFuncFluid.\
                                append(boundary.boundaryFuncFluid[itr])
                            boundaryTypeFluid.\
                                append(boundary.boundaryTypeFluid[itr])
                        if boundary.phase is True:
                            phase = True
                            boundaryFuncPhase.\
                                append(boundary.boundaryFuncPhase[itr])
                            boundaryTypePhase.\
                                append(boundary.boundaryTypePhase[itr])
                        if boundary.T is True:
                            T = True
                            boundaryFuncT.append(boundary.boundaryFuncT[itr])
                            boundaryTypeT.\
                                append(boundary.boundaryTypeT[itr])
                        outDirections.append(boundary.outDirections[itr])
                        invDirections.append(boundary.invDirections[itr])
                dataToProc = (noOfBoundaries, faceList, nbList,
                              outDirections, invDirections,
                              boundaryFuncFluid,
                              boundaryFuncPhase, boundaryFuncT,
                              boundaryTypeFluid,
                              boundaryTypePhase, boundaryTypeT,
                              fluid, phase, T)
                rank_send = nx * mpiParams.nProc_y + ny
                if rank_send == 0:
                    boundary.noOfBoundaries = dataToProc[0]
                    boundary.faceList = dataToProc[1]
                    boundary.nbList = dataToProc[2]
                    boundary.outDirections = dataToProc[3]
                    boundary.invDirections = dataToProc[4]
                    boundary.boundaryFuncFluid = dataToProc[5]
                    boundary.boundaryFuncPhase = dataToProc[6]
                    boundary.boundaryFuncT = dataToProc[7]
                    boundary.boundaryTypeFluid = dataToProc[8]
                    boundary.boundaryTypePhase = dataToProc[9]
                    boundary.boundaryTypeT = dataToProc[10]
                    boundary.fluid = dataToProc[11]
                    boundary.phase = dataToProc[12]
                    boundary.T = dataToProc[13]
                else:
                    comm.send(dataToProc, dest=rank_send, tag=rank_send)
    else:
        dataFromRoot = comm.recv(source=0, tag=rank)
        boundary.noOfBoundaries = dataFromRoot[0]
        boundary.faceList = dataFromRoot[1]
        boundary.nbList = dataFromRoot[2]
        boundary.outDirections = dataFromRoot[3]
        boundary.invDirections = dataFromRoot[4]
        boundary.boundaryFuncFluid = dataFromRoot[5]
        boundary.boundaryFuncPhase = dataFromRoot[6]
        boundary.boundaryFuncT = dataFromRoot[7]
        boundary.boundaryTypeFluid = dataFromRoot[8]
        boundary.boundaryTypePhase = dataFromRoot[9]
        boundary.boundaryTypeT = dataFromRoot[10]
        boundary.fluid = dataFromRoot[11]
        boundary.phase = dataFromRoot[12]
        boundary.T = dataFromRoot[13]
    comm.Barrier()
    if rank == 0:
        print('Distributing boundaries done!', flush=True)


def distributeForceNodes_mpi(options, mpiParams, mesh, rank, size, precision,
                             comm):
    if rank == 0:
        print('\nMPI option with force computation selected', flush=True)
        print('Distributing nodes to sub-domains...', flush=True)
    N_local = mpiParams.N_local
    if rank == 0:
        for nx in range(mpiParams.nProc_x - 1, -1, -1):
            for ny in range(mpiParams.nProc_y - 1, -1, -1):
                Nx_local = N_local[nx, ny, 0]
                Ny_local = N_local[nx, ny, 1]
                noOfSurfaces = 0
                surfaceNames = []
                surfaceNodes = []
                solidNbNodes = []
                surfaceInvList = []
                surfaceOutList = []
                obstacleFlag = []
                for itr in range(options.noOfSurfaces):
                    flag = 0
                    tempFaces = []
                    tempSolidFaces = []
                    for num in range(options.surfaceNodes[itr].shape[0]):
                        ind = options.surfaceNodes[itr][num]
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
                    for num in range(options.solidNbNodes[itr].shape[0]):
                        ind = options.solidNbNodes[itr][num]
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
                            tempSolidFaces.append((i_local + 1) *
                                                  (Ny_local + 2)
                                                  + (j_local + 1))
                    if flag != 0:
                        noOfSurfaces += 1
                        surfaceNodes.append(np.array(tempFaces,
                                            dtype=np.int64))
                        solidNbNodes.append(np.array(tempSolidFaces,
                                            dtype=np.int64))
                        surfaceNames.append(options.
                                            surfaceNames[itr])
                        surfaceInvList.append(options.
                                              surfaceInvList[itr])
                        surfaceOutList.append(options.
                                              surfaceOutList[itr])
                        obstacleFlag.append(options.
                                            obstacleFlag[itr])
                dataToProc = (noOfSurfaces, surfaceNames, surfaceNodes,
                              solidNbNodes, surfaceInvList,
                              surfaceOutList, obstacleFlag)
                rank_send = nx * mpiParams.nProc_y + ny
                if rank_send == 0:
                    options.noOfSurfaces = dataToProc[0]
                    options.surfaceNames = dataToProc[1]
                    options.surfaceNodes = dataToProc[2]
                    options.solidNbNodes = dataToProc[3]
                    options.surfaceInvList = dataToProc[4]
                    options.surfaceOutList = dataToProc[5]
                    options.obstacleFlag = dataToProc[6]
                else:
                    comm.send(dataToProc, dest=rank_send, tag=rank_send)
    else:
        dataFromRoot = comm.recv(source=0, tag=rank)
        options.noOfSurfaces = dataFromRoot[0]
        options.surfaceNames = dataFromRoot[1]
        options.surfaceNodes = dataFromRoot[2]
        options.solidNbNodes = dataFromRoot[3]
        options.surfaceInvList = dataFromRoot[4]
        options.surfaceOutList = dataFromRoot[5]
        options.obstacleFlag = dataFromRoot[6]
    comm.Barrier()
    if rank == 0:
        print('Distributing nodes to sub-domains done!', flush=True)
