import numpy as np
import os
import sys
import numba


def proc_boundary(Nx_local, Ny_local, c, invList, f_new, f, nx, ny,
                  nProc_x, nProc_y, comm):
    current_rank = nx * nProc_y + ny
    for i in range(Nx_local):
        if ny % 2 == 0:
            if ny < nProc_y - 1:
                for j in range(Ny_local - 1, Ny_local):
                    for k in range(1, 9):
                        if j + int(c[k, 1]) > Ny_local - 1:
                            rank_send = nx * nProc_y + ny + 1
                            ind = i * Ny_local + j
                            comm.send(f_new[ind, k], dest=rank_send,
                                      tag=1*i*rank_send)
                            f_new[ind, invList[k]] = \
                                comm.recv(source=rank_send,
                                          tag=3*i*current_rank
                                          )
            if ny > 0:
                for j in range(0, 1):
                    for k in range(1, 9):
                        if j - int(c[k, 1]) < 0:
                            rank_send = nx * nProc_y + ny - 1
                            ind = i * Ny_local + j
                            f_new[ind, k] = \
                                comm.recv(source=rank_send,
                                          tag=4*i*current_rank)
                            comm.send(f_new[ind, invList[k]], dest=rank_send,
                                      tag=2*i*rank_send)
        elif ny % 2 != 0:
            if ny > 0:
                for j in range(0, 1):
                    for k in range(1, 9):
                        if j - int(c[k, 1]) < 0:
                            rank_send = nx * nProc_y + ny - 1
                            ind = i * Ny_local + j
                            f_new[ind, k] = \
                                comm.recv(source=rank_send,
                                          tag=1*i*current_rank
                                          )
                            comm.send(f_new[ind, invList[k]], dest=rank_send,
                                      tag=3*i*rank_send)
            if ny < nProc_y - 1:
                for j in range(Ny_local - 1, Ny_local):
                    for k in range(1, 9):
                        if j + int(c[k, 1]) > Ny_local + 1:
                            rank_send = nx * nProc_y + ny + 1
                            ind = i * Ny_local + j
                            comm.send(f_new[ind, k], dest=rank_send,
                                      tag=4*i*rank_send)
                            f_new[ind, invList[k]] = \
                                comm.recv(source=rank_send,
                                          tag=2*i*current_rank
                                          )
    comm.Barrier()
    for j in range(Ny_local):
        if nx % 2 == 0:
            if nx < nProc_x - 1:
                for i in range(Nx_local - 1, Nx_local):
                    for k in range(1, 9):
                        if i + int(c[k, 0]) > Nx_local - 1:
                            rank_send = (nx + 1) * nProc_y + ny
                            ind = i * Ny_local + j
                            comm.send(f_new[ind, k], dest=rank_send,
                                      tag=1*j*rank_send)
                            f_new[ind, invList[k]] = \
                                comm.recv(source=rank_send,
                                          tag=3*j*current_rank
                                          )
            if nx > 0:
                for i in range(0, 1):
                    for k in range(1, 9):
                        if i - int(c[k, 0]) < 0:
                            rank_send = (nx - 1) * nProc_y + ny
                            ind = i * Ny_local + j
                            f_new[ind, k] = \
                                comm.recv(source=rank_send,
                                          tag=4*j*current_rank
                                          )
                            comm.send(f_new[ind, invList[k]], dest=rank_send,
                                      tag=2*j*rank_send)
        elif nx % 2 != 0:
            if nx > 0:
                for i in range(0, 1):
                    for k in range(1, 9):
                        if i - int(c[k, 0]) < 0:
                            rank_send = (nx - 1) * nProc_y + ny
                            ind = i * Ny_local + j
                            f_new[ind, k] = \
                                comm.recv(source=rank_send,
                                          tag=1*j*current_rank
                                          )
                            comm.send(f_new[ind, invList[k]], dest=rank_send,
                                      tag=3*j*rank_send)
            if nx < nProc_x - 1:
                for i in range(Nx_local - 1, Nx_local):
                    for k in range(1, 9):
                        if i + int(c[k, 0]) > Nx_local - 1:
                            rank_send = (nx + 1) * nProc_y + ny
                            ind = i * Ny_local + j
                            comm.send(f_new[ind, k], dest=rank_send,
                                      tag=4*j*rank_send)
                            f_new[ind, invList[k]] = \
                                comm.recv(source=rank_send,
                                          tag=2*j*current_rank
                                          )
    comm.Barrier()


@numba.njit
def proc_copy(Nx_local, Ny_local, c, f_new):
    for i in range(1, Nx_local - 1):
        for j in range(Ny_local - 2, Ny_local - 1):
            for k in range(1, 9):
                if (j - int(c[k, 1]) > Ny_local - 2
                        and i - int(c[k, 0]) > Nx_local - 2):
                    ind, ind_nb = i * Ny_local + j, (i + 1) * Ny_local + j + 1
                    f_new[ind, k] = f_new[ind_nb, k]
                elif j - int(c[k, 1]) > Ny_local - 2 and i - int(c[k, 0]) < 1:
                    ind, ind_nb = i * Ny_local + j, (i - 1) * Ny_local + j + 1
                    f_new[ind, k] = f_new[ind_nb, k]
                elif j - int(c[k, 1]) > Ny_local - 2:
                    ind, ind_nb = i * Ny_local + j, i * Ny_local + j + 1
                    f_new[ind, k] = f_new[ind_nb, k]
        for j in range(1, 2):
            for k in range(1, 9):
                if (j - int(c[k, 1]) < 1 and i - int(c[k, 0]) < 1):
                    ind, ind_nb = i * Ny_local + j, (i - 1) * Ny_local + j - 1
                    f_new[ind, k] = f_new[ind_nb, k]
                elif j - int(c[k, 1]) < 1 and i - int(c[k, 0]) > Nx_local - 2:
                    ind, ind_nb = i * Ny_local + j, (i + 1) * Ny_local + j - 1
                    f_new[ind, k] = f_new[ind_nb, k]
                elif j - int(c[k, 1]) < 1:
                    ind, ind_nb = i * Ny_local + j, i * Ny_local + j - 1
                    f_new[ind, k] = f_new[ind_nb, k]
    for j in range(1, Ny_local - 1):
        for i in range(1, 2):
            for k in range(1, 9):
                if i - int(c[k, 0]) < 1 and j - int(c[k, 1]) < 1:
                    ind, ind_nb = i * Ny_local + j, (i - 1) * Ny_local + j - 1
                    f_new[ind, k] = f_new[ind_nb, k]
                elif i - int(c[k, 0]) < 1 and j - int(c[k, 1]) > Ny_local - 2:
                    ind, ind_nb = i * Ny_local + j, (i - 1) * Ny_local + j + 1
                    f_new[ind, k] = f_new[ind_nb, k]
                elif i - int(c[k, 0]) < 1:
                    ind, ind_nb = i * Ny_local + j, (i - 1) * Ny_local + j
                    f_new[ind, k] = f_new[ind_nb, k]
        for i in range(Nx_local - 2, Nx_local - 1):
            for k in range(1, 9):
                if i - int(c[k, 0]) > Nx_local - 2 and j - int(c[k, 1]) < 1:
                    ind, ind_nb = i * Ny_local + j, (i + 1) * Ny_local + j - 1
                    f_new[ind, k] = f_new[ind_nb, k]
                elif (i - int(c[k, 0]) > Nx_local - 2 and
                      j - int(c[k, 1]) > Ny_local - 2):
                    ind, ind_nb = i * Ny_local + j, (i + 1) * Ny_local + j + 1
                    f_new[ind, k] = f_new[ind_nb, k]
                elif i - int(c[k, 0]) > Nx_local - 2:
                    ind, ind_nb = i * Ny_local + j, (i + 1) * Ny_local + j
                    f_new[ind, k] = f_new[ind_nb, k]


def computeResiduals(u_err_sq, u_sq, v_err_sq, v_sq,
                     rho_err_sq, rho_sq, comm, rank, size):
    if size > 1:
        if rank == 0:
            sum_u, sum_v = u_sq, v_sq
            sum_rho = rho_sq
            sum_u_sq, sum_v_sq = u_err_sq, v_err_sq
            sum_rho_sq = rho_err_sq
            for i in range(1, size):
                temp_sum_u = comm.recv(source=i, tag=1*i)
                temp_sum_v = comm.recv(source=i, tag=2*i)
                temp_sum_rho = comm.recv(source=i, tag=3*i)
                temp_sum_u_sq = comm.recv(source=i, tag=4*i)
                temp_sum_v_sq = comm.recv(source=i, tag=5*i)
                temp_sum_rho_sq = comm.recv(source=i, tag=6*i)
                sum_u += temp_sum_u
                sum_v += temp_sum_v
                sum_rho += temp_sum_rho
                sum_u_sq += temp_sum_u_sq
                sum_v_sq += temp_sum_v_sq
                sum_rho_sq += temp_sum_rho_sq
            resU = np.sqrt(sum_u_sq/(sum_u + 1e-9))
            resV = np.sqrt(sum_v_sq/(sum_v + 1e-9))
            resRho = np.sqrt(sum_rho_sq/(sum_rho + 1e-9))
            return resU, resV, resRho
        else:
            comm.send(u_sq, dest=0, tag=1*rank)
            comm.send(v_sq, dest=0, tag=2*rank)
            comm.send(rho_sq, dest=0, tag=3*rank)
            comm.send(u_err_sq, dest=0, tag=4*rank)
            comm.send(v_err_sq, dest=0, tag=5*rank)
            comm.send(rho_err_sq, dest=0, tag=6*rank)
            return 0, 0, 0
    else:
        resU = np.sqrt(u_err_sq/(u_sq + 1e-8))
        resV = np.sqrt(v_err_sq/(v_sq + 1e-8))
        resRho = np.sqrt(rho_err_sq/(rho_sq + 1e-8))
        return resU, resV, resRho


@numba.njit
def gather_copy(u_all, rho_all, solid_all, u_temp, rho_temp,
                solid_temp, N_sub, nx, ny, Ny_global):
    i_write, j_write = 0, 0
    for i in range(int(nx * (N_sub[0] - 2)), int((nx + 1) * (N_sub[0] - 2))):
        for j in range(int(ny * (N_sub[1] - 2)),
                       int((ny + 1) * (N_sub[1] - 2))):
            ind = int(i * Ny_global + j)
            ind_write = int((i_write + 1) * N_sub[1] + (j_write + 1))
            u_all[ind, 0] = u_temp[ind_write, 0]
            u_all[ind, 1] = u_temp[ind_write, 1]
            rho_all[ind] = rho_temp[ind_write]
            solid_all[ind] = solid_temp[ind_write]
            j_write += 1
        j_write = 0
        i_write += 1


def gather(u, rho, solid, rank, comm, nProc_x, nProc_y,
           Nx, Ny, Nx_local, Ny_local, precision):
    if rank == 0:
        u_all = np.zeros((Nx * Ny, 2), dtype=precision)
        rho_all = np.ones((Nx * Ny), dtype=precision)
        solid_all = np.zeros((Nx * Ny), dtype=np.int32)
        fieldsToGather = (u_all, rho_all, solid_all)
        for nx in range(nProc_x):
            for ny in range(nProc_y):
                current_rank = int(nx * nProc_y + ny)
                if current_rank == 0:
                    N_sub = np.array([Nx_local, Ny_local], dtype=np.int32)
                    gather_copy(*fieldsToGather, u, rho, solid, N_sub, nx,
                                ny, Ny)
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
                                N_sub, nx, ny, Ny)
        return u_all, rho_all, solid_all
    else:
        comm.Send(np.array([Nx_local, Ny_local], dtype=np.int32),
                  dest=0, tag=1*rank)
        comm.Send(u, dest=0, tag=2*rank)
        comm.Send(rho, dest=0, tag=3*rank)
        comm.Send(solid, dest=0, tag=4*rank)
        return 0, 0, 0


def decompose(mesh, rank, size):
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
    return mpiParams


class decomposeParams:
    def __init__(self, nProc_x, nProc_y):
        self.nProc_x = nProc_x
        self.nProc_y = nProc_y
        self.nx = 0
        self.ny = 0


def distributeBoundaries_mpi(boundary, mpiParams, mesh, rank, size, comm):
    if rank == 0:
        for nx in range(mpiParams.nProc_x - 1, -1, -1):
            for ny in range(mpiParams.nProc_y - 1, -1, -1):
                noOfBoundaries = 0
                faceList = []
                boundaryVector = []
                boundaryScalar = []
                outDirections = []
                invDirections = []
                boundaryFunc = []
                Nx_local = int(np.ceil(mesh.Nx_global/mpiParams.nProc_x))
                Ny_local = int(np.ceil(mesh.Ny_global/mpiParams.nProc_y))
                if nx == mpiParams.nProc_x - 1:
                    Nx_local = mesh.Nx_global - \
                        nx * int(np.ceil(mesh.Nx_global/mpiParams.nProc_x))
                if ny == mpiParams.nProc_y - 1:
                    Ny_local = mesh.Ny_global - \
                        ny * int(np.ceil(mesh.Ny_global/mpiParams.nProc_y))
                for itr in range(boundary.noOfBoundaries):
                    flag = 0
                    tempFaces = []
                    for ind in boundary.faceList[itr]:
                        i = int(ind / mesh.Ny_global)
                        j = int(ind % mesh.Nx_global)
                        if (i >= nx * Nx_local and i < (nx + 1) * Nx_local
                                and j >= ny * Ny_local and j < (ny + 1) *
                                Ny_local):
                            flag = 1
                            i_local = int(i % Nx_local)
                            j_local = int(j % Ny_local)
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
