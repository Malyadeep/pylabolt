import numpy as np
import numba


def proc_boundary(Nx, Ny, data, data_send_topBottom,
                  data_recv_topBottom, data_send_leftRight,
                  data_recv_leftRight, nx, ny, nProc_x, nProc_y, comm):
    current_rank = nx * nProc_y + ny
    if ny % 2 == 0 and nProc_y > 1:
        # Even top
        nx_send = (nx + nProc_x) % nProc_x
        ny_send = (ny + 1 + nProc_y) % nProc_y
        rank_send = nx_send * nProc_y + ny_send
        sendLims = (0, Nx, Ny - 2, Ny - 1, Nx, Ny)
        sendCopy(data, data_send_topBottom, *sendLims)
        comm.Send(data_send_topBottom, dest=rank_send,
                  tag=current_rank*rank_send)
        comm.Recv(data_recv_topBottom, source=rank_send,
                  tag=rank_send*current_rank)
        recvLims = (0, Nx, Ny - 1, Ny, Nx, Ny)
        recvCopy(data, data_recv_topBottom, *recvLims)
        # Even bottom
        nx_send = (nx + nProc_x) % nProc_x
        ny_send = (ny - 1 + nProc_y) % nProc_y
        rank_send = nx_send * nProc_y + ny_send
        comm.Recv(data_recv_topBottom, source=rank_send,
                  tag=rank_send*current_rank)
        recvLims = (0, Nx, 0, 1, Nx, Ny)
        recvCopy(data, data_recv_topBottom, *recvLims)
        sendLims = (0, Nx, 1, 2, Nx, Ny)
        sendCopy(data, data_send_topBottom, *sendLims)
        comm.Send(data_send_topBottom, dest=rank_send,
                  tag=current_rank*rank_send)
    elif ny % 2 != 0 and nProc_y > 1:
        # Odd bottom
        nx_send = (nx + nProc_x) % nProc_x
        ny_send = (ny - 1 + nProc_y) % nProc_y
        rank_send = nx_send * nProc_y + ny_send
        comm.Recv(data_recv_topBottom, source=rank_send,
                  tag=rank_send*current_rank)
        recvLims = (0, Nx, 0, 1, Nx, Ny)
        recvCopy(data, data_recv_topBottom, *recvLims)
        sendLims = (0, Nx, 1, 2, Nx, Ny)
        sendCopy(data, data_send_topBottom, *sendLims)
        comm.Send(data_send_topBottom, dest=rank_send,
                  tag=current_rank*rank_send)
        # Odd top
        nx_send = (nx + nProc_x) % nProc_x
        ny_send = (ny + 1 + nProc_y) % nProc_y
        rank_send = nx_send * nProc_y + ny_send
        sendLims = (0, Nx, Ny - 2, Ny - 1, Nx, Ny)
        sendCopy(data, data_send_topBottom, *sendLims)
        comm.Send(data_send_topBottom, dest=rank_send,
                  tag=current_rank*rank_send)
        comm.Recv(data_recv_topBottom, source=rank_send,
                  tag=rank_send*current_rank)
        recvLims = (0, Nx, Ny - 1, Ny, Nx, Ny)
        recvCopy(data, data_recv_topBottom, *recvLims)
    elif nProc_y == 1:
        # top boundary copy to bottom boundary
        sendLims = (0, Nx, Ny - 2, Ny - 1, Nx, Ny)
        sendCopy(data, data_send_topBottom, *sendLims)
        data_recv_topBottom = np.copy(data_send_topBottom)
        recvLims = (0, Nx, 0, 1, Nx, Ny)
        recvCopy(data, data_recv_topBottom, *recvLims)
        # bottom boundary copy to top boundary
        sendLims = (0, Nx, 1, 2, Nx, Ny)
        sendCopy(data, data_send_topBottom, *sendLims)
        data_recv_topBottom = np.copy(data_send_topBottom)
        recvLims = (0, Nx, Ny - 1, Ny, Nx, Ny)
        recvCopy(data, data_recv_topBottom, *recvLims)
    comm.Barrier()
    if nx % 2 == 0 and nProc_x > 1:
        # Even right
        nx_send = (nx + 1 + nProc_x) % nProc_x
        ny_send = (ny + nProc_y) % nProc_y
        rank_send = nx_send * nProc_y + ny_send
        sendLims = (Nx - 2, Nx - 1, 0, Ny, Nx, Ny)
        sendCopy(data, data_send_leftRight, *sendLims)
        comm.Send(data_send_leftRight, dest=rank_send,
                  tag=current_rank*rank_send)
        comm.Recv(data_recv_leftRight, source=rank_send,
                  tag=rank_send*current_rank)
        recvLims = (Nx - 1, Nx, 0, Ny, Nx, Ny)
        recvCopy(data, data_recv_leftRight, *recvLims)
        # Even left
        nx_send = (nx - 1 + nProc_x) % nProc_x
        ny_send = (ny + nProc_y) % nProc_y
        rank_send = nx_send * nProc_y + ny_send
        comm.Recv(data_recv_leftRight, source=rank_send,
                  tag=rank_send*current_rank)
        recvLims = (0, 1, 0, Ny, Nx, Ny)
        recvCopy(data, data_recv_leftRight, *recvLims)
        sendLims = (1, 2, 0, Ny, Nx, Ny)
        sendCopy(data, data_send_leftRight, *sendLims)
        comm.Send(data_send_leftRight, dest=rank_send,
                  tag=current_rank*rank_send)
    elif nx % 2 != 0 and nProc_x > 1:
        # Odd left
        nx_send = (nx - 1 + nProc_x) % nProc_x
        ny_send = (ny + nProc_y) % nProc_y
        rank_send = nx_send * nProc_y + ny_send
        comm.Recv(data_recv_leftRight, source=rank_send,
                  tag=rank_send*current_rank)
        recvLims = (0, 1, 0, Ny, Nx, Ny)
        recvCopy(data, data_recv_leftRight, *recvLims)
        sendLims = (1, 2, 0, Ny, Nx, Ny)
        sendCopy(data, data_send_leftRight, *sendLims)
        comm.Send(data_send_leftRight, dest=rank_send,
                  tag=current_rank*rank_send)
        # Odd right
        nx_send = (nx + 1 + nProc_x) % nProc_x
        ny_send = (ny + nProc_y) % nProc_y
        rank_send = nx_send * nProc_y + ny_send
        sendLims = (Nx - 2, Nx - 1, 0, Ny, Nx, Ny)
        sendCopy(data, data_send_leftRight, *sendLims)
        comm.Send(data_send_leftRight, dest=rank_send,
                  tag=current_rank*rank_send)
        comm.Recv(data_recv_leftRight, source=rank_send,
                  tag=rank_send*current_rank)
        recvLims = (Nx - 1, Nx, 0, Ny, Nx, Ny)
        recvCopy(data, data_recv_leftRight, *recvLims)
    elif nProc_x == 1:
        # right boundary copy to left boundary
        sendLims = (Nx - 2, Nx - 1, 0, Ny, Nx, Ny)
        sendCopy(data, data_send_leftRight, *sendLims)
        data_recv_leftRight = np.copy(data_send_leftRight)
        recvLims = (0, 1, 0, Ny, Nx, Ny)
        recvCopy(data, data_recv_leftRight, *recvLims)
        # left boundary copy to right boundary
        sendLims = (1, 2, 0, Ny, Nx, Ny)
        sendCopy(data, data_send_leftRight, *sendLims)
        data_recv_leftRight = np.copy(data_send_leftRight)
        recvLims = (Nx - 1, Nx, 0, Ny, Nx, Ny)
        recvCopy(data, data_recv_leftRight, *recvLims)
    comm.Barrier()


@numba.njit
def sendCopy(data, data_send, Nx_i, Nx_f, Ny_i, Ny_f, Nx, Ny):
    itr = 0
    for i in range(Nx_i, Nx_f):
        for j in range(Ny_i, Ny_f):
            ind = i * Ny + j
            data_send[itr, :] = data[ind, :]
            itr += 1


@numba.njit
def recvCopy(data, data_recv, Nx_i, Nx_f, Ny_i, Ny_f, Nx, Ny):
    itr = 0
    for i in range(Nx_i, Nx_f):
        for j in range(Ny_i, Ny_f):
            ind = i * Ny + j
            data[ind, :] = data_recv[itr, :]
            itr += 1


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
            resU = np.sqrt(sum_u_sq/(sum_u + 1e-10))
            resV = np.sqrt(sum_v_sq/(sum_v + 1e-10))
            resRho = np.sqrt(sum_rho_sq/(sum_rho + 1e-10))
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
        resU = np.sqrt(u_err_sq/(u_sq + 1e-10))
        resV = np.sqrt(v_err_sq/(v_sq + 1e-10))
        resRho = np.sqrt(rho_err_sq/(rho_sq + 1e-10))
        return resU, resV, resRho


def gatherForces_mpi(forces, comm, rank, size, precision):
    if rank == 0:
        sumF = np.zeros((len(forces.surfaceNamesGlobal), 2),
                        dtype=precision)
        for i in range(size):
            if i == 0:
                for itr, name in enumerate(forces.surfaceNamesGlobal):
                    for itr_local, local_name in \
                            enumerate(forces.surfaceNames):
                        if name == local_name:
                            sumF[itr, 0] += forces.forces[itr_local][0]
                            sumF[itr, 1] += forces.forces[itr_local][1]
            else:
                surfaceNames_local = comm.recv(source=i, tag=1*i)
                forces_local = comm.recv(source=i, tag=2*i)
                for itr, name in enumerate(forces.surfaceNamesGlobal):
                    for itr_local, local_name in \
                            enumerate(surfaceNames_local):
                        if name == local_name:
                            sumF[itr, 0] += forces_local[itr_local][0]
                            sumF[itr, 1] += forces_local[itr_local][1]
        return forces.surfaceNamesGlobal, sumF
    else:
        comm.send(forces.surfaceNames, dest=0, tag=1*rank)
        comm.send(forces.forces, dest=0, tag=2*rank)
        return 0, 0
