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


def computeResiduals(residues, tempResidues, comm, rank, size):
    np.array(residues)
    if size > 1:
        if rank == 0:
            sum_u, sum_v = residues[0], residues[1]
            sum_rho = residues[2]
            sum_u_sq, sum_v_sq = residues[3], residues[4]
            sum_rho_sq = residues[5]
            for i in range(1, size):
                comm.Recv(tempResidues, source=i, tag=1*i)
                sum_u += tempResidues[0]
                sum_v += tempResidues[1]
                sum_rho += tempResidues[2]
                sum_u_sq += tempResidues[3]
                sum_v_sq += tempResidues[4]
                sum_rho_sq += tempResidues[5]
            if np.isclose(sum_u, 0, rtol=1e-10):
                sum_u += 1e-10
            if np.isclose(sum_v, 0, rtol=1e-10):
                sum_v += 1e-10
            if np.isclose(sum_rho, 0, rtol=1e-10):
                sum_rho += 1e-10
            resU = np.sqrt(sum_u_sq/(sum_u))
            resV = np.sqrt(sum_v_sq/(sum_v))
            resRho = np.sqrt(sum_rho_sq/(sum_rho))
            return resU, resV, resRho
        else:
            comm.Send(residues, dest=0, tag=1*rank)
            return 0, 0, 0
    else:
        if np.isclose(residues[0], 0, rtol=1e-10):
            residues[0] += 1e-10
        if np.isclose(residues[1], 0, rtol=1e-10):
            residues[1] += 1e-10
        if np.isclose(residues[2], 0, rtol=1e-10):
            residues[2] += 1e-10
        resU = np.sqrt(residues[3]/(residues[0]))
        resV = np.sqrt(residues[4]/(residues[1]))
        resRho = np.sqrt(residues[5]/(residues[2]))
        return resU, resV, resRho


def gatherForcesTorque_mpi(options, comm, rank, size, precision):
    if rank == 0:
        sumF = np.zeros((len(options.surfaceNamesGlobal), 2),
                        dtype=precision)
        sumT = np.zeros((len(options.surfaceNamesGlobal)),
                        dtype=precision)
        for i in range(size):
            if i == 0:
                for itr, name in enumerate(options.surfaceNamesGlobal):
                    for itr_local, local_name in \
                            enumerate(options.surfaceNames):
                        if name == local_name:
                            sumF[itr, 0] += options.forces[itr_local][0]
                            sumF[itr, 1] += options.forces[itr_local][1]
                            sumT[itr] += options.torque[itr_local]
            else:
                surfaceNames_local = comm.recv(source=i, tag=1*i)
                forces_local = comm.recv(source=i, tag=2*i)
                torque_local = comm.recv(source=i, tag=3*i)
                for itr, name in enumerate(options.surfaceNamesGlobal):
                    for itr_local, local_name in \
                            enumerate(surfaceNames_local):
                        if name == local_name:
                            sumF[itr, 0] += forces_local[itr_local][0]
                            sumF[itr, 1] += forces_local[itr_local][1]
                            sumT[itr] += torque_local[itr_local]
        return options.surfaceNamesGlobal, sumF, sumT
    else:
        comm.send(options.surfaceNames, dest=0, tag=1*rank)
        comm.send(options.forces, dest=0, tag=2*rank)
        comm.send(options.torque, dest=0, tag=3*rank)
        return 0, 0, 0
