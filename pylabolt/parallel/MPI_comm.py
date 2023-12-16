import numpy as np
import numba


def proc_boundary(Nx, Ny, data, data_send_topBottom,
                  data_recv_topBottom, data_send_leftRight,
                  data_recv_leftRight, nx, ny, nProc_x, nProc_y, comm,
                  inner=False):
    current_rank = nx * nProc_y + ny
    shape = data_recv_leftRight.shape
    if len(shape) > 1:
        sendCopy = sendCopy_vector
        recvCopy = recvCopy_vector
    else:
        sendCopy = sendCopy_scalar
        recvCopy = recvCopy_scalar
    if ny % 2 == 0 and nProc_y > 1:
        # Even top
        nx_send = (nx + nProc_x) % nProc_x
        ny_send = (ny + 1 + nProc_y) % nProc_y
        rank_send = nx_send * nProc_y + ny_send
        if ny == nProc_y - 1 and inner is True:
            sendLims = (0, Nx, Ny - 3, Ny - 2, Nx, Ny)
            recvLims = (0, Nx, Ny - 2, Ny - 1, Nx, Ny)
        else:
            sendLims = (0, Nx, Ny - 2, Ny - 1, Nx, Ny)
            recvLims = (0, Nx, Ny - 1, Ny, Nx, Ny)
        sendCopy(data, data_send_topBottom, *sendLims)
        comm.Send(data_send_topBottom, dest=rank_send,
                  tag=current_rank*rank_send)
        comm.Recv(data_recv_topBottom, source=rank_send,
                  tag=rank_send*current_rank)
        recvCopy(data, data_recv_topBottom, *recvLims)
        # Even bottom
        nx_send = (nx + nProc_x) % nProc_x
        ny_send = (ny - 1 + nProc_y) % nProc_y
        rank_send = nx_send * nProc_y + ny_send
        comm.Recv(data_recv_topBottom, source=rank_send,
                  tag=rank_send*current_rank)
        if ny == 0 and inner is True:
            recvLims = (0, Nx, 1, 2, Nx, Ny)
            sendLims = (0, Nx, 2, 3, Nx, Ny)
        else:
            recvLims = (0, Nx, 0, 1, Nx, Ny)
            sendLims = (0, Nx, 1, 2, Nx, Ny)
        recvCopy(data, data_recv_topBottom, *recvLims)
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
        if ny == nProc_y - 1 and inner is True:
            sendLims = (0, Nx, Ny - 3, Ny - 2, Nx, Ny)
            recvLims = (0, Nx, Ny - 2, Ny - 1, Nx, Ny)
        else:
            sendLims = (0, Nx, Ny - 2, Ny - 1, Nx, Ny)
            recvLims = (0, Nx, Ny - 1, Ny, Nx, Ny)
        sendCopy(data, data_send_topBottom, *sendLims)
        comm.Send(data_send_topBottom, dest=rank_send,
                  tag=current_rank*rank_send)
        comm.Recv(data_recv_topBottom, source=rank_send,
                  tag=rank_send*current_rank)
        recvCopy(data, data_recv_topBottom, *recvLims)
    elif nProc_y == 1:
        # top boundary copy to bottom boundary
        if inner is True and Ny - 3 != 0:
            sendLims = (0, Nx, Ny - 3, Ny - 2, Nx, Ny)
            recvLims = (0, Nx, 1, 2, Nx, Ny)
        else:
            sendLims = (0, Nx, Ny - 2, Ny - 1, Nx, Ny)
            recvLims = (0, Nx, 0, 1, Nx, Ny)
        sendCopy(data, data_send_topBottom, *sendLims)
        data_recv_topBottom = np.copy(data_send_topBottom)
        recvCopy(data, data_recv_topBottom, *recvLims)
        # bottom boundary copy to top boundary
        if inner is True and Ny - 3 != 0:
            sendLims = (0, Nx, 2, 3, Nx, Ny)
            recvLims = (0, Nx, Ny - 2, Ny - 1, Nx, Ny)
        else:
            sendLims = (0, Nx, 1, 2, Nx, Ny)
            recvLims = (0, Nx, Ny - 1, Ny, Nx, Ny)
        sendCopy(data, data_send_topBottom, *sendLims)
        data_recv_topBottom = np.copy(data_send_topBottom)
        recvCopy(data, data_recv_topBottom, *recvLims)
    comm.Barrier()
    if nx % 2 == 0 and nProc_x > 1:
        # Even right
        nx_send = (nx + 1 + nProc_x) % nProc_x
        ny_send = (ny + nProc_y) % nProc_y
        rank_send = nx_send * nProc_y + ny_send
        if nx == nProc_x - 1 and inner is True:
            sendLims = (Nx - 3, Nx - 2, 0, Ny, Nx, Ny)
            recvLims = (Nx - 2, Nx - 1, 0, Ny, Nx, Ny)
        else:
            sendLims = (Nx - 2, Nx - 1, 0, Ny, Nx, Ny)
            recvLims = (Nx - 1, Nx, 0, Ny, Nx, Ny)
        sendCopy(data, data_send_leftRight, *sendLims)
        comm.Send(data_send_leftRight, dest=rank_send,
                  tag=current_rank*rank_send)
        comm.Recv(data_recv_leftRight, source=rank_send,
                  tag=rank_send*current_rank)
        recvCopy(data, data_recv_leftRight, *recvLims)
        # Even left
        nx_send = (nx - 1 + nProc_x) % nProc_x
        ny_send = (ny + nProc_y) % nProc_y
        rank_send = nx_send * nProc_y + ny_send
        if nx == 0 and inner is True:
            recvLims = (1, 2, 0, Ny, Nx, Ny)
            sendLims = (2, 3, 0, Ny, Nx, Ny)
        else:
            recvLims = (0, 1, 0, Ny, Nx, Ny)
            sendLims = (1, 2, 0, Ny, Nx, Ny)
        comm.Recv(data_recv_leftRight, source=rank_send,
                  tag=rank_send*current_rank)
        recvCopy(data, data_recv_leftRight, *recvLims)
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
        if nx == nProc_x - 1 and inner is True:
            sendLims = (Nx - 3, Nx - 2, 0, Ny, Nx, Ny)
            recvLims = (Nx - 2, Nx - 1, 0, Ny, Nx, Ny)
        else:
            sendLims = (Nx - 2, Nx - 1, 0, Ny, Nx, Ny)
            recvLims = (Nx - 1, Nx, 0, Ny, Nx, Ny)
        sendCopy(data, data_send_leftRight, *sendLims)
        comm.Send(data_send_leftRight, dest=rank_send,
                  tag=current_rank*rank_send)
        comm.Recv(data_recv_leftRight, source=rank_send,
                  tag=rank_send*current_rank)
        recvCopy(data, data_recv_leftRight, *recvLims)
    elif nProc_x == 1:
        # right boundary copy to left boundary
        if inner is True and Nx - 3 != 0:
            sendLims = (Nx - 3, Nx - 2, 0, Ny, Nx, Ny)
            recvLims = (1, 2, 0, Ny, Nx, Ny)
        else:
            sendLims = (Nx - 2, Nx - 1, 0, Ny, Nx, Ny)
            recvLims = (0, 1, 0, Ny, Nx, Ny)
        sendCopy(data, data_send_leftRight, *sendLims)
        data_recv_leftRight = np.copy(data_send_leftRight)
        recvCopy(data, data_recv_leftRight, *recvLims)
        # left boundary copy to right boundary
        if inner is True and Nx - 3 != 0:
            sendLims = (2, 3, 0, Ny, Nx, Ny)
            recvLims = (Nx - 2, Nx - 1, 0, Ny, Nx, Ny)
        else:
            sendLims = (1, 2, 0, Ny, Nx, Ny)
            recvLims = (Nx - 1, Nx, 0, Ny, Nx, Ny)
        sendCopy(data, data_send_leftRight, *sendLims)
        data_recv_leftRight = np.copy(data_send_leftRight)
        recvCopy(data, data_recv_leftRight, *recvLims)
    comm.Barrier()


def proc_boundaryGradTerms(Nx, Ny, data, data_send_topBottom,
                           data_recv_topBottom, data_send_leftRight,
                           data_recv_leftRight, nx, ny, nProc_x, nProc_y, comm,
                           inner=False):
    current_rank = nx * nProc_y + ny
    shape = data_recv_leftRight.shape
    if len(shape) > 1:
        sendCopy = sendCopy_vector
        recvCopy = recvCopy_vector
    else:
        sendCopy = sendCopy_scalar
        recvCopy = recvCopy_scalar
    if ny % 2 == 0 and nProc_y > 1:
        # Even top
        nx_send = (nx + nProc_x) % nProc_x
        ny_send = (ny + 1 + nProc_y) % nProc_y
        rank_send = nx_send * nProc_y + ny_send
        if ny == nProc_y - 1 and inner is True:
            sendLims = (0, Nx, Ny - 3, Ny - 2, Nx, Ny)
            recvLims = (0, Nx, Ny - 1, Ny, Nx, Ny)
        else:
            sendLims = (0, Nx, Ny - 2, Ny - 1, Nx, Ny)
            recvLims = (0, Nx, Ny - 1, Ny, Nx, Ny)
        sendCopy(data, data_send_topBottom, *sendLims)
        comm.Send(data_send_topBottom, dest=rank_send,
                  tag=current_rank*rank_send)
        comm.Recv(data_recv_topBottom, source=rank_send,
                  tag=rank_send*current_rank)
        recvCopy(data, data_recv_topBottom, *recvLims)
        # Even bottom
        nx_send = (nx + nProc_x) % nProc_x
        ny_send = (ny - 1 + nProc_y) % nProc_y
        rank_send = nx_send * nProc_y + ny_send
        comm.Recv(data_recv_topBottom, source=rank_send,
                  tag=rank_send*current_rank)
        if ny == 0 and inner is True:
            recvLims = (0, Nx, 0, 1, Nx, Ny)
            sendLims = (0, Nx, 2, 3, Nx, Ny)
        else:
            recvLims = (0, Nx, 0, 1, Nx, Ny)
            sendLims = (0, Nx, 1, 2, Nx, Ny)
        recvCopy(data, data_recv_topBottom, *recvLims)
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
        if ny == nProc_y - 1 and inner is True:
            sendLims = (0, Nx, Ny - 3, Ny - 2, Nx, Ny)
            recvLims = (0, Nx, Ny - 1, Ny, Nx, Ny)
        else:
            sendLims = (0, Nx, Ny - 2, Ny - 1, Nx, Ny)
            recvLims = (0, Nx, Ny - 1, Ny, Nx, Ny)
        sendCopy(data, data_send_topBottom, *sendLims)
        comm.Send(data_send_topBottom, dest=rank_send,
                  tag=current_rank*rank_send)
        comm.Recv(data_recv_topBottom, source=rank_send,
                  tag=rank_send*current_rank)
        recvCopy(data, data_recv_topBottom, *recvLims)
    elif nProc_y == 1:
        # top boundary copy to bottom boundary
        if inner is True and Ny - 3 != 0:
            sendLims = (0, Nx, Ny - 3, Ny - 2, Nx, Ny)
            recvLims = (0, Nx, 0, 1, Nx, Ny)
        else:
            sendLims = (0, Nx, Ny - 2, Ny - 1, Nx, Ny)
            recvLims = (0, Nx, 0, 1, Nx, Ny)
        sendCopy(data, data_send_topBottom, *sendLims)
        data_recv_topBottom = np.copy(data_send_topBottom)
        recvCopy(data, data_recv_topBottom, *recvLims)
        # bottom boundary copy to top boundary
        if inner is True and Ny - 3 != 0:
            sendLims = (0, Nx, 2, 3, Nx, Ny)
            recvLims = (0, Nx, Ny - 1, Ny, Nx, Ny)
        else:
            sendLims = (0, Nx, 1, 2, Nx, Ny)
            recvLims = (0, Nx, Ny - 1, Ny, Nx, Ny)
        sendCopy(data, data_send_topBottom, *sendLims)
        data_recv_topBottom = np.copy(data_send_topBottom)
        recvCopy(data, data_recv_topBottom, *recvLims)
    comm.Barrier()
    if nx % 2 == 0 and nProc_x > 1:
        # Even right
        nx_send = (nx + 1 + nProc_x) % nProc_x
        ny_send = (ny + nProc_y) % nProc_y
        rank_send = nx_send * nProc_y + ny_send
        if nx == nProc_x - 1 and inner is True:
            sendLims = (Nx - 3, Nx - 2, 0, Ny, Nx, Ny)
            recvLims = (Nx - 1, Nx, 0, Ny, Nx, Ny)
        else:
            sendLims = (Nx - 2, Nx - 1, 0, Ny, Nx, Ny)
            recvLims = (Nx - 1, Nx, 0, Ny, Nx, Ny)
        sendCopy(data, data_send_leftRight, *sendLims)
        comm.Send(data_send_leftRight, dest=rank_send,
                  tag=current_rank*rank_send)
        comm.Recv(data_recv_leftRight, source=rank_send,
                  tag=rank_send*current_rank)
        recvCopy(data, data_recv_leftRight, *recvLims)
        # Even left
        nx_send = (nx - 1 + nProc_x) % nProc_x
        ny_send = (ny + nProc_y) % nProc_y
        rank_send = nx_send * nProc_y + ny_send
        if nx == 0 and inner is True:
            recvLims = (0, 1, 0, Ny, Nx, Ny)
            sendLims = (2, 3, 0, Ny, Nx, Ny)
        else:
            recvLims = (0, 1, 0, Ny, Nx, Ny)
            sendLims = (1, 2, 0, Ny, Nx, Ny)
        comm.Recv(data_recv_leftRight, source=rank_send,
                  tag=rank_send*current_rank)
        recvCopy(data, data_recv_leftRight, *recvLims)
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
        if nx == nProc_x - 1 and inner is True:
            sendLims = (Nx - 3, Nx - 2, 0, Ny, Nx, Ny)
            recvLims = (Nx - 1, Nx, 0, Ny, Nx, Ny)
        else:
            sendLims = (Nx - 2, Nx - 1, 0, Ny, Nx, Ny)
            recvLims = (Nx - 1, Nx, 0, Ny, Nx, Ny)
        sendCopy(data, data_send_leftRight, *sendLims)
        comm.Send(data_send_leftRight, dest=rank_send,
                  tag=current_rank*rank_send)
        comm.Recv(data_recv_leftRight, source=rank_send,
                  tag=rank_send*current_rank)
        recvCopy(data, data_recv_leftRight, *recvLims)
    elif nProc_x == 1:
        # right boundary copy to left boundary
        if inner is True and Nx - 3 != 0:
            sendLims = (Nx - 3, Nx - 2, 0, Ny, Nx, Ny)
            recvLims = (0, 1, 0, Ny, Nx, Ny)
        else:
            sendLims = (Nx - 2, Nx - 1, 0, Ny, Nx, Ny)
            recvLims = (0, 1, 0, Ny, Nx, Ny)
        sendCopy(data, data_send_leftRight, *sendLims)
        data_recv_leftRight = np.copy(data_send_leftRight)
        recvCopy(data, data_recv_leftRight, *recvLims)
        # left boundary copy to right boundary
        if inner is True and Nx - 3 != 0:
            sendLims = (2, 3, 0, Ny, Nx, Ny)
            recvLims = (Nx - 1, Nx, 0, Ny, Nx, Ny)
        else:
            sendLims = (1, 2, 0, Ny, Nx, Ny)
            recvLims = (Nx - 1, Nx, 0, Ny, Nx, Ny)
        sendCopy(data, data_send_leftRight, *sendLims)
        data_recv_leftRight = np.copy(data_send_leftRight)
        recvCopy(data, data_recv_leftRight, *recvLims)
    comm.Barrier()


@numba.njit
def sendCopy_vector(data, data_send, Nx_i, Nx_f, Ny_i, Ny_f, Nx, Ny):
    itr = 0
    for i in range(Nx_i, Nx_f):
        for j in range(Ny_i, Ny_f):
            ind = i * Ny + j
            data_send[itr, :] = data[ind, :]
            itr += 1


@numba.njit
def sendCopy_scalar(data, data_send, Nx_i, Nx_f, Ny_i, Ny_f, Nx, Ny):
    itr = 0
    for i in range(Nx_i, Nx_f):
        for j in range(Ny_i, Ny_f):
            ind = i * Ny + j
            data_send[itr] = data[ind]
            itr += 1


@numba.njit
def recvCopy_vector(data, data_recv, Nx_i, Nx_f, Ny_i, Ny_f, Nx, Ny):
    itr = 0
    for i in range(Nx_i, Nx_f):
        for j in range(Ny_i, Ny_f):
            ind = i * Ny + j
            data[ind, :] = data_recv[itr, :]
            itr += 1


@numba.njit
def recvCopy_scalar(data, data_recv, Nx_i, Nx_f, Ny_i, Ny_f, Nx, Ny):
    itr = 0
    for i in range(Nx_i, Nx_f):
        for j in range(Ny_i, Ny_f):
            ind = i * Ny + j
            data[ind] = data_recv[itr]
            itr += 1


def reduceComm(dataArray, tempArray, comm, rank, size, precision):
    dataArray = np.array(dataArray)
    noOfData = int(dataArray.shape[0])
    sumData = np.zeros(noOfData, dtype=precision)
    if size > 1:
        if rank == 0:
            sumData += dataArray
            for i in range(1, size):
                comm.Recv(tempArray, source=i, tag=1*i)
                sumData += tempArray
            return sumData
        else:
            dataArray = np.array(dataArray)
            comm.Send(dataArray, dest=0, tag=1*rank)
            return np.zeros(noOfData)
    else:
        return np.array(dataArray)


def gatherForcesTorque_mpi(options, comm, rank, size, precision):
    if rank == 0:
        # print(options.surfaceNamesGlobal)
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
        for i in range(1, size):
            comm.send(sumF, dest=i, tag=1*i)
            comm.send(sumT, dest=i, tag=2*i)
        return options.surfaceNamesGlobal, sumF, sumT
    else:
        comm.send(options.surfaceNames, dest=0, tag=1*rank)
        comm.send(options.forces, dest=0, tag=2*rank)
        comm.send(options.torque, dest=0, tag=3*rank)
        AllForces = comm.recv(source=0, tag=1*rank)
        AllTorque = comm.recv(source=0, tag=2*rank)
        sumF = np.zeros((len(options.surfaceNames), 2),
                        dtype=precision)
        sumT = np.zeros((len(options.surfaceNames)),
                        dtype=precision)
        # print(options.surfaceNamesGlobal)
        for itr_local, local_name in \
                enumerate(options.surfaceNames):
            for itr, name in enumerate(options.surfaceNamesGlobal):
                if name == local_name:
                    # print('here')
                    sumF[itr_local, 0] = AllForces[itr][0]
                    sumF[itr_local, 1] = AllForces[itr][1]
                    sumT[itr_local] = AllTorque[itr]
        # print('all', AllForces)
        # print('sum', sumF)
        options.forces = np.copy(sumF)
        options.torque = np.copy(sumT)
        return options.surfaceNamesGlobal, sumF, sumT
