import numpy as np


def proc_boundary(Nx_local, Ny_local, c, invList, f_new, f, nx, ny,
                  args, comm):
    current_rank = nx * args.ny + ny
    for i in range(Nx_local + 2):
        if ny % 2 == 0:
            if ny < args.ny - 1:
                for j in range(Ny_local + 1, Ny_local + 2):
                    for k in range(1, 9):
                        if j + int(c[k, 1]) > Ny_local + 1:
                            rank_send = nx * args.ny + ny + 1
                            comm.send(f_new[i, j, k], dest=rank_send,
                                      tag=1*i*rank_send)
                            f_new[i, j, invList[k]] = \
                                comm.recv(source=rank_send,
                                          tag=3*i*current_rank
                                          )
            if ny > 0:
                for j in range(0, 1):
                    for k in range(1, 9):
                        if j - int(c[k, 1]) < 0:
                            rank_send = nx * args.ny + ny - 1
                            f_new[i, j, k] = \
                                comm.recv(source=rank_send,
                                          tag=4*i*current_rank
                                          )
                            comm.send(f_new[i, j, invList[k]], dest=rank_send,
                                      tag=2*i*rank_send)
        elif ny % 2 != 0:
            if ny > 0:
                for j in range(0, 1):
                    for k in range(1, 9):
                        if j - int(c[k, 1]) < 0:
                            rank_send = nx * args.ny + ny - 1
                            f_new[i, j, k] = \
                                comm.recv(source=rank_send,
                                          tag=1*i*current_rank
                                          )
                            comm.send(f_new[i, j, invList[k]], dest=rank_send,
                                      tag=3*i*rank_send)
            if ny < args.ny - 1:
                for j in range(Ny_local + 1, Ny_local + 2):
                    for k in range(1, 9):
                        if j + int(c[k, 1]) > Ny_local + 1:
                            rank_send = nx * args.ny + ny + 1
                            comm.send(f_new[i, j, k], dest=rank_send,
                                      tag=4*i*rank_send)
                            f_new[i, j, invList[k]] = \
                                comm.recv(source=rank_send,
                                          tag=2*i*current_rank
                                          )
    comm.Barrier()
    for j in range(Ny_local + 2):
        if nx % 2 == 0:
            if nx < args.nx - 1:
                for i in range(Nx_local + 1, Nx_local + 2):
                    for k in range(1, 9):
                        if i + int(c[k, 0]) > Nx_local + 1:
                            rank_send = (nx + 1) * args.ny + ny
                            comm.send(f_new[i, j, k], dest=rank_send,
                                      tag=1*j*rank_send)
                            f_new[i, j, invList[k]] = \
                                comm.recv(source=rank_send,
                                          tag=3*j*current_rank
                                          )
            if nx > 0:
                for i in range(0, 1):
                    for k in range(1, 9):
                        if i - int(c[k, 0]) < 0:
                            rank_send = (nx - 1) * args.ny + ny
                            f_new[i, j, k] = \
                                comm.recv(source=rank_send,
                                          tag=4*j*current_rank
                                          )
                            comm.send(f_new[i, j, invList[k]], dest=rank_send,
                                      tag=2*j*rank_send)
        elif nx % 2 != 0:
            if nx > 0:
                for i in range(0, 1):
                    for k in range(1, 9):
                        if i - int(c[k, 0]) < 0:
                            rank_send = (nx - 1) * args.ny + ny
                            f_new[i, j, k] = \
                                comm.recv(source=rank_send,
                                          tag=1*j*current_rank
                                          )
                            comm.send(f_new[i, j, invList[k]], dest=rank_send,
                                      tag=3*j*rank_send)
            if nx < args.nx - 1:
                for i in range(Nx_local + 1, Nx_local + 2):
                    for k in range(1, 9):
                        if i + int(c[k, 0]) > Nx_local + 1:
                            rank_send = (nx + 1) * args.ny + ny
                            comm.send(f_new[i, j, k], dest=rank_send,
                                      tag=4*j*rank_send)
                            f_new[i, j, invList[k]] = \
                                comm.recv(source=rank_send,
                                          tag=2*j*current_rank
                                          )
    comm.Barrier()
    for i in range(1, Nx_local + 1):
        for j in range(Ny_local, Ny_local + 1):
            for k in range(1, 9):
                if j - int(c[k, 1]) > Ny_local and i - int(c[k, 0]) > Nx_local:
                    f_new[i, j, k] = f_new[i + 1, j + 1, k]
                elif j - int(c[k, 1]) > Ny_local and i - int(c[k, 0]) < 1:
                    f_new[i, j, k] = f_new[i - 1, j + 1, k]
                elif j - int(c[k, 1]) > Ny_local:
                    f_new[i, j, k] = f_new[i, j + 1, k]
        for j in range(1, 2):
            for k in range(1, 9):
                if j - int(c[k, 1]) < 1 and i - int(c[k, 0]) < 1:
                    f_new[i, j, k] = f_new[i - 1, j - 1, k]
                elif j - int(c[k, 1]) < 1 and i - int(c[k, 0]) > Nx_local:
                    f_new[i, j, k] = f_new[i + 1, j - 1, k]
                elif j - int(c[k, 1]) < 1:
                    f_new[i, j, k] = f_new[i, j - 1, k]
    for j in range(1, Ny_local + 1):
        for i in range(1, 2):
            for k in range(1, 9):
                if i - int(c[k, 0]) < 1 and j - int(c[k, 1]) < 1:
                    f_new[i, j, k] = f_new[i - 1, j - 1, k]
                elif i - int(c[k, 0]) < 1 and j - int(c[k, 1]) > Ny_local:
                    f_new[i, j, k] = f_new[i - 1, j + 1, k]
                elif i - int(c[k, 0]) < 1:
                    f_new[i, j, k] = f_new[i - 1, j, k]
        for i in range(Nx_local, Nx_local + 1):
            for k in range(1, 9):
                if i - int(c[k, 0]) > Nx_local and j - int(c[k, 1]) < 1:
                    f_new[i, j, k] = f_new[i + 1, j - 1, k]
                elif (i - int(c[k, 0]) > Nx_local and
                      j - int(c[k, 1]) > Ny_local):
                    f_new[i, j, k] = f_new[i + 1, j + 1, k]
                elif i - int(c[k, 0]) > Nx_local:
                    f_new[i, j, k] = f_new[i + 1, j, k]


def computeResiduals(u_sq_err, u_err, v_sq_err, v_err,
                     rho_sq_err, rho_err, comm, rank, size):
    if rank == 0:
        sum_u, sum_v = u_err, v_err
        sum_rho = rho_err
        sum_u_sq, sum_v_sq = u_sq_err, v_sq_err
        sum_rho_sq = rho_sq_err
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
        comm.send(u_err, dest=0, tag=1*rank)
        comm.send(v_err, dest=0, tag=2*rank)
        comm.send(rho_err, dest=0, tag=3*rank)
        comm.send(u_sq_err, dest=0, tag=4*rank)
        comm.send(v_sq_err, dest=0, tag=5*rank)
        comm.send(rho_sq_err, dest=0, tag=6*rank)
        return 0, 0, 0


def gather(u, rho, rank, comm, nProc_x, nProc_y, Nx, Ny, Nx_local, Ny_local):
    if rank == 0:
        u_all = np.zeros((Nx, Ny, 2), dtype=np.float64)
        rho_all = np.ones((Nx, Ny), dtype=np.float64)
        for i in range(nProc_x):
            for j in range(nProc_y):
                current_rank = int(i * nProc_y + j)
                if current_rank == 0:
                    u_all[:Nx_local, :Ny_local, :] = u
                    rho_all[:Nx_local, :Ny_local] = rho
                else:
                    u_temp = comm.recv(source=current_rank,
                                       tag=1*current_rank)
                    rho_temp = comm.recv(source=current_rank,
                                         tag=2*current_rank)
                    u_all[i * Nx_local:(i + 1) * Nx_local, j * Ny_local:(j + 1)
                          * Ny_local, :] = u_temp
                    rho_all[i * Nx_local:(i + 1) * Nx_local, j * Ny_local:(j +
                            1) * Ny_local] = rho_temp
        return u_all, rho_all
    else:
        comm.send(u, dest=0, tag=1*rank)
        comm.send(rho, dest=0, tag=2*rank)
        return 0, 0
