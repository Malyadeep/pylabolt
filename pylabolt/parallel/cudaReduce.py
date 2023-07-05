from numba import cuda
import numpy as np
import os


@cuda.jit
def cudaReduce_binary(u_num_1, rho_num_1, u_den_1, rho_den_1,
                      u_num_2, rho_num_2, u_den_2, rho_den_2):
    i = cuda.grid(1)
    sharedArray_u_num = cuda.shared.array(shape=(1024, 2), dtype=np.float64)
    sharedArray_rho_num = cuda.shared.array(shape=(1024,), dtype=np.float64)
    sharedArray_u_den = cuda.shared.array(shape=(1024, 2), dtype=np.float64)
    sharedArray_rho_den = cuda.shared.array(shape=(1024,), dtype=np.float64)
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_width = cuda.blockDim.x
    if i < u_num_1.shape[0]:
        sharedArray_u_num[thread_idx, 0] = u_num_1[i, 0]
        sharedArray_u_num[thread_idx, 1] = u_num_1[i, 1]
        sharedArray_rho_num[thread_idx] = rho_num_1[i]
        sharedArray_u_den[thread_idx, 0] = u_den_1[i, 0]
        sharedArray_u_den[thread_idx, 1] = u_den_1[i, 1]
        sharedArray_rho_den[thread_idx] = rho_den_1[i]
    else:
        sharedArray_u_num[thread_idx, 0] = 0.
        sharedArray_u_num[thread_idx, 1] = 0.
        sharedArray_rho_num[thread_idx] = 0.
        sharedArray_u_den[thread_idx, 0] = 0.
        sharedArray_u_den[thread_idx, 1] = 0.
        sharedArray_rho_den[thread_idx] = 0.
    cuda.syncthreads()
    if block_width >= 512:
        if thread_idx < 256:
            sharedArray_u_num[thread_idx, 0] += \
                sharedArray_u_num[thread_idx + 256, 0]
            sharedArray_u_num[thread_idx, 1] += \
                sharedArray_u_num[thread_idx + 256, 1]
            sharedArray_rho_num[thread_idx] += \
                sharedArray_rho_num[thread_idx + 256]
            sharedArray_u_den[thread_idx, 0] += \
                sharedArray_u_den[thread_idx + 256, 0]
            sharedArray_u_den[thread_idx, 1] += \
                sharedArray_u_den[thread_idx + 256, 1]
            sharedArray_rho_den[thread_idx] += \
                sharedArray_rho_den[thread_idx + 256]
            cuda.syncthreads()
    if block_width >= 256:
        if thread_idx < 128:
            sharedArray_u_num[thread_idx, 0] += \
                sharedArray_u_num[thread_idx + 128, 0]
            sharedArray_u_num[thread_idx, 1] += \
                sharedArray_u_num[thread_idx + 128, 1]
            sharedArray_rho_num[thread_idx] += \
                sharedArray_rho_num[thread_idx + 128]
            sharedArray_u_den[thread_idx, 0] += \
                sharedArray_u_den[thread_idx + 128, 0]
            sharedArray_u_den[thread_idx, 1] += \
                sharedArray_u_den[thread_idx + 128, 1]
            sharedArray_rho_den[thread_idx] += \
                sharedArray_rho_den[thread_idx + 128]
            cuda.syncthreads()
    if block_width >= 128:
        if thread_idx < 64:
            sharedArray_u_num[thread_idx, 0] += \
                sharedArray_u_num[thread_idx + 64, 0]
            sharedArray_u_num[thread_idx, 1] += \
                sharedArray_u_num[thread_idx + 64, 1]
            sharedArray_rho_num[thread_idx] += \
                sharedArray_rho_num[thread_idx + 64]
            sharedArray_u_den[thread_idx, 0] += \
                sharedArray_u_den[thread_idx + 64, 0]
            sharedArray_u_den[thread_idx, 1] += \
                sharedArray_u_den[thread_idx + 64, 1]
            sharedArray_rho_den[thread_idx] += \
                sharedArray_rho_den[thread_idx + 64]
            cuda.syncthreads()
    if block_width >= 64:
        if thread_idx < 32:
            sharedArray_u_num[thread_idx, 0] += \
                sharedArray_u_num[thread_idx + 32, 0]
            sharedArray_u_num[thread_idx, 1] += \
                sharedArray_u_num[thread_idx + 32, 1]
            sharedArray_rho_num[thread_idx] += \
                sharedArray_rho_num[thread_idx + 32]
            sharedArray_u_den[thread_idx, 0] += \
                sharedArray_u_den[thread_idx + 32, 0]
            sharedArray_u_den[thread_idx, 1] += \
                sharedArray_u_den[thread_idx + 32, 1]
            sharedArray_rho_den[thread_idx] += \
                sharedArray_rho_den[thread_idx + 32]
            cuda.syncthreads()
    if block_width >= 32:
        if thread_idx < 16:
            sharedArray_u_num[thread_idx, 0] += \
                sharedArray_u_num[thread_idx + 16, 0]
            sharedArray_u_num[thread_idx, 1] += \
                sharedArray_u_num[thread_idx + 16, 1]
            sharedArray_rho_num[thread_idx] += \
                sharedArray_rho_num[thread_idx + 16]
            sharedArray_u_den[thread_idx, 0] += \
                sharedArray_u_den[thread_idx + 16, 0]
            sharedArray_u_den[thread_idx, 1] += \
                sharedArray_u_den[thread_idx + 16, 1]
            sharedArray_rho_den[thread_idx] += \
                sharedArray_rho_den[thread_idx + 16]
            cuda.syncthreads()
    if block_width >= 16:
        if thread_idx < 8:
            sharedArray_u_num[thread_idx, 0] += \
                sharedArray_u_num[thread_idx + 8, 0]
            sharedArray_u_num[thread_idx, 1] += \
                sharedArray_u_num[thread_idx + 8, 1]
            sharedArray_rho_num[thread_idx] += \
                sharedArray_rho_num[thread_idx + 8]
            sharedArray_u_den[thread_idx, 0] += \
                sharedArray_u_den[thread_idx + 8, 0]
            sharedArray_u_den[thread_idx, 1] += \
                sharedArray_u_den[thread_idx + 8, 1]
            sharedArray_rho_den[thread_idx] += \
                sharedArray_rho_den[thread_idx + 8]
            cuda.syncthreads()
    if block_width >= 8:
        if thread_idx < 4:
            sharedArray_u_num[thread_idx, 0] += \
                sharedArray_u_num[thread_idx + 4, 0]
            sharedArray_u_num[thread_idx, 1] += \
                sharedArray_u_num[thread_idx + 4, 1]
            sharedArray_rho_num[thread_idx] += \
                sharedArray_rho_num[thread_idx + 4]
            sharedArray_u_den[thread_idx, 0] += \
                sharedArray_u_den[thread_idx + 4, 0]
            sharedArray_u_den[thread_idx, 1] += \
                sharedArray_u_den[thread_idx + 4, 1]
            sharedArray_rho_den[thread_idx] += \
                sharedArray_rho_den[thread_idx + 4]
            cuda.syncthreads()
    if block_width >= 4:
        if thread_idx < 2:
            sharedArray_u_num[thread_idx, 0] += \
                sharedArray_u_num[thread_idx + 2, 0]
            sharedArray_u_num[thread_idx, 1] += \
                sharedArray_u_num[thread_idx + 2, 1]
            sharedArray_rho_num[thread_idx] += \
                sharedArray_rho_num[thread_idx + 2]
            sharedArray_u_den[thread_idx, 0] += \
                sharedArray_u_den[thread_idx + 2, 0]
            sharedArray_u_den[thread_idx, 1] += \
                sharedArray_u_den[thread_idx + 2, 1]
            sharedArray_rho_den[thread_idx] += \
                sharedArray_rho_den[thread_idx + 2]
            cuda.syncthreads()
    if block_width >= 2:
        if thread_idx < 1:
            sharedArray_u_num[thread_idx, 0] += \
                sharedArray_u_num[thread_idx + 1, 0]
            sharedArray_u_num[thread_idx, 1] += \
                sharedArray_u_num[thread_idx + 1, 1]
            sharedArray_rho_num[thread_idx] += \
                sharedArray_rho_num[thread_idx + 1]
            sharedArray_u_den[thread_idx, 0] += \
                sharedArray_u_den[thread_idx + 1, 0]
            sharedArray_u_den[thread_idx, 1] += \
                sharedArray_u_den[thread_idx + 1, 1]
            sharedArray_rho_den[thread_idx] += \
                sharedArray_rho_den[thread_idx + 1]
            cuda.syncthreads()
    u_num_2[block_idx, 0] = sharedArray_u_num[0, 0]
    u_num_2[block_idx, 1] = sharedArray_u_num[0, 1]
    rho_num_2[block_idx] = sharedArray_rho_num[0]
    u_den_2[block_idx, 0] = sharedArray_u_den[0, 0]
    u_den_2[block_idx, 1] = sharedArray_u_den[0, 1]
    rho_den_2[block_idx] = sharedArray_rho_den[0]


def cudaSum(blocks, threads, u_sq, u_err_sq, rho_sq, rho_err_sq, blockSize):
    cudaReduce_binary[blocks, threads](u_err_sq, rho_err_sq, u_sq, rho_sq,
                                       u_err_sq, rho_err_sq, u_sq, rho_sq)
    cudaReduce_binary[1, blockSize](u_err_sq[:blocks, :], rho_err_sq[:blocks],
                                    u_sq[:blocks, :], rho_sq[:blocks],
                                    u_err_sq, rho_err_sq, u_sq, rho_sq)
    resU_num = u_err_sq[0, 0]
    resU_den = u_sq[0, 0]
    resV_num = u_err_sq[0, 1]
    resV_den = u_sq[0, 1]
    resRho_num = rho_err_sq[0]
    resRho_den = rho_sq[0]
    return resU_num, resU_den, resV_num, resV_den, resRho_num, resRho_den


def setBlockSize(blocks):
    if blocks >= 512:
        blockSize = 1024
    elif blocks >= 256:
        blockSize = 512
    elif blocks >= 128:
        blockSize = 256
    elif blocks >= 64:
        blockSize = 128
    elif blocks >= 32:
        blockSize = 64
    elif blocks >= 16:
        blockSize = 32
    elif blocks >= 8:
        blockSize = 16
    elif blocks >= 4:
        blockSize = 8
    elif blocks >= 2:
        blockSize = 4
    elif blocks >= 1:
        blockSize = 2
    else:
        print('Error! Invalid Block Size!')
        os._exit(1)
    return blockSize
