from numba import cuda
from numba import float64
import os

arrayShape = 0


@cuda.jit
def cudaReduce_binary(arr_1, arr_2):
    i = cuda.grid(1)
    sharedArray = cuda.shared.array(shape=(1024, arrayShape),
                                    dtype=float64)
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_width = cuda.blockDim.x
    if i < arr_1.shape[0]:
        for k in range(arr_1.shape[1]):
            sharedArray[thread_idx, k] = arr_1[i, k]
    else:
        for k in range(arr_1.shape[1]):
            sharedArray[thread_idx, k] = 0.
    cuda.syncthreads()
    if block_width >= 512:
        if thread_idx < 256:
            for k in range(arr_1.shape[1]):
                sharedArray[thread_idx, k] += \
                    sharedArray[thread_idx + 256, k]
            cuda.syncthreads()
    if block_width >= 256:
        if thread_idx < 128:
            for k in range(arr_1.shape[1]):
                sharedArray[thread_idx, k] += \
                    sharedArray[thread_idx + 128, k]
            cuda.syncthreads()
    if block_width >= 128:
        if thread_idx < 64:
            for k in range(arr_1.shape[1]):
                sharedArray[thread_idx, k] += \
                    sharedArray[thread_idx + 64, k]
            cuda.syncthreads()
    if block_width >= 64:
        if thread_idx < 32:
            for k in range(arr_1.shape[1]):
                sharedArray[thread_idx, k] += \
                    sharedArray[thread_idx + 32, k]
            cuda.syncthreads()
    if block_width >= 32:
        if thread_idx < 16:
            for k in range(arr_1.shape[1]):
                sharedArray[thread_idx, k] += \
                    sharedArray[thread_idx + 16, k]
            cuda.syncthreads()
    if block_width >= 16:
        if thread_idx < 8:
            for k in range(arr_1.shape[1]):
                sharedArray[thread_idx, k] += \
                    sharedArray[thread_idx + 8, k]
            cuda.syncthreads()
    if block_width >= 8:
        if thread_idx < 4:
            for k in range(arr_1.shape[1]):
                sharedArray[thread_idx, k] += \
                    sharedArray[thread_idx + 4, k]
            cuda.syncthreads()
    if block_width >= 4:
        if thread_idx < 2:
            for k in range(arr_1.shape[1]):
                sharedArray[thread_idx, k] += \
                    sharedArray[thread_idx + 2, k]
            cuda.syncthreads()
    if block_width >= 2:
        if thread_idx < 1:
            for k in range(arr_1.shape[1]):
                sharedArray[thread_idx, k] += \
                    sharedArray[thread_idx + 1, k]
            cuda.syncthreads()
    for k in range(arr_1.shape[1]):
        arr_2[block_idx, k] = sharedArray[0, k]


def cudaSum(blocks, threads, blockSize, arr_device):
    cudaReduce_binary[blocks, threads](arr_device, arr_device)
    cudaReduce_binary[1, blockSize](arr_device[:blocks, :], arr_device)


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
