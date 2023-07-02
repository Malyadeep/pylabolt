from numba import cuda
import numpy as np


@cuda.jit(device=True)
def sumDevice(a, b):
    return a + b


@cuda.jit
def cudaReduce(arr1, arr2):
    i = cuda.grid(1)
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_width = cuda.blockDim.x
    sharedArray = cuda.shared.array(shape=(512,), dtype=np.float64)
    if i < arr1.shape[0]:
        sharedArray[thread_idx] = arr1[i]
        cuda.syncthreads()
        if thread_idx == 0:
            partialSum = arr1[i]
            for k in range(1, block_width):
                if k + block_idx * block_width < arr1.shape[0]:
                    partialSum = sumDevice(partialSum, sharedArray[k])
            arr2[block_idx] = partialSum


def cudaSum(blocks, threads, arr_device):
    cudaReduce[blocks, threads](arr_device, arr_device)
    cudaReduce[1, blocks](arr_device[:blocks], arr_device)
    cudaSum = arr_device[0]
    return cudaSum
