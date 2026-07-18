from numba import cuda
from numba import float64

from pylabolt.parallel.backend import REDUCE_BLOCK_SIZE


@cuda.jit
def compute_residue_scalar(
    size,
    solid,
    ghost_node,
    field,
    field_old,
    partial_numerator,
    partial_denominator
):
    """
    Kernel to compute residue for scalar fields
    Residues always accumulated in float64
    Args:

    Returns:

    """
    ind = cuda.grid(1)
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_dim = cuda.blockDim.x

    shared_numerator = cuda.shared.array(REDUCE_BLOCK_SIZE, float64)
    shared_denominator = cuda.shared.array(REDUCE_BLOCK_SIZE, float64)

    numerator = 0.
    denominator = 0.

    if ind < size:
        if not solid[ind] and not ghost_node[ind]:
            field_old_local = field_old[ind]
            field_local = field[ind]
            diff_field = field_local - field_old_local
            numerator = diff_field * diff_field
            denominator = field_old_local * field_old_local
            field_old[ind] = field_local

    shared_numerator[thread_idx] = numerator
    shared_denominator[thread_idx] = denominator

    cuda.syncthreads()

    stride = block_dim // 2
    while stride > 0:
        if thread_idx < stride:
            shared_numerator[thread_idx] +=\
                shared_numerator[thread_idx + stride]
            shared_denominator[thread_idx] +=\
                shared_denominator[thread_idx + stride]

        cuda.syncthreads()

        stride = stride // 2

    if thread_idx == 0:
        partial_numerator[block_idx] = shared_numerator[0]
        partial_denominator[block_idx] = shared_denominator[0]


@cuda.jit
def reduce_residue_scalar(
    partial_size,
    partial_numerator,
    partial_denominator
):
    """
    Reduction kernel to recursively compute residues
    Residues always accumulated in float64
    Args:

    Returns:

    """
    ind = cuda.grid(1)
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_dim = cuda.blockDim.x

    shared_numerator = cuda.shared.array(REDUCE_BLOCK_SIZE, float64)
    shared_denominator = cuda.shared.array(REDUCE_BLOCK_SIZE, float64)

    if ind < partial_size:
        shared_numerator[thread_idx] = partial_numerator[ind]
        shared_denominator[thread_idx] = partial_denominator[ind]
    else:
        shared_numerator[thread_idx] = 0.0
        shared_denominator[thread_idx] = 0.0

    cuda.syncthreads()

    stride = block_dim // 2
    while stride > 0:
        if thread_idx < stride:
            shared_numerator[thread_idx] +=\
                shared_numerator[thread_idx + stride]
            shared_denominator[thread_idx] +=\
                shared_denominator[thread_idx + stride]

        cuda.syncthreads()

        stride = stride // 2

    if thread_idx == 0:
        partial_numerator[block_idx] = shared_numerator[0]
        partial_denominator[block_idx] = shared_denominator[0]


@cuda.jit
def compute_residue_vector(
    size,
    solid,
    ghost_node,
    field,
    field_old,
    partial_numerator,
    partial_denominator
):
    """
    Kernel to compute residue for vector fields
    Residues always accumulated in float64
    Args:

    Returns:

    """
    ind = cuda.grid(1)
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_dim = cuda.blockDim.x

    shared_numerator = cuda.shared.array((REDUCE_BLOCK_SIZE, 2), float64)
    shared_denominator = cuda.shared.array((REDUCE_BLOCK_SIZE, 2), float64)

    numerator_x = 0.
    denominator_x = 0.
    numerator_y = 0.
    denominator_y = 0.

    if ind < size:
        if not solid[ind] and not ghost_node[ind]:
            field_old_local_x = field_old[ind, 0]
            field_old_local_y = field_old[ind, 1]
            field_local_x = field[ind, 0]
            field_local_y = field[ind, 1]
            diff_field_x = field_local_x - field_old_local_x
            diff_field_y = field_local_y - field_old_local_y
            numerator_x = diff_field_x * diff_field_x
            numerator_y = diff_field_y * diff_field_y
            denominator_x = field_old_local_x * field_old_local_x
            denominator_y = field_old_local_y * field_old_local_y
            field_old[ind, 0] = field_local_x
            field_old[ind, 1] = field_local_y

    shared_numerator[thread_idx, 0] = numerator_x
    shared_denominator[thread_idx, 0] = denominator_x
    shared_numerator[thread_idx, 1] = numerator_y
    shared_denominator[thread_idx, 1] = denominator_y

    cuda.syncthreads()

    stride = block_dim // 2
    while stride > 0:
        if thread_idx < stride:
            shared_numerator[thread_idx, 0] +=\
                shared_numerator[thread_idx + stride, 0]
            shared_denominator[thread_idx, 0] +=\
                shared_denominator[thread_idx + stride, 0]
            shared_numerator[thread_idx, 1] +=\
                shared_numerator[thread_idx + stride, 1]
            shared_denominator[thread_idx, 1] +=\
                shared_denominator[thread_idx + stride, 1]

        cuda.syncthreads()

        stride = stride // 2

    if thread_idx == 0:
        partial_numerator[block_idx, 0] = shared_numerator[0, 0]
        partial_denominator[block_idx, 0] = shared_denominator[0, 0]
        partial_numerator[block_idx, 1] = shared_numerator[0, 1]
        partial_denominator[block_idx, 1] = shared_denominator[0, 1]


@cuda.jit
def reduce_residue_vector(
    partial_size,
    partial_numerator,
    partial_denominator
):
    """
    Reduction kernel to recursively compute residues for vectors
    Residues always accumulated in float64
    Args:

    Returns:

    """
    ind = cuda.grid(1)
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_dim = cuda.blockDim.x

    shared_numerator = cuda.shared.array((REDUCE_BLOCK_SIZE, 2), float64)
    shared_denominator = cuda.shared.array((REDUCE_BLOCK_SIZE, 2), float64)

    if ind < partial_size:
        shared_numerator[thread_idx, 0] = partial_numerator[ind, 0]
        shared_denominator[thread_idx, 0] = partial_denominator[ind, 0]
        shared_numerator[thread_idx, 1] = partial_numerator[ind, 1]
        shared_denominator[thread_idx, 1] = partial_denominator[ind, 1]
    else:
        shared_numerator[thread_idx, 0] = 0.0
        shared_denominator[thread_idx, 0] = 0.0
        shared_numerator[thread_idx, 1] = 0.0
        shared_denominator[thread_idx, 1] = 0.0

    cuda.syncthreads()

    stride = block_dim // 2
    while stride > 0:
        if thread_idx < stride:
            shared_numerator[thread_idx, 0] +=\
                shared_numerator[thread_idx + stride, 0]
            shared_denominator[thread_idx, 0] +=\
                shared_denominator[thread_idx + stride, 0]
            shared_numerator[thread_idx, 1] +=\
                shared_numerator[thread_idx + stride, 1]
            shared_denominator[thread_idx, 1] +=\
                shared_denominator[thread_idx + stride, 1]

        cuda.syncthreads()

        stride = stride // 2

    if thread_idx == 0:
        partial_numerator[block_idx, 0] = shared_numerator[0, 0]
        partial_denominator[block_idx, 0] = shared_denominator[0, 0]
        partial_numerator[block_idx, 1] = shared_numerator[0, 1]
        partial_denominator[block_idx, 1] = shared_denominator[0, 1]
