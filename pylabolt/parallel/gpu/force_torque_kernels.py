import numpy as np
from numba import cuda
from numba import float64

from pylabolt.parallel.backend import REDUCE_BLOCK_SIZE
from pylabolt.parallel.gpu.MPI_kernels import local_to_global


# --------------------------------------------------------------------------#
""" Kernels to compute force on obstacles """


@cuda.jit
def compute_force_torque_single_phase(
    size,
    shape,
    offset,
    grid_global_shape,
    cx,
    cy,
    inv_list,
    no_of_directions,
    x_periodic,
    y_periodic,
    solid,
    solid_id,
    fluid_boundary,
    ghost_node,
    pop,
    pop_new,
    ref_point,
    current_solid_id,
    partial_force_torque,
    itr
):
    """
    Compute force acting on solid for density based
    distributions, typical of single-phase flows
    Args:

    Returns:

    """
    ind = cuda.grid(1)
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_dim = cuda.blockDim.x

    shared_force_torque = cuda.shared.array((REDUCE_BLOCK_SIZE, 3), float64)

    force_sum_x = 0.
    force_sum_y = 0.
    torque_sum = 0.
    Nx, Ny = grid_global_shape

    if ind < size:
        if (not ghost_node[ind] and fluid_boundary[ind] and
                solid_id[ind] == current_solid_id[itr, 0]):
            x = ind // shape[1]
            y = ind - x * shape[1]
            x_global, y_global = local_to_global(
                x - 1, y - 1, offset
            )
            rx = x_global - ref_point[itr, 0]
            ry = y_global - ref_point[itr, 1]
            rx_min = rx
            ry_min = ry
            if x_periodic:
                if abs(rx + Nx) < abs(rx_min):
                    rx_min = rx + Nx
                if abs(rx - Nx) < abs(rx_min):
                    rx_min = rx - Nx
            if y_periodic:
                if abs(ry + Ny) < abs(ry_min):
                    ry_min = ry + Ny
                if abs(ry - Ny) < abs(ry_min):
                    ry_min = ry - Ny
            pop_local = pop[ind]
            pop_new_local = pop_new[ind]
            for k in range(no_of_directions):
                x_nb = x + cx[k]
                y_nb = y + cy[k]
                ind_nb = x_nb * shape[1] + y_nb
                if solid[ind_nb]:
                    k_inv = inv_list[k]
                    value_x = (
                        pop_local[k] * cx[k] -
                        pop_new_local[k_inv] * cx[k_inv]
                    )
                    value_y = (
                        pop_local[k] * cy[k] -
                        pop_new_local[k_inv] * cy[k_inv]
                    )
                    force_sum_x += value_x
                    force_sum_y += value_y
                    torque_sum += (
                        rx_min * value_y -
                        ry_min * value_x
                    )

    shared_force_torque[thread_idx, 0] = force_sum_x
    shared_force_torque[thread_idx, 1] = force_sum_y
    shared_force_torque[thread_idx, 2] = torque_sum

    cuda.syncthreads()

    stride = block_dim // 2
    while stride > 0:
        if thread_idx < stride:
            for component in range(3):
                shared_force_torque[thread_idx, component] +=\
                    shared_force_torque[thread_idx + stride, component]

        cuda.syncthreads()

        stride = stride // 2

    if thread_idx == 0:
        for component in range(3):
            partial_force_torque[block_idx, component] =\
                shared_force_torque[0, component]


@cuda.jit
def reduce_force_torque(
    partial_size,
    partial_force_torque,
    local_force,
    local_torque,
    itr
):
    # TODO: On GPU decide should we merge force, torque buffers?
    # For multi GPU with MPI, clubbing together should
    # save communication time
    """
    Reduction kernel to recursively compute force on obstacle
    Residues always accumulated in float64
    Args:

    Returns:

    """
    ind = cuda.grid(1)
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_dim = cuda.blockDim.x

    shared_force_torque = cuda.shared.array((REDUCE_BLOCK_SIZE, 3), float64)

    if ind < partial_size:
        for component in range(3):
            shared_force_torque[thread_idx, component] =\
                partial_force_torque[ind, component]
    else:
        for component in range(3):
            shared_force_torque[thread_idx, component] = 0.

    cuda.syncthreads()

    stride = block_dim // 2
    while stride > 0:
        if thread_idx < stride:
            for component in range(3):
                shared_force_torque[thread_idx, component] +=\
                    shared_force_torque[thread_idx + stride, component]

        cuda.syncthreads()

        stride = stride // 2

    if thread_idx == 0:
        for component in range(3):
            partial_force_torque[block_idx, component] =\
                shared_force_torque[0, component]
        if block_idx == 0:
            local_force[itr, 0] = shared_force_torque[0, 0]
            local_force[itr, 1] = shared_force_torque[0, 1]
            local_torque[itr, 0] = shared_force_torque[0, 2]


# --------------------------------------------------------------------------#
""" Kernels to compute force on boundary """


@cuda.jit
def compute_boundary_force_single_phase(
    cx,
    cy,
    solid,
    pop,
    pop_new,
    boundary_nodes,
    out_list,
    inv_list,
    partial_force
):
    """
    Compute force acting on domain boundary for density based
    distributions, typical of single-phase flows
    Args:

    Returns:

    """
    itr = cuda.grid(1)
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_dim = cuda.blockDim.x

    shared_force = cuda.shared.array((REDUCE_BLOCK_SIZE, 2), float64)

    force_sum_x = 0.
    force_sum_y = 0.

    if itr < boundary_nodes.shape[0]:
        ind = boundary_nodes[itr]
        if not solid[ind]:
            pop_local = pop[ind]
            pop_new_local = pop_new[ind]
            for k in range(out_list.shape[0]):
                k_out = out_list[k]
                k_inv = inv_list[k]
                value_x = (
                    pop_local[k_out] * cx[k_out] -
                    pop_new_local[k_inv] * cx[k_inv]
                )
                value_y = (
                    pop_local[k_out] * cy[k_out] -
                    pop_new_local[k_inv] * cy[k_inv]
                )
                force_sum_x += value_x
                force_sum_y += value_y

    shared_force[thread_idx, 0] = force_sum_x
    shared_force[thread_idx, 1] = force_sum_y

    cuda.syncthreads()

    stride = block_dim // 2
    while stride > 0:
        if thread_idx < stride:
            for component in range(2):
                shared_force[thread_idx, component] +=\
                    shared_force[thread_idx + stride, component]

        cuda.syncthreads()

        stride = stride // 2

    if thread_idx == 0:
        for component in range(2):
            partial_force[block_idx, component] =\
                shared_force[0, component]


@cuda.jit
def reduce_boundary_force(
    partial_size,
    partial_force,
    local_force
):
    """
    Reduction kernel to recursively compute force on boundaries
    Residues always accumulated in float64
    Args:

    Returns:

    """
    ind = cuda.grid(1)
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_dim = cuda.blockDim.x

    shared_force = cuda.shared.array((REDUCE_BLOCK_SIZE, 2), float64)

    if ind < partial_size:
        for component in range(2):
            shared_force[thread_idx, component] =\
                partial_force[ind, component]
    else:
        for component in range(2):
            shared_force[thread_idx, component] = 0.

    cuda.syncthreads()

    stride = block_dim // 2
    while stride > 0:
        if thread_idx < stride:
            for component in range(2):
                shared_force[thread_idx, component] +=\
                    shared_force[thread_idx + stride, component]

        cuda.syncthreads()

        stride = stride // 2

    if thread_idx == 0:
        for component in range(2):
            partial_force[block_idx, component] =\
                shared_force[0, component]
        if block_idx == 0:
            for component in range(2):
                local_force[component] = shared_force[0, component]
