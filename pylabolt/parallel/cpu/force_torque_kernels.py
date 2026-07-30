import numpy as np
import numba
from numba import prange

from pylabolt.parallel.cpu.MPI_kernels import local_to_global

# --------------------------------------------------------------------------#
""" Kernels to compute force on obstacles """


@numba.njit(parallel=True, nogil=True)
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
    current_solid_id
):
    """
    Compute force acting on solid for density based
    distributions, typical of single-phase flows
    Args:

    Returns:

    """
    force_sum_x = 0.
    force_sum_y = 0.
    torque_sum = 0.
    Nx, Ny = grid_global_shape
    for ind in prange(size):
        if (not ghost_node[ind] and fluid_boundary[ind] and
                solid_id[ind] == current_solid_id):
            x = ind // shape[1]
            y = ind - x * shape[1]
            x_global, y_global = local_to_global(
                x - 1, y - 1, offset
            )
            rx = x_global - ref_point[0]
            ry = y_global - ref_point[1]
            rx_min = rx
            ry_min = ry
            if x_periodic:
                if np.abs(rx + Nx) < np.abs(rx_min):
                    rx_min = rx + Nx
                if np.abs(rx - Nx) < np.abs(rx_min):
                    rx_min = rx - Nx
            if y_periodic:
                if np.abs(ry + Ny) < np.abs(ry_min):
                    ry_min = ry + Ny
                if np.abs(ry - Ny) < np.abs(ry_min):
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
    return np.array([
        force_sum_x,
        force_sum_y,
        torque_sum
    ])


# --------------------------------------------------------------------------#
""" Kernels to compute force on boundary """


@numba.njit(parallel=True, nogil=True)
def compute_boundary_force_single_phase(
    cx,
    cy,
    solid,
    pop,
    pop_new,
    boundary_nodes,
    out_list,
    inv_list
):
    """
    Compute force acting on domain boundary for density based
    distributions, typical of single-phase flows
    Args:

    Returns:

    """
    force_sum_x = 0.
    force_sum_y = 0.
    for itr in prange(boundary_nodes.shape[0]):
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
    return np.array([
        force_sum_x,
        force_sum_y
    ])
