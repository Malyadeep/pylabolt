import numba
from numba import prange


@numba.njit(parallel=True, nogil=True)
def bounce_back(
    solid,
    pop,
    pop_new,
    boundary_nodes,
    out_list,
    inv_list
):
    """
    Bounce back boundary condition kernel
    Args:

    Returns:

    """
    for itr in prange(boundary_nodes.shape[0]):
        ind = boundary_nodes[itr]
        if not solid[ind]:
            for k in range(out_list.shape[0]):
                pop_new[ind, inv_list[k]] = pop[ind, out_list[k]]


@numba.njit(parallel=True, nogil=True)
def fixed_velocity_density_based(
    cx,
    cy,
    weights,
    inv_cs_2,
    solid,
    density,
    pop,
    pop_new,
    boundary_nodes,
    out_list,
    inv_list,
    boundary_velocity
):
    """
    Fixed velocity boundary condition kernel
    for density based equilibrium distributions
    Args:

    Returns:

    """
    for itr in prange(boundary_nodes.shape[0]):
        ind = boundary_nodes[itr]
        if not solid[ind]:
            density_local = density[ind]
            for k in range(out_list.shape[0]):
                out_dir = out_list[k]
                inv_dir = inv_list[k]
                temp = 2 * weights[out_dir] * density_local * inv_cs_2 * (
                    cx[out_dir] * boundary_velocity[0] +
                    cy[out_dir] * boundary_velocity[1]
                )
                pop_new[ind, inv_dir] = pop[ind, out_dir] - temp


@numba.njit(parallel=True, nogil=True)
def fixed_velocity_no_density_based(
    cx,
    cy,
    weights,
    inv_cs_2,
    solid,
    pop,
    pop_new,
    boundary_nodes,
    out_list,
    inv_list,
    boundary_velocity
):
    """
    Fixed velocity boundary condition kernel
    for velocity based equilibrium distributions
    Args:

    Returns:

    """
    for itr in prange(boundary_nodes.shape[0]):
        ind = boundary_nodes[itr]
        if not solid[ind]:
            for k in range(out_list.shape[0]):
                out_dir = out_list[k]
                inv_dir = inv_list[k]
                temp = 2 * weights[out_dir] * inv_cs_2 * (
                    cx[out_dir] * boundary_velocity[0] +
                    cy[out_dir] * boundary_velocity[1]
                )
                pop_new[ind, inv_dir] = pop[ind, out_dir] - temp


@numba.njit(parallel=True, nogil=True)
def fixed_pressure_density_based(
    shape,
    cx,
    cy,
    weights,
    inv_cs_2,
    inv_cs_4,
    solid,
    velocity,
    pop,
    pop_new,
    boundary_nodes,
    out_list,
    inv_list,
    boundary_density,
    surface_normals
):
    """
    Fixed pressure boundary condition kernel
    for density based equilibrium distributions
    Args:

    Returns:

    """
    for itr in prange(boundary_nodes.shape[0]):
        ind = boundary_nodes[itr]
        if not solid[ind]:
            i = ind // shape[1]
            j = ind - i * shape[1]
            velocity_local_x = velocity[ind, 0]
            velocity_local_y = velocity[ind, 1]
            i_normal = i + surface_normals[0]
            j_normal = j + surface_normals[1]
            ind_normal = i_normal * shape[1] + j_normal
            velocity_extrapolated_x = velocity_local_x + 0.5 * (
                velocity_local_x - velocity[ind_normal, 0]
            )
            velocity_extrapolated_y = velocity_local_y + 0.5 * (
                velocity_local_y - velocity[ind_normal, 1]
            )
            u2 = velocity_extrapolated_x * velocity_extrapolated_x +\
                velocity_extrapolated_y * velocity_extrapolated_y
            for k in range(out_list.shape[0]):
                out_dir = out_list[k]
                inv_dir = inv_list[k]
                cu = cx[out_dir] * velocity_extrapolated_x +\
                    cy[out_dir] * velocity_extrapolated_y
                temp = 2 * weights[out_dir] * boundary_density * (
                    1 + 0.5 * inv_cs_4 * cu * cu - 0.5 * inv_cs_2 * u2
                )
                pop_new[ind, inv_dir] = - pop[ind, out_dir] + temp


@numba.njit(parallel=True, nogil=True)
def fixed_pressure_no_density_based(
    shape,
    cx,
    cy,
    weights,
    inv_cs_2,
    inv_cs_4,
    solid,
    velocity,
    pop,
    pop_new,
    boundary_nodes,
    out_list,
    inv_list,
    boundary_pressure,
    surface_normals
):
    """
    Fixed pressure boundary condition kernel
    for velocity based equilibrium distributions
    Args:

    Returns:

    """
    for itr in prange(boundary_nodes.shape[0]):
        ind = boundary_nodes[itr]
        if not solid[ind]:
            i = ind // shape[1]
            j = ind - i * shape[1]
            velocity_local_x = velocity[ind, 0]
            velocity_local_y = velocity[ind, 1]
            i_normal = i + surface_normals[0]
            j_normal = j + surface_normals[1]
            ind_normal = i_normal * shape[1] + j_normal
            velocity_extrapolated_x = velocity_local_x + 0.5 * (
                velocity_local_x - velocity[ind_normal, 0]
            )
            velocity_extrapolated_y = velocity_local_y + 0.5 * (
                velocity_local_y - velocity[ind_normal, 1]
            )
            u2 = velocity_extrapolated_x * velocity_extrapolated_x +\
                velocity_extrapolated_y * velocity_extrapolated_y
            for k in range(out_list.shape[0]):
                out_dir = out_list[k]
                inv_dir = inv_list[k]
                cu = cx[out_dir] * velocity_extrapolated_x +\
                    cy[out_dir] * velocity_extrapolated_y
                temp = 2 * weights[out_dir] * (
                    boundary_pressure + 0.5 * inv_cs_4 * cu * cu -
                    0.5 * inv_cs_2 * u2
                )
                pop_new[ind, inv_dir] = - pop[ind, out_dir] + temp
