import numba
from numba import prange


@numba.njit(parallel=True, nogil=True)
def bounce_back(
    boundary_nodes,
    out_list,
    inv_list,
    solid,
    pop,
    pop_new
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
    shape,
    cx,
    cy,
    weights,
    inv_cs_2,
    boundary_nodes,
    out_list,
    inv_list,
    solid,
    density,
    velocity,
    pop,
    pop_new
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
            i = ind // shape[1]
            j = ind - i * shape[1]
            density_local = density[ind]
            for k in range(out_list.shape[0]):
                out_dir = out_list[k]
                inv_dir = inv_list[k]
                i_nb = i + cx[out_dir]
                j_nb = j + cy[out_dir]
                ind_nb = i_nb * shape[1] + j_nb
                temp = 2 * weights[out_dir] * density_local * inv_cs_2 * (
                    cx[out_dir] * velocity[ind_nb, 0] +
                    cy[out_dir] * velocity[ind_nb, 1]
                )
                pop_new[ind, inv_dir] = pop[ind, out_dir] - temp


@numba.njit(parallel=True, nogil=True)
def fixed_velocity_no_density_based(
    shape,
    cx,
    cy,
    weights,
    inv_cs_2,
    boundary_nodes,
    out_list,
    inv_list,
    solid,
    velocity,
    pop,
    pop_new
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
            i = ind // shape[1]
            j = ind - i * shape[1]
            for k in range(out_list.shape[0]):
                out_dir = out_list[k]
                inv_dir = inv_list[k]
                i_nb = i + cx[out_dir]
                j_nb = j + cy[out_dir]
                ind_nb = i_nb * shape[1] + j_nb
                temp = 2 * weights[out_dir] * inv_cs_2 * (
                    cx[out_dir] * velocity[ind_nb, 0] +
                    cy[out_dir] * velocity[ind_nb, 1]
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
    boundary_nodes,
    out_list,
    inv_list,
    surface_normals,
    solid,
    density,
    velocity,
    pop,
    pop_new
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
            density_local = density[ind]
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
            density_extrapolated = density_local + 0.5 * (
                density_local - density[ind_normal]
            )
            u2 = velocity_extrapolated_x * velocity_extrapolated_x +\
                velocity_extrapolated_y * velocity_extrapolated_y
            for k in range(out_list.shape[0]):
                out_dir = out_list[k]
                inv_dir = inv_list[k]
                cu = cx[out_dir] * velocity_extrapolated_x +\
                    cy[out_dir] * velocity_extrapolated_y
                temp = 2 * weights[out_dir] * density_extrapolated * (
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
    boundary_nodes,
    out_list,
    inv_list,
    surface_normals,
    solid,
    pressure,
    velocity,
    pop,
    pop_new
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
            pressure_local = pressure[ind]
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
            pressure_extrapolated = pressure_local + 0.5 * (
                pressure_local - pressure[ind_normal]
            )
            u2 = velocity_extrapolated_x * velocity_extrapolated_x +\
                velocity_extrapolated_y * velocity_extrapolated_y
            for k in range(out_list.shape[0]):
                out_dir = out_list[k]
                inv_dir = inv_list[k]
                cu = cx[out_dir] * velocity_extrapolated_x +\
                    cy[out_dir] * velocity_extrapolated_y
                temp = 2 * weights[out_dir] * (
                    pressure_extrapolated + 0.5 * inv_cs_4 * cu * cu -
                    0.5 * inv_cs_2 * u2
                )
                pop_new[ind, inv_dir] = - pop[ind, out_dir] + temp
