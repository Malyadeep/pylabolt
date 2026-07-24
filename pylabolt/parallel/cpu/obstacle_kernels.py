import numpy as np
import numba
from numba import prange

from pylabolt.parallel.domain import local_to_global

# --------------------------------------------------------------------------#
""" Kernels for obstacle type circle """


@numba.njit(nogil=True)
def is_circle(
    x_global,
    y_global,
    grid_global_shape,
    x_periodic,
    y_periodic,
    center,
    radius
):
    Nx, Ny = grid_global_shape
    rx = x_global - center[0]
    ry = y_global - center[1]
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
    dist_sq_from_center = rx_min * rx_min + ry_min * ry_min
    inside_solid = False
    if dist_sq_from_center <= radius * radius:
        inside_solid = True
    return inside_solid, rx_min, ry_min


@numba.njit(parallel=True, nogil=True)
def construct_circle(
    size,
    shape,
    offset,
    grid_global_shape,
    x_periodic,
    y_periodic,
    solid,
    solid_id,
    ghost_node,
    density,
    velocity,
    linear_velocity,
    angular_velocity,
    solid_density,
    center,
    radius,
    current_solid_id
):
    """
    Sets solid node values for obstacle type circle
    Args:

    Returns:

    """
    for ind in prange(size):
        if not ghost_node[ind]:
            x = ind // shape[1]
            y = ind - x * shape[1]
            x_global, y_global = local_to_global(
                x - 1, y - 1, offset
            )
            inside_solid, rx, ry = is_circle(
                x_global,
                y_global,
                grid_global_shape,
                x_periodic,
                y_periodic,
                center,
                radius
            )
            if inside_solid:
                solid[ind] = True
                solid_id[ind] = current_solid_id
                velocity[ind, 0] = linear_velocity[0] -\
                    angular_velocity * ry
                velocity[ind, 1] = linear_velocity[1] +\
                    angular_velocity * rx
                density[ind] = solid_density


@numba.njit(parallel=True, nogil=True)
def compute_normals_circle(
    size,
    shape,
    offset,
    solid_boundary,
    fluid_boundary,
    solid_id,
    surface_normals,
    center,
    current_solid_id
):
    """
    Compute surface normals for obstacle type ellipse
    Args:

    Returns:

    """
    # TODO: periodic wrapping of normals when solid crosses boundary
    for ind in prange(size):
        if (solid_id[ind] == current_solid_id and
                (solid_boundary[ind] or fluid_boundary[ind])):
            x = ind // shape[1]
            y = ind - x * shape[1]
            x_global, y_global = local_to_global(
                x - 1, y - 1, offset
            )
            rx = x_global - center[0]
            ry = y_global - center[1]
            mag = np.sqrt(rx * rx + ry * ry)
            surface_normals[ind, 0] = rx / mag
            surface_normals[ind, 1] = ry / mag


# --------------------------------------------------------------------------#
""" Kernels for obstacle type ellipse """


@numba.njit(nogil=True)
def is_ellipse(
    x_global,
    y_global,
    grid_global_shape,
    x_periodic,
    y_periodic,
    center,
    semi_major_axis,
    semi_minor_axis,
    cos_alpha,
    sin_alpha
):
    Nx, Ny = grid_global_shape
    rx = x_global - center[0]
    ry = y_global - center[1]
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
    x_proj = rx_min * cos_alpha + ry_min * sin_alpha
    y_proj = - rx_min * sin_alpha + ry_min * cos_alpha
    scaled_dist =\
        (x_proj * x_proj) / (semi_major_axis * semi_major_axis) +\
        (y_proj * y_proj) / (semi_minor_axis * semi_minor_axis)
    inside_solid = False
    if scaled_dist <= 1:
        inside_solid = True
    return inside_solid, rx_min, ry_min


@numba.njit(parallel=True, nogil=True)
def construct_ellipse(
    size,
    shape,
    offset,
    grid_global_shape,
    x_periodic,
    y_periodic,
    solid,
    solid_id,
    ghost_node,
    density,
    velocity,
    linear_velocity,
    angular_velocity,
    solid_density,
    center,
    semi_major_axis,
    semi_minor_axis,
    inclination_angle,
    current_solid_id
):
    cos_alpha = np.cos(inclination_angle)
    sin_alpha = np.sin(inclination_angle)
    for ind in prange(size):
        if not ghost_node[ind]:
            x = ind // shape[1]
            y = ind - x * shape[1]
            x_global, y_global = local_to_global(
                x - 1, y - 1, offset
            )
            inside_solid, rx, ry = is_ellipse(
                x_global,
                y_global,
                grid_global_shape,
                x_periodic,
                y_periodic,
                center,
                semi_major_axis,
                semi_minor_axis,
                cos_alpha,
                sin_alpha
            )
            if inside_solid:
                solid[ind] = True
                solid_id[ind] = current_solid_id
                velocity[ind, 0] = linear_velocity[0] -\
                    angular_velocity * ry
                velocity[ind, 1] = linear_velocity[1] +\
                    angular_velocity * rx
                density[ind] = solid_density


@numba.njit(parallel=True, nogil=True)
def compute_normals_ellipse(
    size,
    shape,
    offset,
    solid_boundary,
    fluid_boundary,
    solid_id,
    surface_normals,
    center,
    semi_major_axis,
    semi_minor_axis,
    inclination_angle,
    current_solid_id
):
    """
    Compute surface normals for obstacle type ellipse
    Args:

    Returns:

    """
    # TODO: periodic wrapping of normals when solid crosses boundary
    cos_alpha = np.cos(inclination_angle)
    sin_alpha = np.sin(inclination_angle)
    for ind in prange(size):
        if (solid_id[ind] == current_solid_id and
                (solid_boundary[ind] or fluid_boundary[ind])):
            x = ind // shape[1]
            y = ind - x * shape[1]
            x_global, y_global = local_to_global(
                x - 1, y - 1, offset
            )
            rx = x_global - center[0]
            ry = y_global - center[1]
            x_proj = rx * cos_alpha + ry * sin_alpha
            y_proj = - rx * sin_alpha + ry * cos_alpha
            gx = x_proj / (semi_major_axis * semi_major_axis)
            gy = y_proj / (semi_minor_axis * semi_minor_axis)
            x_g = gx * cos_alpha - gy * sin_alpha
            y_g = gx * sin_alpha + gy * cos_alpha
            mag = np.sqrt(x_g * x_g + y_g * y_g)
            surface_normals[ind, 0] = x_g / mag
            surface_normals[ind, 1] = y_g / mag


# --------------------------------------------------------------------------#
""" Kernels to compute obstacle boundary nodes """


@numba.njit(parallel=True, nogil=True)
def compute_obstacle_boundary(
    size,
    shape,
    cx,
    cy,
    no_of_directions,
    solid,
    solid_id,
    solid_boundary,
    fluid_boundary,
    ghost_node
):
    """
    Compute fluid solid boundary nodes
    Args:

    Returns:

    """
    for ind in prange(size):
        if not ghost_node[ind]:
            if solid[ind]:
                x = ind // shape[1]
                y = ind - x * shape[1]
                for k in range(no_of_directions):
                    x_nb = x + cx[k]
                    y_nb = y + cy[k]
                    ind_nb = x_nb * shape[1] + y_nb
                    if not solid[ind_nb]:
                        solid_boundary[ind] = True
                        break
            elif not solid[ind]:
                x = ind // shape[1]
                y = ind - x * shape[1]
                for k in range(no_of_directions):
                    x_nb = x + cx[k]
                    y_nb = y + cy[k]
                    ind_nb = (x_nb * shape[1] + y_nb)
                    if solid[ind_nb]:
                        fluid_boundary[ind] = True
                        solid_id[ind] = solid_id[ind_nb]
                        break


@numba.njit(parallel=True, nogil=True)
def check_fluid_boundary_overlap(
    size,
    shape,
    cx,
    cy,
    no_of_directions,
    solid_id,
    fluid_boundary,
    ghost_node
):
    """
    Compute fluid solid boundary nodes
    Args:

    Returns:

    """
    sum_fluid_boundary_overlap = 0
    for ind in prange(size):
        if not ghost_node[ind]:
            if fluid_boundary[ind]:
                x = ind // shape[1]
                y = ind - x * shape[1]
                for k in range(no_of_directions):
                    x_nb = x + cx[k]
                    y_nb = y + cy[k]
                    ind_nb = x_nb * shape[1] + y_nb
                    if not (solid_id[ind_nb] == solid_id[ind] or
                            solid_id[ind_nb] == -1):
                        sum_fluid_boundary_overlap += 1
                        break
    return sum_fluid_boundary_overlap


# --------------------------------------------------------------------------#
""" Kernels to compute forces on obstacles """


@numba.njit(parallel=True, nogil=True)
def compute_forces_torque(
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
    Compute forces acting on solid for density based
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
