import numpy as np
import numba
from numba import prange

from pylabolt.parallel.domain import local_to_global


@numba.njit(nogil=True)
def construct_circle(
    i_global,
    j_global,
    grid_global_shape,
    x_periodic,
    y_periodic,
    center,
    radius
):
    Nx, Ny = grid_global_shape
    rx = i_global - center[0]
    ry = j_global - center[1]
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


@numba.njit(nogil=True)
def construct_ellipse(
    i_global,
    j_global,
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
    rx = i_global - center[0]
    ry = j_global - center[1]
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
    x = rx_min * cos_alpha + ry_min * sin_alpha
    y = - rx_min * sin_alpha + ry_min * cos_alpha
    scaled_dist =\
        (x * x) / (semi_major_axis * semi_major_axis) +\
        (y * y) / (semi_minor_axis * semi_minor_axis)
    inside_solid = False
    if scaled_dist <= 1:
        inside_solid = True
    return inside_solid, rx_min, ry_min


# @numba.njit(parallel=True, nogil=True)
@numba.njit(parallel=False, nogil=True)
def compute_obstacle_boundary(
    size,
    shape,
    cx,
    cy,
    no_of_directions,
    solid,
    solid_boundary,
    fluid_boundary,
    ghost_node
):
    """
    Compute fluid solid boundary nodes
    Args:

    Returns:

    """
    solid_boundary_nodes = []
    fluid_boundary_nodes = []
    for ind in range(size):
        if not ghost_node[ind]:
            if solid[ind]:
                i = ind // shape[1]
                j = ind - i * shape[1]
                for k in range(no_of_directions):
                    i_nb = i + cx[k]
                    j_nb = j + cy[k]
                    if not solid[(i_nb * shape[1] + j_nb)]:
                        solid_boundary_nodes.append(ind)
                        solid_boundary[ind] = True
                        break
            elif not solid[ind]:
                i = ind // shape[1]
                j = ind - i * shape[1]
                for k in range(no_of_directions):
                    i_nb = i + cx[k]
                    j_nb = j + cy[k]
                    if solid[(i_nb * shape[1] + j_nb)]:
                        fluid_boundary_nodes.append(ind)
                        fluid_boundary[ind] = True
                        break
    return np.array(solid_boundary_nodes, dtype=np.int64), \
        np.array(fluid_boundary_nodes, dtype=np.int64)


@numba.njit(parallel=True, nogil=True)
def compute_normals_circle(
    shape,
    offset,
    center,
    solid_boundary_nodes,
    fluid_boundary_nodes,
    surface_normals_solid,
    surface_normals_fluid,
):
    """
    Compute surface normals for obstacle type circle
    Args:

    Returns:

    """
    # TODO: periodic wrapping of normals when solid crosses boundary
    for itr in prange(solid_boundary_nodes.shape[0]):
        ind = solid_boundary_nodes[itr]
        i = ind // shape[1]
        j = ind - i * shape[1]
        i_global, j_global = local_to_global(
            i - 1, j - 1, offset
        )
        rx = i_global - center[0]
        ry = j_global - center[1]
        mag = np.sqrt(rx * rx + ry * ry)
        surface_normals_solid[itr, 0] = rx / mag
        surface_normals_solid[itr, 1] = ry / mag
    for itr in prange(fluid_boundary_nodes.shape[0]):
        ind = fluid_boundary_nodes[itr]
        i = ind // shape[1]
        j = ind - i * shape[1]
        i_global, j_global = local_to_global(
            i - 1, j - 1, offset
        )
        rx = i_global - center[0]
        ry = j_global - center[1]
        mag = np.sqrt(rx * rx + ry * ry)
        surface_normals_fluid[itr, 0] = rx / mag
        surface_normals_fluid[itr, 1] = ry / mag


@numba.njit(parallel=True, nogil=True)
def compute_normals_ellipse(
    shape,
    offset,
    center,
    semi_major_axis,
    semi_minor_axis,
    inclination_angle,
    solid_boundary_nodes,
    fluid_boundary_nodes,
    surface_normals_solid,
    surface_normals_fluid,
):
    """
    Compute surface normals for obstacle type ellipse
    Args:

    Returns:

    """
    # TODO: periodic wrapping of normals when solid crosses boundary
    cos_alpha = np.cos(inclination_angle)
    sin_alpha = np.sin(inclination_angle)
    for itr in prange(solid_boundary_nodes.shape[0]):
        ind = solid_boundary_nodes[itr]
        i = ind // shape[1]
        j = ind - i * shape[1]
        i_global, j_global = local_to_global(
            i - 1, j - 1, offset
        )
        rx = i_global - center[0]
        ry = j_global - center[1]
        x = rx * cos_alpha + ry * sin_alpha
        y = - rx * sin_alpha + ry * cos_alpha
        gx = x / (semi_major_axis * semi_major_axis)
        gy = y / (semi_minor_axis * semi_minor_axis)
        x_g = gx * cos_alpha - gy * sin_alpha
        y_g = gx * sin_alpha + gy * cos_alpha
        mag = np.sqrt(x_g * x_g + y_g * y_g)
        surface_normals_solid[itr, 0] = x_g / mag
        surface_normals_solid[itr, 1] = y_g / mag
    for itr in prange(fluid_boundary_nodes.shape[0]):
        ind = fluid_boundary_nodes[itr]
        i = ind // shape[1]
        j = ind - i * shape[1]
        i_global, j_global = local_to_global(
            i - 1, j - 1, offset
        )
        rx = i_global - center[0]
        ry = j_global - center[1]
        x = rx * cos_alpha + ry * sin_alpha
        y = - rx * sin_alpha + ry * cos_alpha
        gx = x / (semi_major_axis * semi_major_axis)
        gy = y / (semi_minor_axis * semi_minor_axis)
        x_g = gx * cos_alpha - gy * sin_alpha
        y_g = gx * sin_alpha + gy * cos_alpha
        mag = np.sqrt(x_g * x_g + y_g * y_g)
        surface_normals_fluid[itr, 0] = x_g / mag
        surface_normals_fluid[itr, 1] = y_g / mag
