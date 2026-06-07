import numpy as np
from numba import prange

from pylabolt.parallel.domain import local_to_global


def compute_obstacle_boundary(
    # domain args
    size,
    shape,
    # lattice args
    cx,
    cy,
    no_of_directions,
    # fields
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
    for ind in prange(size):
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


def compute_normals_circle(
    # domain args
    shape,
    offset,
    # obstacle properties
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


def compute_normals_ellipse(
    # domain args
    shape,
    offset,
    # obstacle properties
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
        x = rx * cos_alpha + ry * sin_alpha
        y = - rx * sin_alpha + ry * cos_alpha
        gx = x / (semi_major_axis * semi_major_axis)
        gy = y / (semi_minor_axis * semi_minor_axis)
        x_g = gx * cos_alpha - gy * sin_alpha
        y_g = gx * sin_alpha + gy * cos_alpha
        mag = np.sqrt(x_g * x_g + y_g * y_g)
        surface_normals_fluid[itr, 0] = rx / mag
        surface_normals_fluid[itr, 1] = ry / mag
