import numpy as np
import math
from numba import cuda, int64

from pylabolt.parallel.gpu.MPI_kernels import local_to_global
from pylabolt.parallel.backend import REDUCE_BLOCK_SIZE

# --------------------------------------------------------------------------#
""" Kernels for obstacle type circle """


@cuda.jit(device=True)
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
        if abs(rx + Nx) < abs(rx_min):
            rx_min = rx + Nx
        if abs(rx - Nx) < abs(rx_min):
            rx_min = rx - Nx
    if y_periodic:
        if abs(ry + Ny) < abs(ry_min):
            ry_min = ry + Ny
        if abs(ry - Ny) < abs(ry_min):
            ry_min = ry - Ny
    dist_sq_from_center = rx_min * rx_min + ry_min * ry_min
    inside_solid = False
    if dist_sq_from_center <= radius * radius:
        inside_solid = True
    return inside_solid, rx_min, ry_min


@cuda.jit
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
    ind = cuda.grid(1)
    if ind < size:
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


@cuda.jit
def compute_normals_circle(
    size,
    shape,
    offset,
    grid_global_shape,
    x_periodic,
    y_periodic,
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
    Nx, Ny = grid_global_shape
    ind = cuda.grid(1)
    if ind < size:
        if (solid_id[ind] == current_solid_id and
                (solid_boundary[ind] or fluid_boundary[ind])):
            x = ind // shape[1]
            y = ind - x * shape[1]
            x_global, y_global = local_to_global(
                x - 1, y - 1, offset
            )
            rx = x_global - center[0]
            ry = y_global - center[1]
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
            mag = math.sqrt(rx_min * rx_min + ry_min * ry_min)
            surface_normals[ind, 0] = rx_min / mag
            surface_normals[ind, 1] = ry_min / mag


# --------------------------------------------------------------------------#
""" Kernels for obstacle type ellipse """


@cuda.jit(device=True)
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
        if abs(rx + Nx) < abs(rx_min):
            rx_min = rx + Nx
        if abs(rx - Nx) < abs(rx_min):
            rx_min = rx - Nx
    if y_periodic:
        if abs(ry + Ny) < abs(ry_min):
            ry_min = ry + Ny
        if abs(ry - Ny) < abs(ry_min):
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


@cuda.jit
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
    ind = cuda.grid(1)
    if ind < size:
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


@cuda.jit
def compute_normals_ellipse(
    size,
    shape,
    offset,
    grid_global_shape,
    x_periodic,
    y_periodic,
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
    cos_alpha = np.cos(inclination_angle)
    sin_alpha = np.sin(inclination_angle)
    Nx, Ny = grid_global_shape
    ind = cuda.grid(1)
    if ind < size:
        if (solid_id[ind] == current_solid_id and
                (solid_boundary[ind] or fluid_boundary[ind])):
            x = ind // shape[1]
            y = ind - x * shape[1]
            x_global, y_global = local_to_global(
                x - 1, y - 1, offset
            )
            rx = x_global - center[0]
            ry = y_global - center[1]
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
            x_proj = rx_min * cos_alpha + ry_min * sin_alpha
            y_proj = - rx_min * sin_alpha + ry_min * cos_alpha
            gx = x_proj / (semi_major_axis * semi_major_axis)
            gy = y_proj / (semi_minor_axis * semi_minor_axis)
            x_g = gx * cos_alpha - gy * sin_alpha
            y_g = gx * sin_alpha + gy * cos_alpha
            mag = math.sqrt(x_g * x_g + y_g * y_g)
            surface_normals[ind, 0] = x_g / mag
            surface_normals[ind, 1] = y_g / mag


# --------------------------------------------------------------------------#
""" Kernels to compute obstacle boundary nodes """


@cuda.jit
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
    ind = cuda.grid(1)
    if ind < size:
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


@cuda.jit
def check_fluid_boundary_overlap(
    size,
    shape,
    cx,
    cy,
    no_of_directions,
    solid_id,
    fluid_boundary,
    ghost_node,
    partial_fluid_boundary_overlap
):
    """
    Check and count number of overlapping fluid boundary nodes
    Args:

    Returns:

    """
    ind = cuda.grid(1)
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_dim = cuda.blockDim.x

    shared_fluid_boundary_overlap = cuda.shared.array(REDUCE_BLOCK_SIZE, int64)

    sum_fluid_boundary_overlap = 0

    if ind < size:
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
                        sum_fluid_boundary_overlap = 1
                        break

    shared_fluid_boundary_overlap[thread_idx] = sum_fluid_boundary_overlap

    cuda.syncthreads()

    stride = block_dim // 2
    while stride > 0:
        if thread_idx < stride:
            shared_fluid_boundary_overlap[thread_idx] +=\
                shared_fluid_boundary_overlap[thread_idx + stride]

        cuda.syncthreads()

        stride = stride // 2

    if thread_idx == 0:
        partial_fluid_boundary_overlap[block_idx] =\
            shared_fluid_boundary_overlap[0]


@cuda.jit
def reduce_fluid_boundary_overlap(
    partial_size,
    partial_fluid_boundary_overlap,
    local_count_fluid_boundary_overlap
):
    """
    Reduction kernel to recursively count the number of
    overlapping fluid boundary nodes
    Args:

    Returns:

    """
    ind = cuda.grid(1)
    thread_idx = cuda.threadIdx.x
    block_idx = cuda.blockIdx.x
    block_dim = cuda.blockDim.x

    shared_fluid_boundary_overlap = cuda.shared.array(REDUCE_BLOCK_SIZE, int64)

    if ind < partial_size:
        shared_fluid_boundary_overlap[thread_idx] =\
            partial_fluid_boundary_overlap[ind]
    else:
        shared_fluid_boundary_overlap[thread_idx] = 0

    cuda.syncthreads()

    stride = block_dim // 2
    while stride > 0:
        if thread_idx < stride:
            shared_fluid_boundary_overlap[thread_idx] +=\
                shared_fluid_boundary_overlap[thread_idx + stride]

        cuda.syncthreads()

        stride = stride // 2

    if thread_idx == 0:
        partial_fluid_boundary_overlap[block_idx] =\
            shared_fluid_boundary_overlap[0]
        if block_idx == 0:
            local_count_fluid_boundary_overlap[0] =\
                shared_fluid_boundary_overlap[0]


# --------------------------------------------------------------------------#
""" Kernels to update obstacle position and velocities """


@cuda.jit
def update_position_velocity(
    grid_global_shape,
    x_periodic,
    y_periodic,
    gravity,
    no_of_obstacles,
    force,
    torque,
    linear_velocity,
    angular_velocity,
    center,
    inclination_angle,
    ref_point,
    mass,
    moment_of_inertia,
    static,
    calculated,
    rotation_allowed,
    translation_allowed
):
    """
    Update position and velocities of moving obstacles
    Args:

    Returns:

    """
    obs_no = cuda.grid(1)
    if obs_no < no_of_obstacles:
        if not static[obs_no, 0]:
            angular_velocity_old = angular_velocity[obs_no, 0]
            linear_velocity_old = linear_velocity[obs_no]
            torque_temp = 0
            force_temp_x, force_temp_y = 0, 0
            if calculated[obs_no, 0]:
                if rotation_allowed[obs_no, 0]:
                    angular_velocity[obs_no, 0] +=\
                        torque[obs_no, 0] / moment_of_inertia[obs_no, 0]
                    torque_temp = torque[obs_no, 0]
                if translation_allowed[obs_no, 0]:
                    linear_velocity[obs_no, 0] +=\
                        force[obs_no, 0] / mass[obs_no, 0] + gravity[0]
                    linear_velocity[obs_no, 1] +=\
                        force[obs_no, 1] / mass[obs_no, 0] + gravity[1]
                    force_temp_x = force[obs_no, 0]
                    force_temp_y = force[obs_no, 1]
            if rotation_allowed[obs_no, 0]:
                inclination_angle[obs_no, 0] = angular_velocity_old +\
                    0.5 * torque_temp / moment_of_inertia[obs_no, 0]
            if translation_allowed[obs_no, 0]:
                if x_periodic:
                    center[obs_no, 0] = (
                        center[obs_no, 0] + linear_velocity_old[0] +
                        0.5 * (force_temp_x / mass[obs_no, 0] + gravity[0]) +
                        grid_global_shape[0]
                    ) % grid_global_shape[0]
                else:
                    center[obs_no, 0] += linear_velocity_old[0] +\
                        0.5 * (force_temp_x / mass[obs_no, 0] + gravity[0])
                ref_point[obs_no, 0] = center[obs_no, 0]
                if y_periodic:
                    center[obs_no, 1] = (
                        center[obs_no, 1] + linear_velocity_old[1] +
                        0.5 * (force_temp_y / mass[obs_no, 0] + gravity[1]) +
                        grid_global_shape[1]
                    ) % grid_global_shape[1]
                else:
                    center[obs_no, 1] += linear_velocity_old[1] +\
                        0.5 * (force_temp_y / mass[obs_no, 0] + gravity[1])
                ref_point[obs_no, 1] = center[obs_no, 1]
