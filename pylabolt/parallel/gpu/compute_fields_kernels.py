from numba import cuda


@cuda.jit
def density_based_no_force(
    float_min,
    size,
    cx,
    cy,
    no_of_directions,
    solid,
    ghost_node,
    density,
    velocity,
    pop_new
):
    """
    Kernel to compute density and velocity from
    distributions
    Args:

    Returns:

    """
    ind = cuda.grid(1)
    if ind < size:
        if not solid[ind] and not ghost_node[ind]:
            density_sum = 0.
            velocity_x_sum = 0.
            velocity_y_sum = 0.
            pop_local = pop_new[ind]
            for k in range(no_of_directions):
                density_sum += pop_local[k]
                velocity_x_sum += cx[k] * pop_local[k]
                velocity_y_sum += cy[k] * pop_local[k]
            density[ind] = density_sum
            inv_density = 1 / (density_sum + float_min)
            velocity[ind, 0] = velocity_x_sum * inv_density
            velocity[ind, 1] = velocity_y_sum * inv_density


@cuda.jit
def density_based_force(
    float_min,
    size,
    cx,
    cy,
    no_of_directions,
    solid,
    ghost_node,
    density,
    velocity,
    pop_new,
    gravity
):
    """
    Kernel to compute density and velocity from
    distributions
    Args:

    Returns:

    """
    ind = cuda.grid(1)
    if ind < size:
        if not solid[ind] and not ghost_node[ind]:
            density_sum = 0.
            velocity_x_sum = 0.
            velocity_y_sum = 0.
            pop_local = pop_new[ind]
            for k in range(no_of_directions):
                density_sum += pop_local[k]
                velocity_x_sum += cx[k] * pop_local[k]
                velocity_y_sum += cy[k] * pop_local[k]
            velocity_x_sum += 0.5 * gravity[0]
            velocity_y_sum += 0.5 * gravity[1]
            density[ind] = density_sum
            inv_density = 1 / (density_sum + float_min)
            velocity[ind, 0] = velocity_x_sum * inv_density
            velocity[ind, 1] = velocity_y_sum * inv_density
