from numba import cuda


@cuda.jit
def BGK_density_based_second_order_None(
    size,
    cx,
    cy,
    weights,
    no_of_directions,
    inv_cs_2,
    inv_cs_4,
    solid,
    ghost_node,
    density,
    velocity,
    pop,
    pop_new,
    gravity,
    omega
):
    """
    Collision kernel for single phase flow
    for density based equilibrium distributions
    without any forcing
    Args:

    Returns:

    """
    ind = cuda.grid(1)
    if ind < size:
        if not solid[ind] and not ghost_node[ind]:
            density_local = density[ind]
            velocity_local_x = velocity[ind, 0]
            velocity_local_y = velocity[ind, 1]
            u2 = (velocity_local_x * velocity_local_x +
                  velocity_local_y * velocity_local_y)
            for k in range(no_of_directions):
                cu = cx[k] * velocity_local_x + cy[k] * velocity_local_y
                pop_eq = weights[k] * density_local * (
                    1 + inv_cs_2 * cu + 0.5 * inv_cs_4 * cu * cu -
                    0.5 * inv_cs_2 * u2
                )
                pop[ind, k] = (1 - omega) * pop_new[ind, k] + omega * pop_eq


@cuda.jit
def BGK_density_based_second_order_guo_linear(
    size,
    cx,
    cy,
    weights,
    no_of_directions,
    inv_cs_2,
    inv_cs_4,
    solid,
    ghost_node,
    density,
    velocity,
    pop,
    pop_new,
    gravity,
    omega
):
    """
    Collision kernel for single phase flow
    for density based equilibrium distributions
    with linear Guo forcing
    Args:

    Returns:

    """
    ind = cuda.grid(1)
    if ind < size:
        if not solid[ind] and not ghost_node[ind]:
            density_local = density[ind]
            velocity_local_x = velocity[ind, 0]
            velocity_local_y = velocity[ind, 1]
            force_local_x = density_local * gravity[0]
            force_local_y = density_local * gravity[1]
            u2 = (velocity_local_x * velocity_local_x +
                  velocity_local_y * velocity_local_y)
            for k in range(no_of_directions):
                cu = cx[k] * velocity_local_x + cy[k] * velocity_local_y
                pop_eq = weights * density * (
                    1 + inv_cs_2 * cu + 0.5 * inv_cs_4 * cu * cu -
                    0.5 * inv_cs_2 * u2
                )
                force_term = weights[k] * (
                    cx[k] * force_local_x + cy[k] * force_local_y
                )
                pop[ind, k] = (
                    (1 - omega) * pop_new[ind, k] +
                    omega * pop_eq +
                    (1 - 0.5 * omega) * force_term
                )


@cuda.jit
def BGK_density_based_second_order_guo_second_order(
    size,
    cx,
    cy,
    weights,
    no_of_directions,
    inv_cs_2,
    inv_cs_4,
    solid,
    ghost_node,
    density,
    velocity,
    pop,
    pop_new,
    gravity,
    omega
):
    """
    Collision kernel for single phase flow
    for density based equilibrium distributions
    with second order Guo forcing
    Args:

    Returns:

    """
    ind = cuda.grid(1)
    if ind < size:
        if not solid[ind] and not ghost_node[ind]:
            density_local = density[ind]
            velocity_local_x = velocity[ind, 0]
            velocity_local_y = velocity[ind, 1]
            force_local_x = density_local * gravity[0]
            force_local_y = density_local * gravity[1]
            u2 = (velocity_local_x * velocity_local_x +
                  velocity_local_y * velocity_local_y)
            for k in range(no_of_directions):
                cu = cx[k] * velocity_local_x + cy[k] * velocity_local_y
                pop_eq = weights * density * (
                    1 + inv_cs_2 * cu + 0.5 * inv_cs_4 * cu * cu -
                    0.5 * inv_cs_2 * u2
                )
                const_x = (cx[k] - velocity_local_x) * inv_cs_2 +\
                    cu * cx[k] * inv_cs_4
                const_y = (cy[k] - velocity_local_y) * inv_cs_2 +\
                    cu * cy[k] * inv_cs_4
                force_term = weights[k] * (
                    const_x * force_local_x + const_y * force_local_y
                )
                pop[ind, k] = (
                    (1 - omega) * pop_new[ind, k] +
                    omega * pop_eq +
                    (1 - 0.5 * omega) * force_term
                )
