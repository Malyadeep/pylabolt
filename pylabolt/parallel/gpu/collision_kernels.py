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
    force_field,
    pop,
    pop_new,
    omega
):
    """
    BGK collision kernel for single phase flow
    for second order density based equilibrium distribution
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
    force_field,
    pop,
    pop_new,
    omega
):
    """
    BGK collision kernel for single phase flow
    for second order density based equilibrium distribution
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
            force_local_x = force_field[ind, 0]
            force_local_y = force_field[ind, 1]
            u2 = (velocity_local_x * velocity_local_x +
                  velocity_local_y * velocity_local_y)
            for k in range(no_of_directions):
                cu = cx[k] * velocity_local_x + cy[k] * velocity_local_y
                pop_eq = weights[k] * density_local * (
                    1 + inv_cs_2 * cu + 0.5 * inv_cs_4 * cu * cu -
                    0.5 * inv_cs_2 * u2
                )
                force_term = weights[k] * (
                    cx[k] * force_local_x + cy[k] * force_local_y
                ) * inv_cs_2
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
    force_field,
    pop,
    pop_new,
    omega
):
    """
    BGK collision kernel for single phase flow
    for second order density based equilibrium distribution
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
            force_local_x = force_field[ind, 0]
            force_local_y = force_field[ind, 1]
            u2 = (velocity_local_x * velocity_local_x +
                  velocity_local_y * velocity_local_y)
            for k in range(no_of_directions):
                cu = cx[k] * velocity_local_x + cy[k] * velocity_local_y
                pop_eq = weights[k] * density_local * (
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


@cuda.jit(device=True)
def equilibrium_second_order(
    cx_k,
    cy_k,
    weights_k,
    inv_cs_2,
    inv_cs_4,
    u2,
    velocity_local_x,
    velocity_local_y,
    density_local
):
    cu = cx_k * velocity_local_x + cy_k * velocity_local_y
    return weights_k * density_local * (
        1 + inv_cs_2 * cu + 0.5 * inv_cs_4 * cu * cu -
        0.5 * inv_cs_2 * u2
    )


@cuda.jit(device=True)
def guo_force(
    cx_k,
    cy_k,
    weights_k,
    inv_cs_2,
    inv_cs_4,
    uF,
    velocity_local_x,
    velocity_local_y,
    force_local_x,
    force_local_y
):
    cu = cx_k * velocity_local_x + cy_k * velocity_local_y
    cF = cx_k * force_local_x + cy_k * force_local_y
    return weights_k * (
        cF * (inv_cs_2 + cu * inv_cs_4) - uF * inv_cs_2
    )


@cuda.jit
def MRT_density_based_second_order_None(
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
    force_field,
    pop,
    pop_new,
    omega
):
    """
    MRT collision kernel for single phase flow
    for second order density based equilibrium distribution
    without any forcing
    Args:

    Returns:

    """
    ind = cuda.grid(1)
    if ind < size:
        if not solid[ind] and not ghost_node[ind]:
            velocity_local_x = velocity[ind, 0]
            velocity_local_y = velocity[ind, 1]
            u2 = (velocity_local_x * velocity_local_x +
                  velocity_local_y * velocity_local_y)
            density_local = density[ind]
            pop_eq_0 = weights[0] * density_local * (1 - 0.5 * inv_cs_2 * u2)
            pop_eq_1 = equilibrium_second_order(
                cx[1], cy[1], weights[1], inv_cs_2, inv_cs_4, u2,
                velocity_local_x, velocity_local_y, density_local
            )
            pop_eq_2 = equilibrium_second_order(
                cx[2], cy[2], weights[2], inv_cs_2, inv_cs_4, u2,
                velocity_local_x, velocity_local_y, density_local
            )
            pop_eq_3 = equilibrium_second_order(
                cx[3], cy[3], weights[3], inv_cs_2, inv_cs_4,
                u2, velocity_local_x, velocity_local_y, density_local
            )
            pop_eq_4 = equilibrium_second_order(
                cx[4], cy[4], weights[4], inv_cs_2, inv_cs_4,
                u2, velocity_local_x, velocity_local_y, density_local
            )
            pop_eq_5 = equilibrium_second_order(
                cx[5], cy[5], weights[5], inv_cs_2, inv_cs_4,
                u2, velocity_local_x, velocity_local_y, density_local
            )
            pop_eq_6 = equilibrium_second_order(
                cx[6], cy[6], weights[6], inv_cs_2, inv_cs_4,
                u2, velocity_local_x, velocity_local_y, density_local
            )
            pop_eq_7 = equilibrium_second_order(
                cx[7], cy[7], weights[7], inv_cs_2, inv_cs_4,
                u2, velocity_local_x, velocity_local_y, density_local
            )
            pop_eq_8 = equilibrium_second_order(
                cx[8], cy[8], weights[8], inv_cs_2, inv_cs_4,
                u2, velocity_local_x, velocity_local_y, density_local
            )

            straight_terms = (omega - 1) * 0.25 * (
                (pop_new[ind, 1] - pop_eq_1) -
                (pop_new[ind, 2] - pop_eq_2) +
                (pop_new[ind, 3] - pop_eq_3) -
                (pop_new[ind, 4] - pop_eq_4)
            )
            corner_terms = (omega - 1) * 0.25 * (
                (pop_new[ind, 5] - pop_eq_5) -
                (pop_new[ind, 6] - pop_eq_6) +
                (pop_new[ind, 7] - pop_eq_7) -
                (pop_new[ind, 8] - pop_eq_8)
            )

            pop[ind, 0] = pop_eq_0
            pop[ind, 1] = pop_eq_1 - straight_terms
            pop[ind, 2] = pop_eq_2 + straight_terms
            pop[ind, 3] = pop_eq_3 - straight_terms
            pop[ind, 4] = pop_eq_4 + straight_terms
            pop[ind, 5] = pop_eq_5 - corner_terms
            pop[ind, 6] = pop_eq_6 + corner_terms
            pop[ind, 7] = pop_eq_7 - corner_terms
            pop[ind, 8] = pop_eq_8 + corner_terms


@cuda.jit
def MRT_density_based_second_order_guo_second_order(
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
    force_field,
    pop,
    pop_new,
    omega
):
    """
    MRT collision kernel for single phase flow
    for second order density based equilibrium distributions
    with second order Guo forcing
    Args:

    Returns:

    """
    ind = cuda.grid(1)
    if ind < size:
        if not solid[ind] and not ghost_node[ind]:
            velocity_local_x = velocity[ind, 0]
            velocity_local_y = velocity[ind, 1]
            force_local_x = force_field[ind, 0]
            force_local_y = force_field[ind, 1]
            density_local = density[ind]
            u2 = (velocity_local_x * velocity_local_x +
                  velocity_local_y * velocity_local_y)
            uF = (velocity_local_x * force_local_x +
                  velocity_local_y * force_local_y)
            pop_eq_0 = weights[0] * density_local * (1 - 0.5 * inv_cs_2 * u2)
            pop_eq_1 = equilibrium_second_order(
                cx[1], cy[1], weights[1], inv_cs_2, inv_cs_4, u2,
                velocity_local_x, velocity_local_y, density_local
            )
            pop_eq_2 = equilibrium_second_order(
                cx[2], cy[2], weights[2], inv_cs_2, inv_cs_4, u2,
                velocity_local_x, velocity_local_y, density_local
            )
            pop_eq_3 = equilibrium_second_order(
                cx[3], cy[3], weights[3], inv_cs_2, inv_cs_4,
                u2, velocity_local_x, velocity_local_y, density_local
            )
            pop_eq_4 = equilibrium_second_order(
                cx[4], cy[4], weights[4], inv_cs_2, inv_cs_4,
                u2, velocity_local_x, velocity_local_y, density_local
            )
            pop_eq_5 = equilibrium_second_order(
                cx[5], cy[5], weights[5], inv_cs_2, inv_cs_4,
                u2, velocity_local_x, velocity_local_y, density_local
            )
            pop_eq_6 = equilibrium_second_order(
                cx[6], cy[6], weights[6], inv_cs_2, inv_cs_4,
                u2, velocity_local_x, velocity_local_y, density_local
            )
            pop_eq_7 = equilibrium_second_order(
                cx[7], cy[7], weights[7], inv_cs_2, inv_cs_4,
                u2, velocity_local_x, velocity_local_y, density_local
            )
            pop_eq_8 = equilibrium_second_order(
                cx[8], cy[8], weights[8], inv_cs_2, inv_cs_4,
                u2, velocity_local_x, velocity_local_y, density_local
            )

            straight_terms = (omega - 1) * 0.25 * (
                (pop_new[ind, 1] - pop_eq_1) -
                (pop_new[ind, 2] - pop_eq_2) +
                (pop_new[ind, 3] - pop_eq_3) -
                (pop_new[ind, 4] - pop_eq_4)
            )
            corner_terms = (omega - 1) * 0.25 * (
                (pop_new[ind, 5] - pop_eq_5) -
                (pop_new[ind, 6] - pop_eq_6) +
                (pop_new[ind, 7] - pop_eq_7) -
                (pop_new[ind, 8] - pop_eq_8)
            )

            force_term_0 = guo_force(
                cx[0], cy[0], weights[0], inv_cs_2, inv_cs_4, uF,
                velocity_local_x, velocity_local_y, force_local_x,
                force_local_y
            )
            force_term_1 = guo_force(
                cx[1], cy[1], weights[1], inv_cs_2, inv_cs_4, uF,
                velocity_local_x, velocity_local_y, force_local_x,
                force_local_y
            )
            force_term_2 = guo_force(
                cx[2], cy[2], weights[2], inv_cs_2, inv_cs_4, uF,
                velocity_local_x, velocity_local_y, force_local_x,
                force_local_y
            )
            force_term_3 = guo_force(
                cx[3], cy[3], weights[3], inv_cs_2, inv_cs_4, uF,
                velocity_local_x, velocity_local_y, force_local_x,
                force_local_y
            )
            force_term_4 = guo_force(
                cx[4], cy[4], weights[4], inv_cs_2, inv_cs_4, uF,
                velocity_local_x, velocity_local_y, force_local_x,
                force_local_y
            )
            force_term_5 = guo_force(
                cx[5], cy[5], weights[5], inv_cs_2, inv_cs_4, uF,
                velocity_local_x, velocity_local_y, force_local_x,
                force_local_y
            )
            force_term_6 = guo_force(
                cx[6], cy[6], weights[6], inv_cs_2, inv_cs_4, uF,
                velocity_local_x, velocity_local_y, force_local_x,
                force_local_y
            )
            force_term_7 = guo_force(
                cx[7], cy[7], weights[7], inv_cs_2, inv_cs_4, uF,
                velocity_local_x, velocity_local_y, force_local_x,
                force_local_y
            )
            force_term_8 = guo_force(
                cx[8], cy[8], weights[8], inv_cs_2, inv_cs_4, uF,
                velocity_local_x, velocity_local_y, force_local_x,
                force_local_y
            )

            straight_terms_force = (omega - 1) * 0.125 * (
                force_term_1 -
                force_term_2 +
                force_term_3 -
                force_term_4
            )
            corner_terms_force = (omega - 1) * 0.125 * (
                force_term_5 -
                force_term_6 +
                force_term_7 -
                force_term_8
            )

            pop[ind, 0] = pop_eq_0 + 0.5 * force_term_0
            pop[ind, 1] = pop_eq_1 - straight_terms +\
                0.5 * force_term_1 - straight_terms_force
            pop[ind, 2] = pop_eq_2 + straight_terms +\
                0.5 * force_term_2 + straight_terms_force
            pop[ind, 3] = pop_eq_3 - straight_terms +\
                0.5 * force_term_3 - straight_terms_force
            pop[ind, 4] = pop_eq_4 + straight_terms +\
                0.5 * force_term_4 + straight_terms_force
            pop[ind, 5] = pop_eq_5 - corner_terms +\
                0.5 * force_term_5 - corner_terms_force
            pop[ind, 6] = pop_eq_6 + corner_terms +\
                0.5 * force_term_6 + corner_terms_force
            pop[ind, 7] = pop_eq_7 - corner_terms +\
                0.5 * force_term_7 - corner_terms_force
            pop[ind, 8] = pop_eq_8 + corner_terms +\
                0.5 * force_term_8 + corner_terms_force
