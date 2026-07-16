from numba import cuda


@cuda.jit(device=True)
def equilibrium_density_based_second_order(
    weight,
    inv_cs_2,
    inv_cs_4,
    density_local,
    cu,
    u2
):
    """
    Equilibrium kernel for density based second order
    equilibrium distribution per lattice direction
    Args:

    Returns:

    """
    pop_local = weight * density_local * (
        1 + inv_cs_2 * cu + 0.5 * inv_cs_4 * cu * cu -
        0.5 * inv_cs_2 * u2
    )
    return pop_local


@cuda.jit
def initialize_pop_density_based_second_order(
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
    pop_new
):
    """
    Initialization of populations using density based
    second order equilibrium
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
            for k in range(no_of_directions):
                cu = cx[k] * velocity_local_x + cy[k] * velocity_local_y
                pop_eq = equilibrium_density_based_second_order(
                    weights[k],
                    inv_cs_2,
                    inv_cs_4,
                    density[ind],
                    cu,
                    u2
                )
                pop[ind, k] = pop_eq
                pop_new[ind, k] = pop_eq
