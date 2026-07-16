import numpy as np
import numba
from numba import prange


@numba.njit(inline="always", nogil=True)
def equilibrium_density_based_second_order(
    cx,
    cy,
    weights,
    no_of_directions,
    inv_cs_2,
    inv_cs_4,
    density_local,
    velocity_local,
    pop_local
):
    """
    Equilibrium kernel for density based second order
    equilibrium distribution
    Args:

    Returns:

    """
    velocity_local_x = velocity_local[0]
    velocity_local_y = velocity_local[1]
    u2 = (velocity_local_x * velocity_local_x +
          velocity_local_y * velocity_local_y)
    for k in range(no_of_directions):
        cu = cx[k] * velocity_local_x + cy[k] * velocity_local_y
        pop_local[k] = weights[k] * density_local * (
            1 + inv_cs_2 * cu + 0.5 * inv_cs_4 * cu * cu -
            0.5 * inv_cs_2 * u2
        )


@numba.njit(parallel=True, nogil=True)
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
    for ind in prange(size):
        if not solid[ind] and not ghost_node[ind]:
            pop_eq = np.zeros(no_of_directions, dtype=np.float64)
            equilibrium_density_based_second_order(
                cx,
                cy,
                weights,
                no_of_directions,
                inv_cs_2,
                inv_cs_4,
                density[ind],
                velocity[ind],
                pop_eq
            )
            for k in range(no_of_directions):
                pop[ind, k] = pop_eq[k]
                pop_new[ind, k] = pop_eq[k]
