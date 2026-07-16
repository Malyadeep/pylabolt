from numba import cuda


@cuda.jit
def scalar_based_kernel(
    size,
    shape,
    cx,
    cy,
    weights,
    inv_list,
    no_of_directions,
    inv_cs_2,
    solid,
    ghost_node,
    density,
    velocity,
    pop,
    pop_new
):
    """
    Streaming kernel with scalar during bounce-back
    Args:

    Returns:

    """
    ind = cuda.grid(1)
    if ind < size:
        if not solid[ind] and not ghost_node[ind]:
            x = ind // shape[1]
            y = ind - x * shape[1]
            density_local = density[ind]
            pop_new[ind, 0] = pop[ind, 0]
            for k in range(1, no_of_directions):
                x_nb = x - cx[k]
                y_nb = y - cy[k]
                ind_nb = x_nb * shape[1] + y_nb
                if not solid[ind_nb]:
                    pop_new[ind, k] = pop[ind_nb, k]
                else:
                    k_inv = inv_list[k]
                    temp = 2 * weights[k_inv] * density_local * inv_cs_2 * (
                        cx[k_inv] * velocity[ind_nb, 0] +
                        cy[k_inv] * velocity[ind_nb, 1]
                    )
                    pop_new[ind, k] = pop[ind, k_inv] - temp


@cuda.jit
def no_scalar_based_kernel(
    size,
    shape,
    cx,
    cy,
    weights,
    inv_list,
    no_of_directions,
    inv_cs_2,
    solid,
    ghost_node,
    velocity,
    pop,
    pop_new
):
    """
    Streaming kernel without any scalar during bounce-back
    Args:

    Returns:

    """
    ind = cuda.grid(1)
    if ind < size:
        if not solid[ind] and not ghost_node[ind]:
            x = ind // shape[1]
            y = ind - x * shape[1]
            pop_new[ind, 0] = pop[ind, 0]
            for k in range(1, no_of_directions):
                x_nb = x - cx[k]
                y_nb = y - cy[k]
                ind_nb = x_nb * shape[1] + y_nb
                if not solid[ind_nb]:
                    pop_new[ind, k] = pop[ind, k]
                else:
                    k_inv = inv_list[k]
                    temp = 2 * weights[k_inv] * inv_cs_2 * (
                        cx[k_inv] * velocity[ind_nb, 0] +
                        cy[k_inv] * velocity[ind_nb, 1]
                    )
                    pop_new[ind, k] = pop[ind, k_inv] - temp
