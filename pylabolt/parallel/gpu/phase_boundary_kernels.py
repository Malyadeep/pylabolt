import numba
from numba import prange


@numba.njit(parallel=True, nogil=True)
def bounce_back(
    solid,
    pop,
    pop_new,
    boundary_nodes,
    out_list,
    inv_list
):
    """
    Bounce back boundary condition kernel
    Args:

    Returns:

    """
    for itr in prange(boundary_nodes.shape[0]):
        ind = boundary_nodes[itr]
        if not solid[ind]:
            for k in range(out_list.shape[0]):
                pop_new[ind, inv_list[k]] = pop[ind, out_list[k]]
