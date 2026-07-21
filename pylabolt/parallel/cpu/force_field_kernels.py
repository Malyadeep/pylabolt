import numba
from numba import prange


@numba.njit(parallel=True, nogil=True)
def compute_gravity_force(
    size,
    solid,
    ghost_node,
    density,
    force_field,
    gravity
):
    """
    Compute local gravity force acting on fluid
    Args:

    Returns:

    """
    for ind in prange(size):
        if not solid[ind] and not ghost_node[ind]:
            density_local = density[ind]
            force_field[ind, 0] = density_local * gravity[0]
            force_field[ind, 1] = density_local * gravity[1]
