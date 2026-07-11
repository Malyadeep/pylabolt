import numpy as np
import numba
from numba import prange


@numba.njit(parallel=True, nogil=True)
def compute_residue_scalar(
    size,
    solid,
    ghost_node,
    field,
    field_old
):
    """
    Kernel to compute residue for scalar fields
    Args:

    Returns:

    """
    numerator = 0
    denominator = 0
    for ind in prange(size):
        if not solid[ind] and not ghost_node[ind]:
            field_old_local = field_old[ind]
            field_local = field[ind]
            diff_field = field_local - field_old_local
            numerator += diff_field * diff_field
            denominator += field_old_local * field_old_local
            field_old[ind] = field_local
    return np.array([
        numerator,
        denominator
    ])


@numba.njit(parallel=True, nogil=True)
def compute_residue_vector(
    size,
    solid,
    ghost_node,
    field,
    field_old
):
    """
    Kernel to compute residue for scalar fields
    Args:

    Returns:

    """
    numerator_x, numerator_y = 0, 0
    denominator_x, denominator_y = 0, 0
    for ind in prange(size):
        if not solid[ind] and not ghost_node[ind]:
            field_old_local_x = field_old[ind, 0]
            field_old_local_y = field_old[ind, 1]
            field_local_x = field[ind, 0]
            field_local_y = field[ind, 1]
            diff_field_x = field_local_x - field_old_local_x
            diff_field_y = field_local_y - field_old_local_y
            numerator_x += diff_field_x * diff_field_x
            numerator_y += diff_field_y * diff_field_y
            denominator_x += field_old_local_x * field_old_local_x
            denominator_y += field_old_local_y * field_old_local_y
            field_old[ind, 0] = field_local_x
            field_old[ind, 1] = field_local_y
    return np.array([
        numerator_x,
        denominator_x,
        numerator_y,
        denominator_y
    ])
