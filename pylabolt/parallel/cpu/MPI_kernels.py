import numba
from numba import prange


@numba.njit(inline="always")
def local_to_global(i, j, offset):
    """
    convert local sub-domain index to global index
    Args:
        i: int
        j: int
        offset: (mesh.dimensions,) int array
    Returns:
        i_global: int
        j_global: int
    """
    return i + offset[0], j + offset[1]


@numba.njit(inline="always")
def global_to_local(i_global, j_global, offset):
    """
    convert global index to local sub-domain index
    Args:
        i_global: int
        j_global: int
        offset: (mesh.dimensions,) int array
    Returns:
        i: int
        j: int
    """
    return i_global - offset[0], j_global - offset[1]


@numba.njit(parallel=True, nogil=True)
def send_copy_y_vector(
    send_buff,
    field,
    layout_start,
    layout_end,
    shape,
    x=0
):
    for y in prange(0, shape[1]):
        ind = x * shape[1] + y
        for itr in range(layout_start, layout_end):
            send_buff[y, itr] = field[ind, itr - layout_start]


@numba.njit(parallel=True, nogil=True)
def send_copy_y_scalar(
    send_buff,
    field,
    layout_start,
    layout_end,
    shape,
    x=0
):
    for y in prange(0, shape[1]):
        ind = x * shape[1] + y
        send_buff[y, layout_start] = field[ind]


@numba.njit(parallel=True, nogil=True)
def recv_copy_y_vector(
    recv_buff,
    field,
    layout_start,
    layout_end,
    shape,
    x=0
):
    for y in prange(0, shape[1]):
        ind = x * shape[1] + y
        for itr in range(layout_start, layout_end):
            field[ind, itr - layout_start] = recv_buff[y, itr]


@numba.njit(parallel=True, nogil=True)
def recv_copy_y_scalar(
    recv_buff,
    field,
    layout_start,
    layout_end,
    shape,
    x=0
):
    for y in prange(0, shape[1]):
        ind = x * shape[1] + y
        field[ind] = recv_buff[y, layout_start]


@numba.njit(parallel=True, nogil=True)
def send_copy_x_vector(
    send_buff,
    field,
    layout_start,
    layout_end,
    shape,
    y=0
):
    for x in prange(0, shape[0]):
        ind = x * shape[1] + y
        for itr in range(layout_start, layout_end):
            send_buff[x, itr] = field[ind, itr - layout_start]


@numba.njit(parallel=True, nogil=True)
def send_copy_x_scalar(
    send_buff,
    field,
    layout_start,
    layout_end,
    shape,
    y=0
):
    for x in prange(0, shape[0]):
        ind = x * shape[1] + y
        send_buff[x, layout_start] = field[ind]


@numba.njit(parallel=True, nogil=True)
def recv_copy_x_vector(
    recv_buff,
    field,
    layout_start,
    layout_end,
    shape,
    y=0
):
    for x in prange(0, shape[0]):
        ind = x * shape[1] + y
        for itr in range(layout_start, layout_end):
            field[ind, itr - layout_start] = recv_buff[x, itr]


@numba.njit(parallel=True, nogil=True)
def recv_copy_x_scalar(
    recv_buff,
    field,
    layout_start,
    layout_end,
    shape,
    y=0
):
    for x in prange(0, shape[0]):
        ind = x * shape[1] + y
        field[ind] = recv_buff[x, layout_start]
