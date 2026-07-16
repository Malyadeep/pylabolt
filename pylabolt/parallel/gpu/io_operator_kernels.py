from numba import cuda


@cuda.jit
def copy_inner_data_scalar(
    inner_size,
    inner_shape,
    shape,
    field,
    field_save
):
    """
    Kernel to copy core domain data for scalar field
    Args:

    Returns:

    """
    ind = cuda.grid(1)
    if ind < inner_size:
        Ny_inner = inner_shape[1]
        Ny_outer = shape[1]
        i = ind // Ny_inner
        j = ind - i * Ny_inner
        ind_outer = (i + 1) * Ny_outer + j + 1
        field_save[ind] = field[ind_outer]


@cuda.jit
def copy_inner_data_vector(
    inner_size,
    inner_shape,
    shape,
    field,
    field_save
):
    """
    Kernel to copy core domain data for vector field
    with 2 components
    Args:

    Returns:

    """
    ind = cuda.grid(1)
    if ind < inner_size:
        Ny_inner = inner_shape[1]
        Ny_outer = shape[1]
        i = ind // Ny_inner
        j = ind - i * Ny_inner
        ind_outer = (i + 1) * Ny_outer + j + 1
        field_save[ind, 0] = field[ind_outer, 0]
        field_save[ind, 1] = field[ind_outer, 1]
