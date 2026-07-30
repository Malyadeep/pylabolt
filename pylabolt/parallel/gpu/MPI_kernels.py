from numba import cuda


@cuda.jit(device=True)
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


@cuda.jit(device=True)
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


@cuda.jit
def exchange_y_scalar(
    field,
    shape
):
    y = cuda.grid(1)
    if y < shape[1]:
        Nx = shape[0]
        Ny = shape[1]
        ind_left_ghost = y
        ind_right_ghost = (Nx - 1) * Ny + y
        ind_left_inner = Ny + y
        ind_right_inner = (Nx - 2) * Ny + y
        field[ind_left_ghost] = field[ind_right_inner]
        field[ind_right_ghost] = field[ind_left_inner]


@cuda.jit
def exchange_y_vector(
    field,
    shape
):
    y = cuda.grid(1)
    if y < shape[1]:
        Nx = shape[0]
        Ny = shape[1]
        ind_left_ghost = y
        ind_right_ghost = (Nx - 1) * Ny + y
        ind_left_inner = Ny + y
        ind_right_inner = (Nx - 2) * Ny + y
        for k in range(field.shape[1]):
            field[ind_left_ghost, k] = field[ind_right_inner, k]
            field[ind_right_ghost, k] = field[ind_left_inner, k]


@cuda.jit
def exchange_x_scalar(
    field,
    shape
):
    x = cuda.grid(1)
    if x < shape[0]:
        Ny = shape[1]
        ind_bottom_ghost = x * Ny
        ind_top_ghost = x * Ny + Ny - 1
        ind_bottom_inner = x * Ny + 1
        ind_top_inner = x * Ny + Ny - 2
        field[ind_bottom_ghost] = field[ind_top_inner]
        field[ind_top_ghost] = field[ind_bottom_inner]


@cuda.jit
def exchange_x_vector(
    field,
    shape
):
    x = cuda.grid(1)
    if x < shape[0]:
        Ny = shape[1]
        ind_bottom_ghost = x * Ny
        ind_top_ghost = x * Ny + Ny - 1
        ind_bottom_inner = x * Ny + 1
        ind_top_inner = x * Ny + Ny - 2
        for k in range(field.shape[1]):
            field[ind_bottom_ghost, k] = field[ind_top_inner, k]
            field[ind_top_ghost, k] = field[ind_bottom_inner, k]
