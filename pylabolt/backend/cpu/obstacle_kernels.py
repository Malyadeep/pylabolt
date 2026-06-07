import numpy as np
import numba
from numba import prange


def compute_obstacle_boundary(
    # domain args
    size,
    shape,
    # lattice args
    cx,
    cy,
    no_of_directions,
    # fields
    solid,
    solid_boundary,
    fluid_boundary,
    ghost_node
):
    solid_boundary_nodes = []
    fluid_boundary_nodes = []
    for ind in prange(size):
        if not ghost_node[ind]:
            if solid[ind]:
                i = ind // shape[1]
                j = ind - i * shape[1]
                for k in range(no_of_directions):
                    i_nb = i + cx[k]
                    j_nb = j + cy[k]
                    if not solid[(i_nb * shape[1] + j_nb)]:
                        solid_boundary_nodes.append(ind)
                        solid_boundary[ind] = True
                        break
            elif not solid[ind]:
                i = ind // shape[1]
                j = ind - i * shape[1]
                for k in range(no_of_directions):
                    i_nb = i + cx[k]
                    j_nb = j + cy[k]
                    if solid[(i_nb * shape[1] + j_nb)]:
                        fluid_boundary_nodes.append(ind)
                        fluid_boundary[ind] = True
                        break
    return np.array(solid_boundary_nodes, dtype=np.int64), \
        np.array(fluid_boundary_nodes, dtype=np.int64)
