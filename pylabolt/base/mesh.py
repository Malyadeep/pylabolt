import numpy as np

from pylabolt.utils.helpers import print_log


class Mesh:
    def __init__(
        self,
        simulation,
        rank,
        verbose=True
    ):
        """
        Container for full grid information
        Attributes:
            dimensions: int
            grid_global_shape: (dimensions) int array
            grid_global_size: (dimensions) int
        """
        if not hasattr(simulation, "mesh_dict"):
            raise ValueError("mesh_dict not found in simulation.py file")
        mesh_dict = simulation.mesh_dict

        if "grid" not in mesh_dict:
            raise ValueError("grid missing in mesh_dict")
        grid = mesh_dict["grid"]

        if not isinstance(grid, list) or len(grid) != 2:
            raise ValueError("grid entry in mesh_dict must be a list [Nx, Ny]")
        grid = np.array(grid, dtype=np.int32)

        if np.any(grid == 0):
            raise ValueError("grid dimensions cannot be zero")
        if np.all(grid == 1):
            raise ValueError("grid specified is a point")
        if grid[0] == 1 or grid[1] == 1:
            self.dimensions = 1
        else:
            self.dimensions = 2

        self.grid_global_shape = grid
        self.grid_global_size = np.prod(self.grid_global_shape)
        print_log(
            "global grid size set: (" + str(self.grid_global_shape) + ")",
            rank, verbose
        )
