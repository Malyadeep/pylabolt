import numpy as np

from pylabolt.utils.helpers import print_log
from pylabolt.backend.domain import local_to_global
from pylabolt.backend.cpu.obstacle_kernels import (
    compute_obstacle_boundary
)


class ObstacleOperator:
    def __init__(
        self,
        state,
        fluid=False,
        phase=False,
        scalar=False,
        verbose=True
    ):
        """
        Obstacle operator - modifies obstacle and it's properties
        Attributes:

        """
        self.no_of_obstacles = len(state.obstacle.obstacles)
        # ------- Find solid-fluid boundary nodes ------- #
        self.find_obstacle_boundary_nodes(state)

    def find_obstacle_boundary_nodes(
        self,
        state
    ):
        """
        Creates obstacle boundary nodes. Sets solid and fluid boundary
        Args:

        Returns:

        """
        for obstacle_no, obstacle in enumerate(state.obstacle.obstacles):
            obstacle.solid_boundary_nodes, obstacle.fluid_boundary_nodes =\
                compute_obstacle_boundary(
                    state.domain.size,
                    state.domain.shape,
                    state.lattice.cx,
                    state.lattice.cy,
                    state.lattice.no_of_directions,
                    state.fields.solid,
                    state.fields.solid_boundary,
                    state.fields.fluid_boundary,
                    state.fields.ghost_node
                )
