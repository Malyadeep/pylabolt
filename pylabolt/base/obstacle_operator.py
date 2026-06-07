import numpy as np

from pylabolt.utils.helpers import print_log
from pylabolt.parallel.domain import local_to_global
from pylabolt.parallel.cpu.obstacle_kernels import (
    compute_obstacle_boundary,
    compute_normals_circle,
    compute_normals_ellipse
)


class ObstacleOperator:
    def __init__(
        self,
        state,
        backend,
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
        self.find_obstacle_boundary_nodes_cpu(state)

    def find_obstacle_boundary_nodes_cpu(
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

    def find_obstacle_normals_cpu(
        self,
        state
    ):
        """
        Compute obstacle normals on both fluid and solid boundary
        Args:

        Returns:

        """
        for obstacle_no, obstacle in enumerate(state.obstacle.obstacles):
            surface_normals_solid = np.zeros(
                (obstacle.solid_boundary_nodes.shape[0], 2),
                dtype=state.control.precision
            )
            surface_normals_fluid = np.zeros(
                (obstacle.fluid_boundary_nodes.shape[0], 2),
                dtype=state.control.precision
            )
            if obstacle.type == "circle":
                compute_normals_circle(
                    state.domain.shape,
                    state.domain.offset,
                    obstacle.center,
                    obstacle.solid_boundary_nodes,
                    obstacle.fluid_boundary_nodes,
                    surface_normals_solid,
                    surface_normals_fluid
                )
            elif obstacle.type == "ellipse":
                compute_normals_ellipse(
                    state.domain.shape,
                    state.domain.offset,
                    obstacle.center,
                    obstacle.solid_boundary_nodes,
                    obstacle.fluid_boundary_nodes,
                    surface_normals_solid,
                    surface_normals_fluid
                )
            obstacle.surface_normals_solid = surface_normals_solid
            obstacle.surface_normals_fluid = surface_normals_fluid
