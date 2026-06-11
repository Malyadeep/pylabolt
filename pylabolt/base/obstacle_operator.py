import numpy as np

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
        # ------- Find solid-fluid normals ------- #
        self.find_obstacle_normals_cpu(state)

        # """ Output initial fields for testing """
        # import os
        # obstacle_type = "circle"
        # if not os.path.isdir("procs_" + obstacle_type):
        #     os.makedirs("procs_" + obstacle_type)
        # np.savez(
        #     "procs_" + obstacle_type + "/proc_" +
        #     str(state.domain.mpi_rank) + ".npz",
        #     solid=state.fields.solid,
        #     solid_id=state.fields.solid_id,
        #     solid_boundary=state.fields.solid_boundary,
        #     fluid_boundary=state.fields.fluid_boundary,
        #     ghost_node=state.fields.ghost_node,
        #     Nx=state.mesh.grid_global_shape[0],
        #     Ny=state.mesh.grid_global_shape[1],
        #     offset=state.domain.offset,
        #     Nx_rank=state.domain.Nx_rank,
        #     Ny_rank=state.domain.Ny_rank
        # )

        self.set_backend(backend)

    def find_obstacle_boundary_nodes_cpu(
        self,
        state
    ):
        """
        Creates obstacle boundary nodes. Sets solid and fluid boundary
        Args:

        Returns:

        """
        for obstacle in state.obstacle.obstacles:
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
        for obstacle in state.obstacle.obstacles:
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
                    obstacle.semi_major_axis,
                    obstacle.semi_minor_axis,
                    obstacle.inclination_angle,
                    obstacle.solid_boundary_nodes,
                    obstacle.fluid_boundary_nodes,
                    surface_normals_solid,
                    surface_normals_fluid
                )
            obstacle.surface_normals_solid = surface_normals_solid
            obstacle.surface_normals_fluid = surface_normals_fluid

    def find_obstacle_boundary_nodes_gpu(
        self,
        state
    ):
        """
        Creates obstacle boundary nodes. Sets solid and fluid boundary
        Args:

        Returns:

        """
        pass

    def find_obstacle_normals_gpu(
        self,
        state
    ):
        """
        Compute obstacle normals on both fluid and solid boundary
        Args:

        Returns:

        """
        pass

    def set_backend(
        self,
        backend
    ):
        """
        Compute obstacle normals on both fluid and solid boundary
        Args:

        Returns:

        """
        if backend.backend_type == "cpu":
            self.find_obstacle_boundary_nodes =\
                self.find_obstacle_boundary_nodes_cpu
            self.find_obstacle_normals =\
                self.find_obstacle_normals_cpu
        elif backend.backend_type == "gpu":
            self.find_obstacle_boundary_nodes =\
                self.find_obstacle_boundary_nodes_gpu
            self.find_obstacle_normals =\
                self.find_obstacle_normals_gpu
