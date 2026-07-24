import numpy as np

from pylabolt.utils.helpers import print_log
from pylabolt.parallel.cpu.obstacle_kernels import (
    compute_obstacle_boundary,
    check_fluid_boundary_overlap,
    compute_normals_circle,
    compute_normals_ellipse
)
import pylabolt.parallel.cpu.obstacle_kernels as obstacle_kernels_cpu


class ObstacleOperator:
    def __init__(
        self,
        state,
        backend,
        mpi_operator,
        verbose=True
    ):
        """
        Obstacle operator - modifies obstacle and it's properties
        Attributes:

        """
        self.no_of_obstacles = len(state.obstacle.obstacles)
        self.all_obstacles_static = True
        for obstacle in state.obstacle.obstacles:
            if not obstacle.static:
                self.all_obstacles_static = False
                break
        mpi_operator.halo_exchange_cpu(
            state,
            backend,
            bool_buffers=["solid"]
        )
        # ------- Find solid-fluid boundary nodes ------- #
        self.find_obstacle_boundary_nodes_cpu(state, mpi_operator)
        # ------- Find solid-fluid normals ------- #
        self.find_obstacle_normals_cpu(state)

    def compute_forces_cpu(
        self,
        state,
        mpi_operator
    ):
        """
        Compute forces and torque acting on obstacles
        Args:

        Returns:

        """
        for obstacle in state.obstacle.obstacles:
            if obstacle.type in ["circle", "ellipse"]:
                ref_point = obstacle.center
            else:
                ref_point = state.obstacle.ref_point_torque
            local_force_torque = obstacle_kernels_cpu.compute_forces_torque(
                state.domain.size,
                state.domain.shape,
                state.domain.offset,
                state.mesh.grid_global_shape,
                state.lattice.cx,
                state.lattice.cy,
                state.lattice.inv_list,
                state.lattice.no_of_directions,
                state.boundary.x_periodic,
                state.boundary.y_periodic,
                state.fields.solid,
                state.fields.solid_id,
                state.fields.fluid_boundary,
                state.fields.ghost_node,
                state.fields.pop_fluid,
                state.fields.pop_fluid_new,
                ref_point,
                obstacle.id
            )
            global_force_torque = mpi_operator.reduce(
                local_force_torque,
                operation="sum"
            )
            obstacle.forces[0] = global_force_torque[0]
            obstacle.forces[1] = global_force_torque[1]
            obstacle.torque = global_force_torque[2]

    def find_obstacle_boundary_nodes_cpu(
        self,
        state,
        mpi_operator
    ):
        """
        Creates obstacle boundary nodes. Sets solid and fluid boundary
        Args:

        Returns:

        """
        try:
            compute_obstacle_boundary(
                state.domain.size,
                state.domain.shape,
                state.lattice.cx,
                state.lattice.cy,
                state.lattice.no_of_directions,
                state.fields.solid,
                state.fields.solid_id,
                state.fields.solid_boundary,
                state.fields.fluid_boundary,
                state.fields.ghost_node
            )
            local_sum_overlap = check_fluid_boundary_overlap(
                state.domain.size,
                state.domain.shape,
                state.lattice.cx,
                state.lattice.cy,
                state.lattice.no_of_directions,
                state.fields.solid_id,
                state.fields.fluid_boundary,
                state.fields.ghost_node
            )
            global_sum_overlap = mpi_operator.reduce(
                np.array([local_sum_overlap]),
                operation="sum"
            )
            if global_sum_overlap > 0:
                raise RuntimeError
        except RuntimeError:
            print_log(
                f"Fluid Boundary node overlap detected for"
                f" {global_sum_overlap[0]} nodes!",
                state.domain.mpi_rank, verbose=True
            )
            print_log(
                "This indicates two solid obstacles have" +
                " a common fluid boundary node which is illegal!\n" +
                "To avoid this issue, ensure solid particle surfaces have " +
                "2-3 lattice nodes in between.",
                state.domain.mpi_rank, verbose=True
            )
            mpi_operator.comm.Abort()

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
            if obstacle.type == "circle":
                compute_normals_circle(
                    state.domain.size,
                    state.domain.shape,
                    state.domain.offset,
                    state.fields.solid_boundary,
                    state.fields.fluid_boundary,
                    state.fields.solid_id,
                    state.fields.surface_normals,
                    obstacle.center,
                    obstacle.id
                )
            elif obstacle.type == "ellipse":
                compute_normals_ellipse(
                    state.domain.size,
                    state.domain.shape,
                    state.domain.offset,
                    state.fields.solid_boundary,
                    state.fields.fluid_boundary,
                    state.fields.solid_id,
                    state.fields.surface_normals,
                    obstacle.center,
                    obstacle.semi_major_axis,
                    obstacle.semi_minor_axis,
                    obstacle.inclination_angle,
                    obstacle.id
                )

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
