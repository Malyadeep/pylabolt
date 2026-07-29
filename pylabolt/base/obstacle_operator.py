import numpy as np

from pylabolt.utils.helpers import print_log
from pylabolt.parallel.cpu.obstacle_kernels import (
    compute_obstacle_boundary,
    check_fluid_boundary_overlap,
    compute_normals_circle,
    compute_normals_ellipse
)
import pylabolt.parallel.cpu.obstacle_kernels as obstacle_kernels_cpu
import pylabolt.parallel.cpu.force_torque_kernels as\
    force_torque_kernels_cpu


class ObstacleOperator:
    def __init__(
        self,
        model,
        state,
        backend,
        mpi_operator,
        verbose=True
    ):
        """
        Obstacle operator - modifies obstacle and it's properties
        Attributes:

        """
        self.model = model
        self.no_of_obstacles = len(state.obstacle.obstacles)
        self.all_obstacles_static = True
        for obstacle in state.obstacle.obstacles:
            if not obstacle.static:
                self.all_obstacles_static = False
                break
        if state.obstacle.compute_force_torque:
            self.local_force_torque = np.zeros(
                (self.no_of_obstacles, 3),
                dtype=state.control.precision
            )
        mpi_operator.halo_exchange_cpu(
            state,
            backend,
            bool_buffers=["solid"]
        )
        # ------- Find solid-fluid boundary nodes ------- #
        self.find_obstacle_boundary_nodes_cpu(state, mpi_operator)
        # ------- Find solid-fluid normals ------- #
        self.find_obstacle_normals_cpu(state)

    def move_obstacles(
        self,
        state,
        backend,
        mpi_operator
    ):
        """
        Modify obstacles in the grid
        Compute force, update solid position-velocities,
        reconstruct solid bodies, recompute solid-fluid properties
        Args:

        Returns:

        """
        pass

    def compute_force_torque_cpu(
        self,
        state,
        backend,
        mpi_operator
    ):
        """
        Compute force and torque acting on obstacles
        Args:

        Returns:

        """
        if not state.obstacle.compute_force_torque:
            return
        for itr in range(self.no_of_obstacles):
            obstacle = state.obstacle.obstacles[itr]
            self.local_force_torque[itr, :] =\
                self.compute_force_torque_kernel(
                    *self.compute_force_torque_args,
                    obstacle.ref_point,
                    obstacle.id
                )
        global_force_torque = mpi_operator.reduce(
            self.local_force_torque,
            operation="sum"
        )
        for itr in range(self.no_of_obstacles):
            obstacle = state.obstacle.obstacles[itr]
            obstacle.force[0] = global_force_torque[itr, 0]
            obstacle.force[1] = global_force_torque[itr, 1]
            obstacle.torque = global_force_torque[itr, 2]

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

    def compute_force_torque_gpu(
        self,
        state,
        backend,
        mpi_operator
    ):
        """
        Compute force and torque acting on obstacles
        Args:

        Returns:

        """
        pass

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
        state,
        backend
    ):
        """
        Set backend for obstacle operator
        Args:

        Returns:

        """
        if backend.backend_type == "cpu":
            self.find_obstacle_boundary_nodes =\
                self.find_obstacle_boundary_nodes_cpu
            self.find_obstacle_normals =\
                self.find_obstacle_normals_cpu
            self.compute_force_torque =\
                self.compute_force_torque_cpu
            obstacle_kernels_module = obstacle_kernels_cpu
            force_torque_kernels_module = force_torque_kernels_cpu
            arg_suffix = ""
        elif backend.backend_type == "gpu":
            self.find_obstacle_boundary_nodes =\
                self.find_obstacle_boundary_nodes_gpu
            self.find_obstacle_normals =\
                self.find_obstacle_normals_gpu
            self.compute_force_torque =\
                self.compute_force_torque_gpu
            # obstacle_kernels_module = obstacle_kernels_gpu
            # force_torque_kernels_module = force_torque_kernels_gpu
            arg_suffix = "_device"

        self.obstacle_kernels_type = self.model.obstacle_kernels_type

        self.compute_force_torque_kernel = getattr(
            force_torque_kernels_module,
            "compute_force_torque_" + self.obstacle_kernels_type
        )
        if self.obstacle_kernels_type == "single_phase":
            args_dict = {
                "domain": ["size", "shape", "offset"],
                "mesh": ["grid_global_shape"],
                "lattice": ["cx", "cy", "inv_list", "no_of_directions"],
                "boundary": ["x_periodic", "y_periodic"],
                "fields": ["solid", "solid_id", "fluid_boundary",
                           "ghost_node", "pop_fluid", "pop_fluid_new"]
            }
        self.compute_force_torque_args = ()
        for arg_item in args_dict:
            args_list = args_dict[arg_item]
            arg_obj = getattr(state, arg_item)
            for arg_name in args_list:
                arg = getattr(arg_obj, arg_name + arg_suffix)
                self.compute_force_torque_args += tuple([arg])
