import numpy as np
from numba import cuda

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
import pylabolt.parallel.gpu.obstacle_kernels as obstacle_kernels_gpu
import pylabolt.parallel.gpu.force_torque_kernels as\
    force_torque_kernels_gpu


class ObstacleOperator:
    def __init__(
        self,
        model,
        state,
        backend,
        mpi_operator,
        force_operator,
        verbose=True
    ):
        """
        Obstacle operator - modifies obstacle and it's properties
        Attributes:

        """
        self.model = model
        self.force_operator = force_operator
        mpi_operator.halo_exchange_cpu(
            state,
            backend,
            bool_buffers=["solid"],
            int_buffers=["solid_id"]
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
        self.update_obstacle_properties(backend)

    def compute_force_torque_cpu(
        self,
        state,
        backend,
        mpi_operator
    ):
        """
        Compute force and torque acting on obstacles
        Backend: CPU
        Args:

        Returns:

        """
        if not state.obstacle.compute_force_torque:
            return
        for itr in range(state.obstacle.no_of_obstacles):
            current_obstacle = state.obstacle.obstacles[itr]
            self.local_force_torque[itr, :] =\
                self.compute_force_torque_kernel(
                    *self.compute_force_torque_args,
                    current_obstacle.ref_point,
                    current_obstacle.id
                )
        global_force_torque = mpi_operator.reduce(
            self.local_force_torque,
            operation="sum"
        )
        for itr in range(state.obstacle.no_of_obstacles):
            # current_obstacle = state.obstacle.obstacles[itr]
            # current_obstacle.force[0] = global_force_torque[itr, 0]
            # current_obstacle.force[1] = global_force_torque[itr, 1]
            # current_obstacle.torque = global_force_torque[itr, 2]
            state.obstacle.obstacle_data.force[itr, 0] =\
                global_force_torque[itr, 0]
            state.obstacle.obstacle_data.force[itr, 1] =\
                global_force_torque[itr, 1]
            state.obstacle.obstacle_data.torque[itr, 0] =\
                global_force_torque[itr, 2]

    def update_obstacle_properties_cpu(
        self,
        backend
    ):
        """
        Update obstacle position and velocities using
        Backend: CPU
        Args:

        Returns:

        """
        self.update_position_velocity_kernel(
            *self.update_position_velocity_args
        )

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
        if not state.obstacle.compute_force_torque:
            return
        for itr in range(state.obstacle.no_of_obstacles):
            self.compute_force_torque_kernel[
                backend.reduce_blocks,
                backend.reduce_threads_per_block,
                backend.numba_stream
            ](
                *self.compute_force_torque_args,
                state.obstacle.obstacle_data.ref_point_device,
                state.obstacle.obstacle_data.id_device,
                self.partial_force_torque_device,
                itr
            )
            partial_size = backend.reduce_blocks
            while partial_size > 1:
                blocks = int(np.ceil(
                    partial_size / backend.reduce_threads_per_block
                ))
                self.reduce_force_torque_kernel[
                    blocks,
                    backend.reduce_threads_per_block,
                    backend.numba_stream
                ](
                    partial_size,
                    self.partial_force_torque_device,
                    state.obstacle.obstacle_data.force_device,
                    state.obstacle.obstacle_data.torque_device,
                    itr
                )
                partial_size = blocks

        # global_force_torque = mpi_operator.reduce(
        #     self.local_force_torque,
        #     operation="sum"
        # )
        # for itr in range(state.obstacle.no_of_obstacles):
        #     obstacle = state.obstacle.obstacles[itr]
        #     obstacle.force[0] = global_force_torque[itr, 0]
        #     obstacle.force[1] = global_force_torque[itr, 1]
        #     obstacle.torque = global_force_torque[itr, 2]

    def update_obstacle_properties_gpu(
        self,
        backend
    ):
        """
        Update obstacle position and velocities using
        Backend: GPU
        Args:

        Returns:

        """
        self.update_position_velocity_kernel[
            1,
            backend.threads_per_block,
            backend.numba_stream
        ](
            *self.update_position_velocity_args
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

    def compile(
        self,
        state,
        backend,
        verbose=True
    ):
        """
        JIT compile obstacle operator kernels
        Args:

        Returns:

        """
        self.kernel_signatures = {}

        if state.obstacle.compute_force_torque:
            if backend.backend_type == "cpu":
                compile_args = backend.make_compile_args(
                    self.compute_force_torque_args
                )
                for itr in range(state.obstacle.no_of_obstacles):
                    current_obstacle = state.obstacle.obstacles[itr]
                    self.compute_force_torque_kernel(
                        *compile_args,
                        current_obstacle.ref_point,
                        current_obstacle.id
                    )
                    self.kernel_signatures.update({
                        self.compute_force_torque_kernel.__name__:
                            set(self.compute_force_torque_kernel.signatures)
                    })

                compile_args = backend.make_compile_args(
                    self.update_position_velocity_args
                )
                self.update_position_velocity_kernel(
                    *compile_args
                )
                self.kernel_signatures.update({
                    self.update_position_velocity_kernel.__name__:
                        set(self.update_position_velocity_kernel.signatures)
                })

            elif backend.backend_type == "gpu":
                for itr in range(state.obstacle.no_of_obstacles):
                    compile_args = backend.make_compile_args(
                        self.compute_force_torque_args
                    )

                    self.compute_force_torque_kernel[
                        backend.reduce_blocks,
                        backend.reduce_threads_per_block,
                        backend.numba_stream
                    ](
                        *compile_args,
                        cuda.device_array_like(
                            state.obstacle.obstacle_data.ref_point_device
                        ),
                        state.obstacle.obstacle_data.id_device,
                        self.partial_force_torque_device,
                        itr
                    )
                    self.reduce_force_torque_kernel[
                        backend.reduce_blocks,
                        backend.reduce_threads_per_block,
                        backend.numba_stream
                    ](
                        backend.reduce_blocks,
                        self.partial_force_torque_device,
                        cuda.device_array_like(
                            state.obstacle.obstacle_data.force_device
                        ),
                        cuda.device_array_like(
                            state.obstacle.obstacle_data.torque_device
                        ),
                        itr
                    )
                    self.kernel_signatures.update({
                        self.compute_force_torque_kernel.__name__:
                            set(self.compute_force_torque_kernel.signatures),
                        self.reduce_force_torque_kernel.__name__:
                            set(self.reduce_force_torque_kernel.signatures),
                    })

                compile_args = backend.make_compile_args(
                    self.update_position_velocity_args
                )
                self.update_position_velocity_kernel[
                    1,
                    backend.threads_per_block,
                    backend.numba_stream
                ](
                    *compile_args
                )
                self.kernel_signatures.update({
                    self.update_position_velocity_kernel.__name__:
                        set(self.update_position_velocity_kernel.signatures)
                })

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
            self.update_obstacle_properties =\
                self.update_obstacle_properties_cpu
            obstacle_kernels_module = obstacle_kernels_cpu
            force_torque_kernels_module = force_torque_kernels_cpu
            arg_suffix = ""
            if (state.obstacle.compute_force_torque or
                    (not state.obstacle.all_obstacles_static)):
                self.local_force_torque = np.zeros(
                    (state.obstacle.no_of_obstacles, 3),
                    dtype=state.control.precision
                )
        elif backend.backend_type == "gpu":
            self.find_obstacle_boundary_nodes =\
                self.find_obstacle_boundary_nodes_gpu
            self.find_obstacle_normals =\
                self.find_obstacle_normals_gpu
            self.compute_force_torque =\
                self.compute_force_torque_gpu
            self.update_obstacle_properties =\
                self.update_obstacle_properties_gpu
            obstacle_kernels_module = obstacle_kernels_gpu
            force_torque_kernels_module = force_torque_kernels_gpu
            arg_suffix = "_device"
            if (state.obstacle.compute_force_torque or
                    (not state.obstacle.all_obstacles_static)):
                self.partial_force_torque_device = cuda.device_array(
                        (backend.reduce_blocks, 3), dtype=float
                    )

        self.obstacle_kernels_type = self.model.obstacle_kernels_type

        self.compute_force_torque_kernel = getattr(
            force_torque_kernels_module,
            "compute_force_torque_" + self.obstacle_kernels_type
        )
        if backend.backend_type == "gpu":
            self.reduce_force_torque_kernel = getattr(
                force_torque_kernels_module,
                "reduce_force_torque"
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

        self.update_position_velocity_kernel =\
            obstacle_kernels_module.update_position_velocity
        obstacle_data = state.obstacle.obstacle_data
        self.update_position_velocity_args = (
            getattr(state.mesh, "grid_global_shape" + arg_suffix),
            getattr(state.boundary, "x_periodic" + arg_suffix),
            getattr(state.boundary, "y_periodic" + arg_suffix),
            getattr(self.force_operator, "gravity" + arg_suffix),
            getattr(obstacle_data, "N"),
            getattr(obstacle_data, "force" + arg_suffix),
            getattr(obstacle_data, "torque" + arg_suffix),
            getattr(obstacle_data, "linear_velocity" + arg_suffix),
            getattr(obstacle_data, "angular_velocity" + arg_suffix),
            getattr(obstacle_data, "center" + arg_suffix),
            getattr(obstacle_data, "inclination_angle" + arg_suffix),
            getattr(obstacle_data, "ref_point" + arg_suffix),
            getattr(obstacle_data, "mass" + arg_suffix),
            getattr(obstacle_data, "moment_of_inertia" + arg_suffix),
            getattr(obstacle_data, "static" + arg_suffix),
            getattr(obstacle_data, "calculated" + arg_suffix),
            getattr(obstacle_data, "rotation_allowed" + arg_suffix),
            getattr(obstacle_data, "translation_allowed" + arg_suffix)
        )

    def verify_kernel_signatures(
        self,
        state,
        backend,
        verbose=True
    ):
        """
        Debug function: Verifies if compiled kernel signatures
        changed or not. Detects recompilation
        Args:

        Returns:

        """
        if backend.backend_type == "cpu":
            obstacle_kernels_module = obstacle_kernels_cpu
            force_torque_kernels_module = force_torque_kernels_cpu
        if backend.backend_type == "gpu":
            obstacle_kernels_module = obstacle_kernels_gpu
            force_torque_kernels_module = force_torque_kernels_gpu

        for kernel_name in self.kernel_signatures:
            kernel = getattr(force_torque_kernels_module, kernel_name)
            if (set(kernel.signatures) !=
                    self.kernel_signatures[kernel_name]):
                raise RuntimeError(
                    f"Developer error! {kernel_name}: in"
                    f" obstacle operator compiled a new signature!"
                )

        print_log("Kernel signatures verified for obstacle operator",
                  state.domain.mpi_rank, verbose)
