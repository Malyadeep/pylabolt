import numpy as np
from numba import cuda

from pylabolt.utils.helpers import print_log
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
        force_operator,
        verbose=True
    ):
        """
        Obstacle operator - modifies obstacle and it's properties
        Attributes:

        """
        print_log("-" * 80, state.domain.mpi_rank, verbose)
        print_log("Setting up obstacle operator...",
                  state.domain.mpi_rank, verbose)
        self.model = model
        self.force_operator = force_operator
        print_log("Setting up obstacle operator done!",
                  state.domain.mpi_rank, verbose)
        print_log("-" * 80, state.domain.mpi_rank, verbose)

    def initialize_obstacles(
        self,
        state,
        backend,
        mpi_operator
    ):
        """
        Initialiizes obstacles on the grid.
        Computes fluid and solid boundaries.
        Computes surface normals
        Args:

        Returns:

        """
        mpi_operator.halo_exchange(
            state,
            backend,
            bool_buffers=["solid"],
            int_buffers=["solid_id"]
        )
        # ------- Find solid-fluid boundary nodes ------- #
        self.find_obstacle_boundary_nodes(state, backend, mpi_operator)
        # ------- Find solid-fluid normals ------- #
        self.find_obstacle_normals(state, backend, initialize=True)
        if backend.backend_type == "gpu":
            try:
                conflict_count =\
                    self.local_count_fluid_boundary_overlap_device.\
                    copy_to_host()
                if conflict_count[0] > 0:
                    raise RuntimeError
            except RuntimeError:
                print_log(
                    f"Fluid Boundary node overlap detected for"
                    f" {conflict_count[0]} nodes!",
                    state.domain.mpi_rank, verbose=True
                )
                print_log(
                    "This indicates two solid obstacles have" +
                    " a common fluid boundary node which is illegal!\n" +
                    "To avoid this issue, ensure solid particle surfaces " +
                    "have 2-3 lattice nodes in between.",
                    state.domain.mpi_rank, verbose=True
                )
                mpi_operator.comm.Abort()

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
        if state.obstacle.all_obstacles_static:
            return
        # ------- List of fields to be exchanged ------- #
        before_snapshot_fields = self._exchange_fields["before_snapshot"]
        after_geometry_fields = self._exchange_fields["after_geometry"]
        after_refill_fields = self._exchange_fields["after_refill"]
        # ------- Update solid position and velocity ------- #
        self.update_obstacle_properties(backend)
        # ------- Halo-exchange (before snapshot) ------- #
        mpi_operator.halo_exchange(
            state,
            backend,
            bool_buffers=before_snapshot_fields["bool_fields"],
            int_buffers=before_snapshot_fields["int_fields"],
            float_buffers=before_snapshot_fields["float_fields"]
        )
        # ------- Make copy of field data for interpolation ------- #
        self.snapshot_fields(backend)
        # ------- Reconstruct solid obstacles ------- #
        self.reconstruct_obstacles(
            state,
            backend
        )
        # ------- Halo-exchange (before snapshot) ------- #
        mpi_operator.halo_exchange(
            state,
            backend,
            bool_buffers=after_geometry_fields["bool_fields"],
            int_buffers=after_geometry_fields["int_fields"],
            float_buffers=after_geometry_fields["float_fields"]
        )
        # ------- Compute obstacle boundary nodes ------- #
        self.find_obstacle_boundary_nodes(
            state,
            backend,
            mpi_operator
        )
        # ------- Compute obstacle surface normals ------- #
        self.find_obstacle_normals(
            state,
            backend,
            initialize=False
        )
        # ------- Refill fresh nodes ------- #
        self.refill_nodes(
            state,
            backend
        )
        # ------- Halo-exchange (before snapshot) ------- #
        mpi_operator.halo_exchange(
            state,
            backend,
            bool_buffers=after_refill_fields["bool_fields"],
            int_buffers=after_refill_fields["int_fields"],
            float_buffers=after_refill_fields["float_fields"]
        )

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
        for obs_no, current_obstacle in enumerate(state.obstacle.obstacles):
            self.local_force_torque[obs_no, :] =\
                self.compute_force_torque_kernel(
                    *self.compute_force_torque_args,
                    state.obstacle.obstacle_data.ref_point[obs_no],
                    current_obstacle.id
                )
        global_force_torque = mpi_operator.reduce(
            self.local_force_torque,
            operation="sum"
        )
        for obs_no in range(state.obstacle.no_of_obstacles):
            # current_obstacle = state.obstacle.obstacles[itr]
            # current_obstacle.force[0] = global_force_torque[itr, 0]
            # current_obstacle.force[1] = global_force_torque[itr, 1]
            # current_obstacle.torque = global_force_torque[itr, 2]
            state.obstacle.obstacle_data.force[obs_no, 0] =\
                global_force_torque[obs_no, 0]
            state.obstacle.obstacle_data.force[obs_no, 1] =\
                global_force_torque[obs_no, 1]
            state.obstacle.obstacle_data.torque[obs_no, 0] =\
                global_force_torque[obs_no, 2]

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

    def snapshot_fields_cpu(
        self,
        backend
    ):
        """
        Takes a snapshot of the required fields into temporary buffer
        Backend: CPU
        Args:

        Returns:

        """
        self.snapshot_fields_kernel(
            *self.snapshot_fields_args
        )

    def reconstruct_obstacles_cpu(
        self,
        state,
        backend
    ):
        """
        Reconstruct moving obstacles
        Backend: CPU
        Args:

        Returns:

        """
        for obs_no, current_obstacle in enumerate(state.obstacle.obstacles):
            if current_obstacle.static:
                continue

            if current_obstacle.type == "circle":
                obstacle_kernels_cpu.construct_circle(
                    *self.reconstruct_obstacles_args[obs_no],
                    obs_no
                )
            elif current_obstacle.type == "ellipse":
                obstacle_kernels_cpu.construct_ellipse(
                    *self.reconstruct_obstacles_args[obs_no],
                    obs_no
                )

    def find_obstacle_boundary_nodes_cpu(
        self,
        state,
        backend,
        mpi_operator
    ):
        """
        Creates obstacle boundary nodes. Sets solid and fluid boundary
        Backend: CPU
        Args:

        Returns:

        """
        try:
            obstacle_kernels_cpu.compute_obstacle_boundary(
                *self.compute_obstacle_boundary_args
            )
            local_sum_overlap =\
                obstacle_kernels_cpu.check_fluid_boundary_overlap(
                    *self.check_fluid_boundary_overlap_args
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
        state,
        backend,
        initialize=False
    ):
        """
        Compute obstacle normals on both fluid and solid boundary
        Backend: CPU
        Args:

        Returns:

        """
        for obs_no, current_obstacle in enumerate(state.obstacle.obstacles):
            if not initialize and current_obstacle.static:
                continue

            if current_obstacle.type == "circle":
                obstacle_kernels_cpu.compute_normals_circle(
                    *self.compute_normals_args[obs_no],
                    obs_no
                )
            elif current_obstacle.type == "ellipse":
                obstacle_kernels_cpu.compute_normals_ellipse(
                    *self.compute_normals_args[obs_no],
                    obs_no
                )

    def refill_nodes_cpu(
        self,
        state,
        backend
    ):
        """
        Refill newly created fluid/solid nodes
        Backend: CPU
        Args:

        Returns:

        """
        self.refill_nodes_kernel(*self.refill_nodes_args)

    def compute_force_torque_gpu(
        self,
        state,
        backend,
        mpi_operator
    ):
        """
        Compute force and torque acting on obstacles
        Backend: GPU
        Args:

        Returns:

        """
        if not state.obstacle.compute_force_torque:
            return
        for obs_no, current_obstacle in enumerate(state.obstacle.obstacles):
            self.compute_force_torque_kernel[
                backend.reduce_blocks,
                backend.reduce_threads_per_block,
                backend.numba_stream
            ](
                *self.compute_force_torque_args,
                state.obstacle.obstacle_data.ref_point_device,
                current_obstacle.id_device,
                self.partial_force_torque_device,
                obs_no
            )
            partial_size = backend.reduce_blocks
            while True:
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
                    obs_no
                )

                if blocks == 1:
                    break

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

    def reconstruct_obstacles_gpu(
        self,
        state,
        backend
    ):
        """
        Reconstruct moving obstacles
        Backend: GPU
        Args:

        Returns:

        """
        for obs_no, current_obstacle in enumerate(state.obstacle.obstacles):
            if current_obstacle.static:
                continue

            if current_obstacle.type == "circle":
                obstacle_kernels_gpu.construct_circle[
                    backend.blocks,
                    backend.threads_per_block,
                    backend.numba_stream
                ](
                    *self.reconstruct_obstacles_args[obs_no],
                    obs_no
                )
            elif current_obstacle.type == "ellipse":
                obstacle_kernels_gpu.construct_ellipse[
                    backend.blocks,
                    backend.threads_per_block,
                    backend.numba_stream
                ](
                    *self.reconstruct_obstacles_args[obs_no],
                    obs_no
                )

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

    def snapshot_fields_gpu(
        self,
        backend
    ):
        """
        Takes a snapshot of the required fields into temporary buffec
        Backend: GPU
        Args:

        Returns:

        """
        self.snapshot_fields_kernel[
            backend.blocks,
            backend.threads_per_block,
            backend.numba_stream
        ](
            *self.snapshot_fields_args
        )

    def find_obstacle_boundary_nodes_gpu(
        self,
        state,
        backend,
        mpi_operator
    ):
        """
        Creates obstacle boundary nodes. Sets solid and fluid boundary
        Backend: GPU
        Args:

        Returns:

        """
        obstacle_kernels_gpu.compute_obstacle_boundary[
            backend.blocks,
            backend.threads_per_block,
            backend.numba_stream
        ](
            *self.compute_obstacle_boundary_args
        )
        obstacle_kernels_gpu.check_fluid_boundary_overlap[
            backend.reduce_blocks,
            backend.reduce_threads_per_block,
            backend.numba_stream
        ](
            *self.check_fluid_boundary_overlap_args,
            self.partial_fluid_boundary_overlap_device
        )
        partial_size = backend.reduce_blocks
        while True:
            blocks = int(np.ceil(
                partial_size / backend.reduce_threads_per_block
            ))
            obstacle_kernels_gpu.reduce_fluid_boundary_overlap[
                blocks,
                backend.reduce_threads_per_block,
                backend.numba_stream
            ](
                partial_size,
                self.partial_fluid_boundary_overlap_device,
                self.local_count_fluid_boundary_overlap_device
            )

            if blocks == 1:
                break

            partial_size = blocks

    def find_obstacle_normals_gpu(
        self,
        state,
        backend,
        initialize=False
    ):
        """
        Compute obstacle normals on both fluid and solid boundary
        Backend: GPU
        Args:

        Returns:

        """
        for obs_no, current_obstacle in enumerate(state.obstacle.obstacles):
            if not initialize and current_obstacle.static:
                continue

            if current_obstacle.type == "circle":
                obstacle_kernels_gpu.compute_normals_circle[
                    backend.blocks,
                    backend.threads_per_block,
                    backend.numba_stream
                ](
                    *self.compute_normals_args[obs_no],
                    obs_no
                )
            elif current_obstacle.type == "ellipse":
                obstacle_kernels_gpu.compute_normals_ellipse[
                    backend.blocks,
                    backend.threads_per_block,
                    backend.numba_stream
                ](
                    *self.compute_normals_args[obs_no],
                    obs_no
                )

    def refill_nodes_gpu(
        self,
        state,
        backend
    ):
        """
        Refill newly created fluid/solid nodes
        Backend: GPU
        Args:

        Returns:

        """
        self.refill_nodes_kernel[
            backend.blocks,
            backend.threads_per_block,
            backend.numba_stream
        ](*self.refill_nodes_args)

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

        # ------- Compile force and torque kernels ------- #
        if state.obstacle.compute_force_torque:
            self.kernel_signatures.update({"force_torque_kernels": {}})
            if backend.backend_type == "cpu":
                compile_args = backend.make_compile_args(
                    self.compute_force_torque_args
                )
                for itr in range(state.obstacle.no_of_obstacles):
                    current_obstacle = state.obstacle.obstacles[itr]
                    self.compute_force_torque_kernel(
                        *compile_args,
                        state.obstacle.obstacle_data.ref_point[itr],
                        current_obstacle.id
                    )
                    self.kernel_signatures["force_torque_kernels"].update({
                        self.compute_force_torque_kernel.__name__:
                            set(self.compute_force_torque_kernel.signatures)
                    })

            elif backend.backend_type == "gpu":
                for itr in range(state.obstacle.no_of_obstacles):
                    current_obstacle = state.obstacle.obstacles[itr]
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
                        current_obstacle.id_device,
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
                    self.kernel_signatures["force_torque_kernels"].update({
                        self.compute_force_torque_kernel.__name__:
                            set(self.compute_force_torque_kernel.signatures),
                        self.reduce_force_torque_kernel.__name__:
                            set(self.reduce_force_torque_kernel.signatures),
                    })

        if not state.obstacle.all_obstacles_static:
            self.kernel_signatures.update({"obstacle_kernels": {}})
            if backend.backend_type == "cpu":
                # ------- Compile position-velocity update kernel ------- #
                compile_args = backend.make_compile_args(
                    self.update_position_velocity_args
                )
                self.update_position_velocity_kernel(
                    *compile_args
                )
                self.kernel_signatures["obstacle_kernels"].update({
                    self.update_position_velocity_kernel.__name__:
                        set(self.update_position_velocity_kernel.signatures)
                })

                # ------- Snapshot fields kernel ------- #
                compile_args = backend.make_compile_args(
                    self.snapshot_fields_args
                )
                self.snapshot_fields_kernel(
                    *compile_args
                )
                self.kernel_signatures["obstacle_kernels"].update({
                    self.snapshot_fields_kernel.__name__:
                        set(self.snapshot_fields_kernel.signatures)
                })

                # ------- Compile obstacle reconstruction kernel ------- #
                for obs_no in range(state.obstacle.no_of_obstacles):
                    current_obstacle = state.obstacle.obstacles[obs_no]
                    if current_obstacle.static:
                        continue
                    compile_args = backend.make_compile_args(
                        self.reconstruct_obstacles_args[obs_no]
                    )
                    if current_obstacle.type == "circle":
                        obstacle_kernels_cpu.construct_circle(
                            *compile_args,
                            obs_no
                        )
                        self.kernel_signatures["obstacle_kernels"].update({
                            obstacle_kernels_cpu.construct_circle.__name__:
                                set(obstacle_kernels_cpu.construct_circle.
                                    signatures)
                        })
                    elif current_obstacle.type == "ellipse":
                        obstacle_kernels_cpu.construct_ellipse(
                            *compile_args,
                            obs_no
                        )
                        self.kernel_signatures["obstacle_kernels"].update({
                            obstacle_kernels_cpu.construct_ellipse.__name__:
                                set(obstacle_kernels_cpu.construct_ellipse.
                                    signatures)
                        })

                # ------- Compile obstacle boundary nodes kernel ------- #
                compile_args = backend.make_compile_args(
                    self.compute_obstacle_boundary_args
                )
                obstacle_kernels_cpu.compute_obstacle_boundary(
                    *compile_args
                )
                self.kernel_signatures["obstacle_kernels"].update({
                    obstacle_kernels_cpu.compute_obstacle_boundary.__name__:
                        set(obstacle_kernels_cpu.compute_obstacle_boundary.
                            signatures)
                })
                compile_args = backend.make_compile_args(
                    self.check_fluid_boundary_overlap_args
                )
                _ = obstacle_kernels_cpu.check_fluid_boundary_overlap(
                    *compile_args
                )
                self.kernel_signatures["obstacle_kernels"].update({
                    obstacle_kernels_cpu.check_fluid_boundary_overlap.__name__:
                        set(obstacle_kernels_cpu.check_fluid_boundary_overlap.
                            signatures)
                })

                # ------- Compile obstacle normals kernel ------- #
                for obs_no in range(state.obstacle.no_of_obstacles):
                    current_obstacle = state.obstacle.obstacles[obs_no]
                    compile_args = backend.make_compile_args(
                        self.compute_normals_args[obs_no]
                    )
                    if current_obstacle.type == "circle":
                        obstacle_kernels_cpu.compute_normals_circle(
                            *compile_args,
                            obs_no
                        )
                        self.kernel_signatures["obstacle_kernels"].update({
                            obstacle_kernels_cpu.compute_normals_circle
                            .__name__: set(
                                obstacle_kernels_cpu.compute_normals_circle.
                                signatures
                            )
                        })
                    elif current_obstacle.type == "ellipse":
                        obstacle_kernels_cpu.compute_normals_ellipse(
                            *compile_args,
                            obs_no
                        )
                        self.kernel_signatures["obstacle_kernels"].update({
                            obstacle_kernels_cpu.compute_normals_ellipse
                            .__name__: set(
                                obstacle_kernels_cpu.compute_normals_ellipse.
                                signatures
                            )
                        })

                # ------- Refill nodes kernel ------- #
                compile_args = backend.make_compile_args(
                    self.refill_nodes_args
                )
                self.refill_nodes_kernel(
                    *compile_args
                )
                self.kernel_signatures["obstacle_kernels"].update({
                    self.refill_nodes_kernel.__name__:
                        set(self.refill_nodes_kernel.signatures)
                })

            elif backend.backend_type == "gpu":
                # ------- Compile position-velocity update kernel ------- #
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
                self.kernel_signatures["obstacle_kernels"].update({
                    self.update_position_velocity_kernel.__name__:
                        set(self.update_position_velocity_kernel.signatures)
                })

                # ------- Snapshot fields kernel ------- #
                compile_args = backend.make_compile_args(
                    self.snapshot_fields_args
                )
                self.snapshot_fields_kernel[
                    backend.blocks,
                    backend.threads_per_block,
                    backend.numba_stream
                ](
                    *compile_args
                )
                self.kernel_signatures["obstacle_kernels"].update({
                    self.snapshot_fields_kernel.__name__:
                        set(self.snapshot_fields_kernel.signatures)
                })

                # ------- Compile obstacle reconstruction kernel ------- #
                for obs_no in range(state.obstacle.no_of_obstacles):
                    current_obstacle = state.obstacle.obstacles[obs_no]
                    if current_obstacle.static:
                        continue
                    compile_args = backend.make_compile_args(
                        self.reconstruct_obstacles_args[obs_no]
                    )
                    if current_obstacle.type == "circle":
                        obstacle_kernels_gpu.construct_circle[
                            backend.blocks,
                            backend.threads_per_block,
                            backend.numba_stream
                        ](
                            *compile_args,
                            obs_no
                        )
                        self.kernel_signatures["obstacle_kernels"].update({
                            obstacle_kernels_gpu.construct_circle.__name__:
                                set(obstacle_kernels_gpu.construct_circle.
                                    signatures)
                        })
                    elif current_obstacle.type == "ellipse":
                        obstacle_kernels_gpu.construct_ellipse[
                            backend.blocks,
                            backend.threads_per_block,
                            backend.numba_stream
                        ](
                            *compile_args,
                            obs_no
                        )
                        self.kernel_signatures["obstacle_kernels"].update({
                            obstacle_kernels_gpu.construct_ellipse.__name__:
                                set(obstacle_kernels_gpu.construct_ellipse.
                                    signatures)
                        })

                # ------- Compile obstacle boundary nodes kernel ------- #
                compile_args = backend.make_compile_args(
                    self.compute_obstacle_boundary_args
                )
                obstacle_kernels_gpu.compute_obstacle_boundary[
                    backend.blocks,
                    backend.threads_per_block,
                    backend.numba_stream
                ](
                    *compile_args
                )
                self.kernel_signatures["obstacle_kernels"].update({
                    obstacle_kernels_gpu.compute_obstacle_boundary.__name__:
                        set(obstacle_kernels_gpu.compute_obstacle_boundary.
                            signatures)
                })
                compile_args = backend.make_compile_args(
                    self.check_fluid_boundary_overlap_args
                )
                obstacle_kernels_gpu.check_fluid_boundary_overlap[
                    backend.reduce_blocks,
                    backend.threads_per_block,
                    backend.numba_stream
                ](
                    *compile_args,
                    self.partial_fluid_boundary_overlap_device
                )
                self.kernel_signatures["obstacle_kernels"].update({
                    obstacle_kernels_gpu.check_fluid_boundary_overlap.__name__:
                        set(obstacle_kernels_gpu.check_fluid_boundary_overlap.
                            signatures)
                })
                obstacle_kernels_gpu.reduce_fluid_boundary_overlap[
                    backend.reduce_blocks,
                    backend.reduce_threads_per_block,
                    backend.numba_stream
                ](
                    backend.reduce_blocks,
                    self.partial_fluid_boundary_overlap_device,
                    cuda.device_array_like(
                        self.local_count_fluid_boundary_overlap_device
                    )
                )
                self.kernel_signatures["obstacle_kernels"].update({
                    obstacle_kernels_gpu.reduce_fluid_boundary_overlap.
                    __name__: set(
                        obstacle_kernels_gpu.reduce_fluid_boundary_overlap.
                        signatures
                    )
                })

                # ------- Compile obstacle normals kernel ------- #
                for obs_no in range(state.obstacle.no_of_obstacles):
                    current_obstacle = state.obstacle.obstacles[obs_no]
                    compile_args = backend.make_compile_args(
                        self.compute_normals_args[obs_no]
                    )
                    if current_obstacle.type == "circle":
                        obstacle_kernels_gpu.compute_normals_circle[
                            backend.blocks,
                            backend.threads_per_block,
                            backend.numba_stream
                        ](
                            *compile_args,
                            obs_no
                        )
                        self.kernel_signatures["obstacle_kernels"].update({
                            obstacle_kernels_gpu.compute_normals_circle
                            .__name__: set(
                                obstacle_kernels_gpu.compute_normals_circle.
                                signatures
                            )
                        })
                    elif current_obstacle.type == "ellipse":
                        obstacle_kernels_gpu.compute_normals_ellipse[
                            backend.blocks,
                            backend.threads_per_block,
                            backend.numba_stream
                        ](
                            *compile_args,
                            obs_no
                        )
                        self.kernel_signatures["obstacle_kernels"].update({
                            obstacle_kernels_gpu.compute_normals_ellipse
                            .__name__: set(
                                obstacle_kernels_gpu.compute_normals_ellipse.
                                signatures
                            )
                        })

                # ------- Refill nodes kernel ------- #
                compile_args = backend.make_compile_args(
                    self.refill_nodes_args
                )
                self.refill_nodes_kernel[
                    backend.blocks,
                    backend.threads_per_block,
                    backend.numba_stream
                ](
                    *compile_args
                )
                self.kernel_signatures["obstacle_kernels"].update({
                    self.refill_nodes_kernel.__name__:
                        set(self.refill_nodes_kernel.signatures)
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
            self.snapshot_fields =\
                self.snapshot_fields_cpu
            self.update_obstacle_properties =\
                self.update_obstacle_properties_cpu
            self.reconstruct_obstacles =\
                self.reconstruct_obstacles_cpu
            self.find_obstacle_boundary_nodes =\
                self.find_obstacle_boundary_nodes_cpu
            self.find_obstacle_normals =\
                self.find_obstacle_normals_cpu
            self.refill_nodes =\
                self.refill_nodes_cpu
            self.compute_force_torque =\
                self.compute_force_torque_cpu
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
            self.update_obstacle_properties =\
                self.update_obstacle_properties_gpu
            self.snapshot_fields =\
                self.snapshot_fields_gpu
            self.reconstruct_obstacles =\
                self.reconstruct_obstacles_gpu
            self.find_obstacle_boundary_nodes =\
                self.find_obstacle_boundary_nodes_gpu
            self.find_obstacle_normals =\
                self.find_obstacle_normals_gpu
            self.refill_nodes =\
                self.refill_nodes_gpu
            self.compute_force_torque =\
                self.compute_force_torque_gpu
            obstacle_kernels_module = obstacle_kernels_gpu
            force_torque_kernels_module = force_torque_kernels_gpu
            arg_suffix = "_device"
            if (state.obstacle.compute_force_torque or
                    (not state.obstacle.all_obstacles_static)):
                self.partial_force_torque_device = cuda.device_array(
                        (backend.reduce_blocks, 3), dtype=float
                    )
            if not state.obstacle.all_obstacles_static:
                self.partial_fluid_boundary_overlap_device = cuda.device_array(
                    backend.reduce_blocks, dtype=int
                )
                self.local_count_fluid_boundary_overlap_device =\
                    cuda.device_array(1, dtype=int)

        self.obstacle_kernels_type = self.model.obstacle_kernels_type

        if state.obstacle.compute_force_torque:
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

        if not state.obstacle.all_obstacles_static:
            self.snapshot_fields_list = [
                "solid",
                "solid_id",
                "solid_boundary",
                "fluid_boundary"
            ]
            if self.obstacle_kernels_type == "single_phase":
                self._exchange_fields = {
                    "before_snapshot": {
                        "bool_fields": None,
                        "int_fields": None,
                        "float_fields": ["density", "pop_fluid_new"]
                    },
                    "after_geometry": {
                        "bool_fields": ["solid"],
                        "int_fields": ["solid_id"],
                        "float_fields": None
                    },
                    "after_refill": {
                        "bool_fields": ["solid_boundary", "fluid_boundary"],
                        "int_fields": ["solid_id"],
                        "float_fields": ["velocity"]
                    }
                }

                self.snapshot_fields_kernel =\
                    obstacle_kernels_module.snapshot_single_phase
                self.snapshot_fields_list.extend([
                    "density",
                    "pop_fluid_new"
                ])
                self.snapshot_fields_dict = {}
                for field_name in self.snapshot_fields_list:
                    if backend.backend_type == "cpu":
                        self.snapshot_fields_dict.update({
                            field_name: np.zeros_like(
                                getattr(state.fields, field_name)
                            )
                        })
                    elif backend.backend_type == "gpu":
                        self.snapshot_fields_dict.update({
                            field_name: cuda.device_array_like(
                                getattr(state.fields, field_name + "_device")
                            )
                        })
                self.snapshot_fields_args = (
                    state.domain.size,
                    state.lattice.no_of_directions,
                    getattr(state.fields, "surface_normals" + arg_suffix)
                )
                for field_name in self.snapshot_fields_list:
                    self.snapshot_fields_args += tuple(
                        [getattr(state.fields, field_name + arg_suffix)]
                    )
                for field_name in self.snapshot_fields_dict:
                    self.snapshot_fields_args += tuple(
                        [self.snapshot_fields_dict[field_name]]
                    )

                self.refill_nodes_kernel =\
                    obstacle_kernels_module.refill_nodes_single_phase
                self.refill_nodes_args = (
                    getattr(state.control, "float_min" + arg_suffix),
                    getattr(state.domain, "size" + arg_suffix),
                    getattr(state.domain, "shape" + arg_suffix),
                    getattr(state.lattice, "cx" + arg_suffix),
                    getattr(state.lattice, "cy" + arg_suffix),
                    getattr(state.lattice, "weights" + arg_suffix),
                    getattr(state.lattice, "no_of_directions" + arg_suffix),
                    getattr(state.lattice, "inv_cs_2" + arg_suffix),
                    getattr(state.lattice, "inv_cs_4" + arg_suffix),
                    getattr(state.fields, "ghost_node" + arg_suffix),
                    getattr(state.fields, "velocity" + arg_suffix),
                    getattr(state.fields, "solid" + arg_suffix),
                    getattr(state.fields, "density" + arg_suffix),
                    getattr(state.fields, "pop_fluid_new" + arg_suffix),
                    self.snapshot_fields_dict["solid"],
                    self.snapshot_fields_dict["density"],
                    self.snapshot_fields_dict["pop_fluid_new"],
                )

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

            self.reconstruct_obstacles_args = []
            for _, current_obstacle in enumerate(
                state.obstacle.obstacles
            ):
                args = ()
                if (current_obstacle.type == "circle" and
                        not current_obstacle.static):
                    args = (
                        getattr(state.domain, "size" + arg_suffix),
                        getattr(state.domain, "shape" + arg_suffix),
                        getattr(state.domain, "offset" + arg_suffix),
                        getattr(state.mesh, "grid_global_shape" + arg_suffix),
                        getattr(state.boundary, "x_periodic" + arg_suffix),
                        getattr(state.boundary, "y_periodic" + arg_suffix),
                        getattr(state.fields, "solid" + arg_suffix),
                        getattr(state.fields, "solid_id" + arg_suffix),
                        getattr(state.fields, "ghost_node" + arg_suffix),
                        getattr(state.fields, "density" + arg_suffix),
                        getattr(state.fields, "velocity" + arg_suffix),
                        getattr(obstacle_data, "linear_velocity" + arg_suffix),
                        getattr(obstacle_data, "angular_velocity" +
                                arg_suffix),
                        getattr(obstacle_data, "solid_density" + arg_suffix),
                        getattr(obstacle_data, "center" + arg_suffix),
                        getattr(current_obstacle, "radius" + arg_suffix),
                        getattr(current_obstacle, "id" + arg_suffix)
                    )
                elif (current_obstacle.type == "ellipse" and
                        not current_obstacle.static):
                    args = (
                        getattr(state.domain, "size" + arg_suffix),
                        getattr(state.domain, "shape" + arg_suffix),
                        getattr(state.domain, "offset" + arg_suffix),
                        getattr(state.mesh, "grid_global_shape" + arg_suffix),
                        getattr(state.boundary, "x_periodic" + arg_suffix),
                        getattr(state.boundary, "y_periodic" + arg_suffix),
                        getattr(state.fields, "solid" + arg_suffix),
                        getattr(state.fields, "solid_id" + arg_suffix),
                        getattr(state.fields, "ghost_node" + arg_suffix),
                        getattr(state.fields, "density" + arg_suffix),
                        getattr(state.fields, "velocity" + arg_suffix),
                        getattr(obstacle_data, "linear_velocity" + arg_suffix),
                        getattr(obstacle_data, "angular_velocity" +
                                arg_suffix),
                        getattr(obstacle_data, "solid_density" + arg_suffix),
                        getattr(obstacle_data, "center" + arg_suffix),
                        getattr(current_obstacle, "semi_major_axis" +
                                arg_suffix),
                        getattr(current_obstacle, "semi_minor_axis" +
                                arg_suffix),
                        getattr(obstacle_data, "inclination_angle" +
                                arg_suffix),
                        getattr(current_obstacle, "id" + arg_suffix)
                    )
                self.reconstruct_obstacles_args.append(args)

            self.compute_obstacle_boundary_args = (
                getattr(state.domain, "size" + arg_suffix),
                getattr(state.domain, "shape" + arg_suffix),
                getattr(state.lattice, "cx" + arg_suffix),
                getattr(state.lattice, "cy" + arg_suffix),
                getattr(state.lattice, "no_of_directions" + arg_suffix),
                getattr(state.fields, "solid" + arg_suffix),
                getattr(state.fields, "solid_id" + arg_suffix),
                getattr(state.fields, "solid_boundary" + arg_suffix),
                getattr(state.fields, "fluid_boundary" + arg_suffix),
                getattr(state.fields, "ghost_node" + arg_suffix)
            )
            self.check_fluid_boundary_overlap_args = (
                getattr(state.domain, "size" + arg_suffix),
                getattr(state.domain, "shape" + arg_suffix),
                getattr(state.lattice, "cx" + arg_suffix),
                getattr(state.lattice, "cy" + arg_suffix),
                getattr(state.lattice, "no_of_directions" + arg_suffix),
                getattr(state.fields, "solid_id" + arg_suffix),
                getattr(state.fields, "fluid_boundary" + arg_suffix),
                getattr(state.fields, "ghost_node" + arg_suffix)
            )

            self.compute_normals_args = []
            for _, current_obstacle in enumerate(
                state.obstacle.obstacles
            ):
                args = ()
                if current_obstacle.type == "circle":
                    args = (
                        getattr(state.domain, "size" + arg_suffix),
                        getattr(state.domain, "shape" + arg_suffix),
                        getattr(state.domain, "offset" + arg_suffix),
                        getattr(state.mesh, "grid_global_shape" + arg_suffix),
                        getattr(state.boundary, "x_periodic" + arg_suffix),
                        getattr(state.boundary, "y_periodic" + arg_suffix),
                        getattr(state.fields, "solid_boundary" + arg_suffix),
                        getattr(state.fields, "fluid_boundary" + arg_suffix),
                        getattr(state.fields, "solid_id" + arg_suffix),
                        getattr(state.fields, "surface_normals" + arg_suffix),
                        getattr(state.obstacle.obstacle_data, "center" +
                                arg_suffix),
                        getattr(current_obstacle, "id" + arg_suffix)
                    )
                elif current_obstacle.type == "ellipse":
                    args = (
                        getattr(state.domain, "size" + arg_suffix),
                        getattr(state.domain, "shape" + arg_suffix),
                        getattr(state.domain, "offset" + arg_suffix),
                        getattr(state.mesh, "grid_global_shape" + arg_suffix),
                        getattr(state.boundary, "x_periodic" + arg_suffix),
                        getattr(state.boundary, "y_periodic" + arg_suffix),
                        getattr(state.fields, "solid_boundary" + arg_suffix),
                        getattr(state.fields, "fluid_boundary" + arg_suffix),
                        getattr(state.fields, "solid_id" + arg_suffix),
                        getattr(state.fields, "surface_normals" + arg_suffix),
                        getattr(state.obstacle.obstacle_data, "center" +
                                arg_suffix),
                        getattr(current_obstacle, "semi_major_axis" +
                                arg_suffix),
                        getattr(current_obstacle, "semi_minor_axis" +
                                arg_suffix),
                        getattr(state.obstacle.obstacle_data,
                                "inclination_angle" + arg_suffix),
                        getattr(current_obstacle, "id" + arg_suffix)
                    )
                self.compute_normals_args.append(args)

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

        if state.obstacle.compute_force_torque:
            for kernel_name in self.kernel_signatures["force_torque_kernels"]:
                kernel = getattr(force_torque_kernels_module, kernel_name)
                if (set(kernel.signatures) !=
                        self.kernel_signatures["force_torque_kernels"]
                        [kernel_name]):
                    raise RuntimeError(
                        f"Developer error! {kernel_name}: in"
                        f" obstacle operator compiled a new signature!"
                    )

        if not state.obstacle.all_obstacles_static:
            for kernel_name in self.kernel_signatures["obstacle_kernels"]:
                kernel = getattr(obstacle_kernels_module, kernel_name)
                if (set(kernel.signatures) !=
                        self.kernel_signatures["obstacle_kernels"]
                        [kernel_name]):
                    raise RuntimeError(
                        f"Developer error! {kernel_name}: in"
                        f" obstacle operator compiled a new signature!"
                    )

        print_log("Kernel signatures verified for obstacle operator",
                  state.domain.mpi_rank, verbose)
