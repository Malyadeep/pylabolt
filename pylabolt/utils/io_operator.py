import os
import json
import numpy as np
from types import SimpleNamespace

from pylabolt.utils.helpers import print_log
import pylabolt.parallel.cpu.io_operator_kernels as io_operator_kernels_cpu
import pylabolt.parallel.gpu.io_operator_kernels as io_operator_kernels_gpu


class InputOutputOperator:
    def __init__(
        self,
        model,
        state,
        backend,
        comm,
        verbose=True
    ):
        """
        I/O operator
        Attributes:

        """
        try:
            print_log("-" * 80, state.domain.mpi_rank, verbose)
            print_log("Setting up I/O operator...\n",
                      state.domain.mpi_rank, verbose)
            self.model = model
            self.setup_write_fields(state)
            self.setup_write_histories(state)
            print_log("\nSetting up I/O operator done!",
                      state.domain.mpi_rank, verbose)
            print_log("-" * 80, state.domain.mpi_rank, verbose)
        except Exception as e:
            print_log("-" * 80, state.domain.mpi_rank, verbose=True)
            print_log("FATAL ERROR!", state.domain.mpi_rank, verbose=True)
            print_log(str(e), state.domain.mpi_rank, verbose=True)
            comm.Abort()

    def setup_write_fields(
        self,
        state,
        verbose=True
    ):
        """
        Setup write fields pipeline
        Args:

        Returns:

        """
        self.fields_list = self.model.save_fields
        self.fields_save = {}
        self.fields_save_metadata = {}
        for field_name in self.fields_list:
            if not hasattr(state.fields, field_name):
                raise ValueError(
                    field_name + " is not a valid field for saving"
                )
            field = getattr(state.fields, field_name)
            if len(field.shape) == 1:
                components = 1
                field_save = np.zeros(
                    state.domain.inner_size,
                    dtype=field.dtype
                )
            else:
                components = field.shape[1]
                field_save = np.zeros(
                    (state.domain.inner_size, components),
                    dtype=field.dtype
                )

            self.fields_save.update({field_name: field_save})
            self.fields_save_metadata.update({
                field_name: {
                    "components": components,
                    "dtype": str(field.dtype)
                }
            })
        if state.domain.mpi_size == 1:
            os.makedirs("output", exist_ok=True)
            os.makedirs("output/fields", exist_ok=True)
            self.field_save_path = "output/fields/"
        elif state.domain.mpi_size > 1:
            os.makedirs("procs", exist_ok=True)
            os.makedirs(
                "procs/proc_" + str(state.domain.mpi_rank),
                exist_ok=True
            )
            self.field_save_path = "procs/proc_" +\
                str(state.domain.mpi_rank) + "/"
        self.dump_metadata(state)

    def dump_metadata(
        self,
        state
    ):
        """
        Dump metadata.json for the simulation
        Args:

        Returns:

        """
        from importlib.metadata import version
        self.global_metadata = {
            "pylabolt": {
                "version": version("pylabolt"),
                "solver": self.model.solver_name
            },
            "control": {
                "end_time": state.control.end_time,
                "start_time": state.control.start_time,
                "save_interval": state.control.save_interval,
                "checkpoint_interval": state.control.checkpoint_interval,
            },
            "mesh": {
                "size": int(state.mesh.grid_global_size),
                "shape": (
                    int(state.mesh.grid_global_shape[0]),
                    int(state.mesh.grid_global_shape[1])
                )
            },
            "decomposition": {
                "nx": int(state.domain.no_of_procs_x),
                "ny": int(state.domain.no_of_procs_y),
            },
            "fields_saved": self.fields_save_metadata
        }
        if state.domain.mpi_rank == 0:
            with open("metadata.json", "w") as f:
                json.dump(self.global_metadata, f, indent=4)
        if state.domain.mpi_size > 1:
            self.rank_metadata = {
                "rank": int(state.domain.mpi_rank),
                "processor_ij": (
                    int(state.domain.i_proc),
                    int(state.domain.j_proc)
                ),
                "domain_size": int(state.domain.inner_size),
                "domain_shape": (
                    int(state.domain.inner_shape[0]),
                    int(state.domain.inner_shape[1])
                ),
                "offset": (
                    int(state.domain.offset[0]),
                    int(state.domain.offset[1])
                )
            }
            with open(
                "procs/proc_" + str(state.domain.mpi_rank) +
                "/rank_metadata.json", "w"
            ) as f:
                json.dump(self.rank_metadata, f, indent=4)

    def write_fields_cpu(
        self,
        state,
        backend,
        time_step
    ):
        """
        Fetch output data using CPU kernels and write to disk
        Args:

        Returns:

        """
        if state.control.save_interval is None:
            return
        if time_step % state.control.save_interval != 0:
            return
        for item in self.fields_save:
            args = (
                state.domain.inner_size,
                state.domain.inner_shape,
                state.domain.shape,
                getattr(state.fields, item),
                self.fields_save[item]
            )
            if self.fields_save_metadata[item]["components"] == 1:
                self.copy_inner_data_kernel_scalar(*args)
            elif self.fields_save_metadata[item]["components"] == 2:
                self.copy_inner_data_kernel_vector(*args)
        np.savez(
            self.field_save_path + "t_" + str(time_step) + ".npz",
            **self.fields_save
        )

    def write_fields_gpu(
        self,
        state,
        backend,
        time_step
    ):
        """
        Fetch output data from GPU and write to disk
        Args:

        Returns:

        """
        if state.control.save_interval is None:
            return
        if time_step % state.control.save_interval != 0:
            return
        for item in self.fields_save_device:
            args = (
                state.domain.inner_size_device,
                state.domain.inner_shape_device,
                state.domain.shape_device,
                getattr(state.fields, item + "_device"),
                self.fields_save_device[item]
            )
            if self.fields_save_metadata[item]["components"] == 1:
                self.copy_inner_data_kernel_scalar[
                    backend.blocks, backend.threads_per_block
                ](*args)
            elif self.fields_save_metadata[item]["components"] == 2:
                self.copy_inner_data_kernel_vector[
                    backend.blocks, backend.threads_per_block
                ](*args)
            self.fields_save[item] = self.fields_save_device[item].\
                copy_to_host()
        np.savez(
            self.field_save_path + "t_" + str(time_step) + ".npz",
            **self.fields_save
        )

    def setup_write_histories(
        self,
        state
    ):
        """
        Setup write histories pipeline
        Args:

        Returns:

        """
        if not (state.obstacle.write_obstacle_data or
                state.boundary.write_boundary_data):
            return
        os.makedirs("output/histories", exist_ok=True)
        self.history_save_path = "output/histories/"
        if state.obstacle.write_obstacle_data and state.domain.mpi_rank == 0:
            for obstacle in state.obstacle.obstacles:
                with open(
                    self.history_save_path + obstacle.name + ".dat", "w"
                ) as current_file:
                    current_file.write(
                        f"{'#':5} {'PyLaBolt obstacle history'}\n"
                        f"{'#':5} {'ID':8}: {obstacle.id}\n"
                        f"{'#':5} {'Name':8}: {obstacle.name}\n"
                        f"{'#':5} {'Type':8}: {obstacle.type}\n"
                        f"{'#':5} {'Columns':8}:\n"
                    )
                    current_file.write(
                        f"{'#':5}"
                        f"{'time':21}"
                        f"{'pos_x':24}"
                        f"{'pos_y':24}"
                        f"{'alpha':24}"
                        f"{'vel_x':24}"
                        f"{'vel_y':24}"
                        f"{'omega':24}"
                        f"{'force_x':24}"
                        f"{'force_y':24}"
                        f"{'torque':24}\n"
                    )
        if state.boundary.write_boundary_data and state.domain.mpi_rank == 0:
            for boundary_element in state.boundary.boundary_elements:
                if not boundary_element.wall:
                    continue
                with open(
                    self.history_save_path + boundary_element.name +
                    ".dat", "w"
                ) as current_file:
                    current_file.write(
                        f"{'#':5} {'PyLaBolt boundary history'}\n"
                        f"{'#':5} {'Name':8}: {boundary_element.name}\n"
                        f"{'#':5} {'Columns':8}:\n"
                    )
                    current_file.write(
                        f"{'#':5}"
                        f"{'time':21}"
                        f"{'force_x':24}"
                        f"{'force_y':24}\n"
                    )

    def write_histories_cpu(
        self,
        state,
        time_step
    ):
        """
        Write to disk obstacle and boundary data including
        force, torque, velocity, position for backend CPU
        Args:

        Returns:

        """
        if (state.obstacle.write_obstacle_data and
                state.domain.mpi_rank == 0 and
                state.obstacle.write_interval is not None):
            if time_step % state.obstacle.write_interval == 0:
                for itr, obstacle in enumerate(state.obstacle.obstacles):
                    obstacle.center[:] =\
                        state.obstacle.obstacle_data.center[itr]
                    obstacle.inclination_angle =\
                        state.obstacle.obstacle_data.inclination_angle[itr, 0]
                    obstacle.linear_velocity[:] =\
                        state.obstacle.obstacle_data.linear_velocity[itr]
                    obstacle.angular_velocity =\
                        state.obstacle.obstacle_data.angular_velocity[itr, 0]
                    obstacle.force[:] =\
                        state.obstacle.obstacle_data.force[itr]
                    obstacle.torque =\
                        state.obstacle.obstacle_data.torque[itr, 0]
                    with open(
                        self.history_save_path + obstacle.name + ".dat", "a"
                    ) as current_file:
                        current_file.write(
                            f"{time_step:<24}"
                            f"{obstacle.center[0]:24.16e}"
                            f"{obstacle.center[1]:24.16e}"
                            f"{obstacle.inclination_angle:24.16e}"
                            f"{obstacle.linear_velocity[0]:24.16e}"
                            f"{obstacle.linear_velocity[1]:24.16e}"
                            f"{obstacle.angular_velocity:24.16e}"
                            f"{obstacle.force[0]:24.16e}"
                            f"{obstacle.force[1]:24.16e}"
                            f"{obstacle.torque:24.16e}\n"
                        )
        if (state.boundary.write_boundary_data and
                state.domain.mpi_rank == 0 and
                state.boundary.write_interval is not None):
            if time_step % state.boundary.write_interval == 0:
                for boundary_element in state.boundary.boundary_elements:
                    if not boundary_element.wall:
                        continue
                    with open(
                        self.history_save_path + boundary_element.name +
                        ".dat", "a"
                    ) as current_file:
                        current_file.write(
                            f"{time_step:<24}"
                            f"{boundary_element.force[0]:24.16e}"
                            f"{boundary_element.force[1]:24.16e}\n"
                        )

    def write_histories_gpu(
        self,
        state,
        time_step
    ):
        """
        Write to disk obstacle and boundary data including
        force, torque, velocity, position for backend GPU
        Args:

        Returns:

        """
        if (state.obstacle.write_obstacle_data and
                state.domain.mpi_rank == 0 and
                state.obstacle.write_interval is not None):
            if time_step % state.obstacle.write_interval == 0:
                global_obstacle_data = SimpleNamespace(
                    force=state.obstacle.obstacle_data.
                    force_device.copy_to_host(),
                    torque=state.obstacle.obstacle_data.
                    torque_device.copy_to_host(),
                    linear_velocity=state.obstacle.obstacle_data.
                    linear_velocity_device.copy_to_host(),
                    angular_velocity=state.obstacle.obstacle_data.
                    angular_velocity_device.copy_to_host(),
                    center=state.obstacle.obstacle_data.
                    center_device.copy_to_host(),
                    inclination_angle=state.obstacle.obstacle_data.
                    inclination_angle_device.copy_to_host()
                )
                for itr, obstacle in enumerate(state.obstacle.obstacles):
                    obstacle.force[:] = global_obstacle_data.force[itr, :]
                    obstacle.torque = global_obstacle_data.torque[itr, 0]
                    obstacle.linear_velocity[:] =\
                        global_obstacle_data.linear_velocity[itr, :]
                    obstacle.angular_velocity =\
                        global_obstacle_data.angular_velocity[itr, 0]
                    obstacle.center[:] = global_obstacle_data.center[itr, :]
                    obstacle.inclination_angle =\
                        global_obstacle_data.inclination_angle[itr, 0]
                    with open(
                        self.history_save_path + obstacle.name + ".dat", "a"
                    ) as current_file:
                        current_file.write(
                            f"{time_step:<24}"
                            f"{obstacle.center[0]:24.16e}"
                            f"{obstacle.center[1]:24.16e}"
                            f"{obstacle.inclination_angle:24.16e}"
                            f"{obstacle.linear_velocity[0]:24.16e}"
                            f"{obstacle.linear_velocity[1]:24.16e}"
                            f"{obstacle.angular_velocity:24.16e}"
                            f"{obstacle.force[0]:24.16e}"
                            f"{obstacle.force[1]:24.16e}"
                            f"{obstacle.torque:24.16e}\n"
                        )
        if (state.boundary.write_boundary_data and
                state.domain.mpi_rank == 0 and
                state.boundary.write_interval is not None):
            if time_step % state.boundary.write_interval == 0:
                local_force = state.boundary.local_force_device.copy_to_host()
                for itr, boundary_element in enumerate(
                    state.boundary.boundary_elements
                ):
                    if not boundary_element.wall:
                        continue
                    boundary_element.force[0] = local_force[itr, 0]
                    boundary_element.force[1] = local_force[itr, 1]
                    with open(
                        self.history_save_path + boundary_element.name +
                        ".dat", "a"
                    ) as current_file:
                        current_file.write(
                            f"{time_step:<24}"
                            f"{boundary_element.force[0]:24.16e}"
                            f"{boundary_element.force[1]:24.16e}\n"
                        )

    def compile(
        self,
        state,
        backend,
        verbose=True
    ):
        """
        JIT compile I/O operator kernels
        Args:

        Returns:

        """
        if backend.backend_type == "cpu":
            for item in self.fields_save:
                args = (
                    state.domain.inner_size,
                    state.domain.inner_shape,
                    state.domain.shape,
                    getattr(state.fields, item),
                    self.fields_save[item]
                )
                compile_args = backend.make_compile_args(args)
                if self.fields_save_metadata[item]["components"] == 1:
                    self.copy_inner_data_kernel_scalar(*compile_args)
                elif self.fields_save_metadata[item]["components"] == 2:
                    self.copy_inner_data_kernel_vector(*compile_args)

        elif backend.backend_type == "gpu":
            for item in self.fields_save:
                args = (
                    state.domain.inner_size_device,
                    state.domain.inner_shape_device,
                    state.domain.shape_device,
                    getattr(state.fields, item + "_device"),
                    self.fields_save_device[item]
                )
                compile_args = backend.make_compile_args(args)
                if self.fields_save_metadata[item]["components"] == 1:
                    self.copy_inner_data_kernel_scalar[
                        backend.blocks, backend.threads_per_block
                    ](*compile_args)
                elif self.fields_save_metadata[item]["components"] == 2:
                    self.copy_inner_data_kernel_vector[
                        backend.blocks, backend.threads_per_block
                    ](*compile_args)

        self.kernel_signatures = {
            self.copy_inner_data_kernel_scalar.__name__:
                set(self.copy_inner_data_kernel_scalar.signatures),
            self.copy_inner_data_kernel_vector.__name__:
                set(self.copy_inner_data_kernel_vector.signatures)
        }

        print_log("Compiled I/O operator",
                  state.domain.mpi_rank, verbose)

    def set_backend(
        self,
        state,
        backend,
        verbose=True
    ):
        """
        Set backend for I/O operator
        Args:

        Returns:

        """
        if backend.backend_type == "cpu":
            self.write_fields = self.write_fields_cpu
            self.write_histories = self.write_histories_cpu
            self.copy_inner_data_kernel_scalar =\
                io_operator_kernels_cpu.copy_inner_data_scalar
            self.copy_inner_data_kernel_vector =\
                io_operator_kernels_cpu.copy_inner_data_vector
        elif backend.backend_type == "gpu":
            self.write_fields = self.write_fields_gpu
            self.write_histories = self.write_histories_gpu
            self.copy_inner_data_kernel_scalar =\
                io_operator_kernels_gpu.copy_inner_data_scalar
            self.copy_inner_data_kernel_vector =\
                io_operator_kernels_gpu.copy_inner_data_vector
            self._device_attrs = []
            self.fields_save_device = {}
            for field_name in self.fields_save:
                arg_device = backend.allocate_to_device(
                    self.fields_save[field_name]
                )
                self.fields_save_device.update({
                    field_name: arg_device
                })

        print_log("Backend set for I/O operator",
                  state.domain.mpi_rank, verbose)

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
            io_operator_kernels_module = io_operator_kernels_cpu
        elif backend.backend_type == "gpu":
            io_operator_kernels_module = io_operator_kernels_gpu
        for kernel_name in self.kernel_signatures:
            kernel = getattr(io_operator_kernels_module, kernel_name)
            if (set(kernel.signatures) !=
                    self.kernel_signatures[kernel_name]):
                raise RuntimeError(
                    f"Developer error! {kernel_name} in"
                    f" I/O operator compiled a new signature!"
                )

        print_log("Kernel signatures verified for I/O operator",
                  state.domain.mpi_rank, verbose)
