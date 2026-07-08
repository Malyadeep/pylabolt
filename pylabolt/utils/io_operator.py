import os
import json
import numpy as np

from pylabolt.utils.helpers import print_log
import pylabolt.parallel.cpu.io_operator_kernels as io_operator_kernels_cpu


class InputOutputOperator:
    def __init__(
        self,
        comm,
        model,
        state,
        backend,
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
            self.setup_write_fields(model, state)
            self.set_backend(backend)
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
        model,
        state,
        verbose=True
    ):
        """
        Setup write fields pipeline
        Args:

        Returns:

        """
        self.fields_list = model.save_fields
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
            if not os.path.isdir("output"):
                os.makedirs("output")
            self.field_save_path = "output/"
        elif state.domain.mpi_size > 1:
            if not os.path.isdir("procs"):
                os.makedirs("procs")
            if not os.path.isdir(
                "procs/proc_" + str(state.domain.mpi_rank)
            ):
                os.makedirs("procs/proc_" + str(state.domain.mpi_rank))
            self.field_save_path = "procs/proc_" +\
                str(state.domain.mpi_rank) + "/"
        self.dump_metadata(state, model)

    def dump_metadata(
        self,
        state,
        model
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
                "solver": model.solver_name
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
        time_step
    ):
        """
        Fetch output data using CPU kernels and write to disk
        Args:

        Returns:

        """
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
        print_log("Compiled I/O operator",
                  state.domain.mpi_rank, verbose)

    def set_backend(
        self,
        backend
    ):
        """
        Set backend for residue operator
        Args:

        Returns:

        """
        if backend.backend_type == "cpu":
            self.write_fields = self.write_fields_cpu
            self.copy_inner_data_kernel_scalar =\
                io_operator_kernels_cpu.copy_inner_data_scalar
            self.copy_inner_data_kernel_vector =\
                io_operator_kernels_cpu.copy_inner_data_vector
        elif backend.backend_type == "gpu":
            pass
            # TODO: transfer residue fields to GPU
