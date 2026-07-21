import numpy as np
from numba import cuda

from pylabolt.utils.helpers import print_log
import pylabolt.parallel.cpu.compute_residues_kernels as\
    compute_residues_kernels_cpu
import pylabolt.parallel.gpu.compute_residues_kernels as\
    compute_residues_kernels_gpu


class ResidueOperator:
    def __init__(
        self,
        model,
        state,
        comm,
        verbose=True
    ):
        """
        Compute residuals
        Attributes:

        """
        try:
            print_log("-" * 80, state.domain.mpi_rank, verbose)
            print_log("Setting up residue operator...\n",
                      state.domain.mpi_rank, verbose)
            self.model = model
            self.setup_residue_operator(state)
            print_log("\nSetting up residue operator done!",
                      state.domain.mpi_rank, verbose)
            print_log("-" * 80, state.domain.mpi_rank, verbose)
        except Exception as e:
            print_log("-" * 80, state.domain.mpi_rank, verbose=True)
            print_log("FATAL ERROR!", state.domain.mpi_rank, verbose=True)
            print_log(str(e), state.domain.mpi_rank, verbose=True)
            comm.Abort()

    def setup_residue_operator(
        self,
        state,
        verbose=True
    ):
        """
        Setup Residue operator
        Args:

        Returns:

        """
        self.fields_list = self.model.residue_fields
        self.residues = {}
        self.fields_old = {}
        for field_name in self.fields_list:
            if not hasattr(state.fields, field_name):
                raise ValueError(
                    field_name + " is not a valid field for" +
                    " residue computation"
                )
            self.fields_old.update({
                field_name: np.zeros_like(getattr(state.fields, field_name))
            })
            field_shape = self.fields_old[field_name].shape
            if len(field_shape) == 1:
                components = 1
            else:
                components = field_shape[1]
            self.residues.update({
                "res_" + field_name:
                np.zeros(components, dtype=state.control.precision)
            })

    def compile(
        self,
        state,
        backend,
        verbose=True
    ):
        """
        JIT compile compute fields kernels
        Args:

        Returns:

        """
        if backend.backend_type == "cpu":
            for item in self.fields_list:
                args = (
                    state.domain.size,
                    state.fields.solid,
                    state.fields.ghost_node,
                    getattr(state.fields, item),
                    self.fields_old[item],
                )
                compile_args = backend.make_compile_args(args)
                if len(self.residues["res_" + item]) == 1:
                    self.compute_residues_kernel_scalar(*compile_args)
                elif len(self.residues["res_" + item]) == 2:
                    self.compute_residues_kernel_vector(*compile_args)

            self.kernel_signatures = {
                self.compute_residues_kernel_scalar.__name__:
                    set(self.compute_residues_kernel_scalar.signatures),
                self.compute_residues_kernel_vector.__name__:
                    set(self.compute_residues_kernel_vector.signatures)
            }

        elif backend.backend_type == "gpu":
            for item in self.fields_list:
                args = (
                    state.domain.size_device,
                    state.fields.solid_device,
                    state.fields.ghost_node_device,
                    getattr(state.fields, item + "_device"),
                    self.fields_old_device[item]
                )
                compile_args = backend.make_compile_args(args)
                if len(self.residues["res_" + item]) == 1:
                    self.compute_residues_kernel_scalar[
                        backend.reduce_blocks, backend.reduce_threads_per_block
                    ](
                        *compile_args,
                        self.partial_numerator_device[item],
                        self.partial_denominator_device[item]
                    )
                    self.reduce_residues_kernel_scalar[
                        backend.reduce_blocks, backend.reduce_threads_per_block
                    ](
                        backend.reduce_blocks,
                        self.partial_numerator_device[item],
                        self.partial_denominator_device[item]
                    )
                elif len(self.residues["res_" + item]) == 2:
                    self.compute_residues_kernel_vector[
                        backend.reduce_blocks, backend.reduce_threads_per_block
                    ](
                        *compile_args,
                        self.partial_numerator_device[item],
                        self.partial_denominator_device[item]
                    )
                    self.reduce_residues_kernel_vector[
                        backend.reduce_blocks, backend.reduce_threads_per_block
                    ](
                        backend.reduce_blocks,
                        self.partial_numerator_device[item],
                        self.partial_denominator_device[item]
                    )

            self.kernel_signatures = {
                self.compute_residues_kernel_scalar.__name__:
                    set(self.compute_residues_kernel_scalar.signatures),
                self.compute_residues_kernel_vector.__name__:
                    set(self.compute_residues_kernel_vector.signatures),
                    self.reduce_residues_kernel_scalar.__name__:
                    set(self.reduce_residues_kernel_scalar.signatures),
                self.reduce_residues_kernel_vector.__name__:
                    set(self.reduce_residues_kernel_vector.signatures)
            }

        print_log("Compiled residue operator",
                  state.domain.mpi_rank, verbose)

    def compute_residues_cpu(
        self,
        state,
        backend,
        mpi_operator
    ):
        """
        Compute residuals on CPU kernels
        Args:

        Returns:

        """
        for item in self.fields_list:
            args = (
                state.domain.size,
                state.fields.solid,
                state.fields.ghost_node,
                getattr(state.fields, item),
                self.fields_old[item]
            )
            if len(self.residues["res_" + item]) == 1:
                local_res_array =\
                    self.compute_residues_kernel_scalar(*args)
                global_res_array = mpi_operator.reduce(
                    local_res_array,
                    operation="sum"
                )
                self.residues["res_" + item][0] = np.sqrt(
                    global_res_array[0] /
                    (global_res_array[1] + state.control.float_min)
                )
            elif len(self.residues["res_" + item]) == 2:
                local_res_array =\
                    self.compute_residues_kernel_vector(*args)
                global_res_array = mpi_operator.reduce(
                    local_res_array,
                    operation="sum"
                )
                self.residues["res_" + item][0] = np.sqrt(
                    global_res_array[0] /
                    (global_res_array[1] + state.control.float_min)
                )
                self.residues["res_" + item][1] = np.sqrt(
                    global_res_array[2] /
                    (global_res_array[3] + state.control.float_min)
                )

    def compute_residues_gpu(
        self,
        state,
        backend,
        mpi_operator
    ):
        """
        Compute residuals on GPU kernels
        Args:

        Returns:

        """
        for item in self.fields_list:
            args = (
                state.domain.size_device,
                state.fields.solid_device,
                state.fields.ghost_node_device,
                getattr(state.fields, item + "_device"),
                self.fields_old_device[item],
                self.partial_numerator_device[item],
                self.partial_denominator_device[item]
            )
            if len(self.residues["res_" + item]) == 1:
                self.compute_residues_kernel_scalar[
                    backend.reduce_blocks, backend.reduce_threads_per_block
                ](*args)
                partial_size = backend.reduce_blocks
                while partial_size > 1:
                    blocks = int(np.ceil(
                        partial_size / backend.reduce_threads_per_block
                    ))
                    self.reduce_residues_kernel_scalar[
                        blocks, backend.reduce_threads_per_block
                    ](
                        partial_size,
                        self.partial_numerator_device[item],
                        self.partial_denominator_device[item]
                    )
                    partial_size = blocks
                local_numerator = self.partial_numerator_device[item][:1].\
                    copy_to_host()[0]
                local_denominator = self.partial_denominator_device[item][:1].\
                    copy_to_host()[0]
                self.residues["res_" + item][0] = np.sqrt(
                    local_numerator /
                    (local_denominator + state.control.float_min)
                )
            elif len(self.residues["res_" + item]) == 2:
                self.compute_residues_kernel_vector[
                    backend.reduce_blocks, backend.reduce_threads_per_block
                ](*args)
                partial_size = backend.reduce_blocks
                while partial_size > 1:
                    blocks = int(np.ceil(
                        partial_size / backend.reduce_threads_per_block
                    ))
                    self.reduce_residues_kernel_vector[
                        blocks, backend.reduce_threads_per_block
                    ](
                        partial_size,
                        self.partial_numerator_device[item],
                        self.partial_denominator_device[item]
                    )
                    partial_size = blocks
                local_numerator = self.partial_numerator_device[item][:1].\
                    copy_to_host()[0]
                local_denominator = self.partial_denominator_device[item][:1].\
                    copy_to_host()[0]
                self.residues["res_" + item][0] = np.sqrt(
                    local_numerator[0] /
                    (local_denominator[0] + state.control.float_min)
                )
                self.residues["res_" + item][1] = np.sqrt(
                    local_numerator[1] /
                    (local_denominator[1] + state.control.float_min)
                )

    def set_backend(
        self,
        state,
        backend,
        verbose=True
    ):
        """
        Set backend for residue operator
        Args:

        Returns:

        """
        if backend.backend_type == "cpu":
            self.compute_residues = self.compute_residues_cpu
            self.compute_residues_kernel_scalar =\
                compute_residues_kernels_cpu.compute_residue_scalar
            self.compute_residues_kernel_vector =\
                compute_residues_kernels_cpu.compute_residue_vector
        elif backend.backend_type == "gpu":
            self.compute_residues = self.compute_residues_gpu
            self.compute_residues_kernel_scalar =\
                compute_residues_kernels_gpu.compute_residue_scalar
            self.compute_residues_kernel_vector =\
                compute_residues_kernels_gpu.compute_residue_vector
            self.reduce_residues_kernel_scalar =\
                compute_residues_kernels_gpu.reduce_residue_scalar
            self.reduce_residues_kernel_vector =\
                compute_residues_kernels_gpu.reduce_residue_vector
            self.fields_old_device = {}
            self.partial_numerator_device = {}
            self.partial_denominator_device = {}
            for field_name in self.fields_list:
                self.fields_old_device.update({
                    field_name: backend.allocate_to_device(
                        self.fields_old[field_name]
                    )
                })
                components = len(self.residues["res_" + field_name])
                if components == 1:
                    self.partial_numerator_device.update({
                        field_name: cuda.device_array(
                            backend.reduce_blocks, dtype=float
                        )
                    })
                    self.partial_denominator_device.update({
                        field_name: cuda.device_array(
                            backend.reduce_blocks, dtype=float
                        )
                    })
                elif components == 2:
                    self.partial_numerator_device.update({
                        field_name: cuda.device_array(
                            (backend.reduce_blocks, 2), dtype=float
                        )
                    })
                    self.partial_denominator_device.update({
                        field_name: cuda.device_array(
                            (backend.reduce_blocks, 2), dtype=float
                        )
                    })

        print_log("Backend set for residue operator",
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
            compute_residues_kernels_module = compute_residues_kernels_cpu
        elif backend.backend_type == "gpu":
            compute_residues_kernels_module = compute_residues_kernels_gpu
        for kernel_name in self.kernel_signatures:
            kernel = getattr(compute_residues_kernels_module, kernel_name)
            if (set(kernel.signatures) !=
                    self.kernel_signatures[kernel_name]):
                raise RuntimeError(
                    f"Developer error! {kernel_name} in"
                    f" residue operator compiled a new signature!"
                )

        print_log("Kernel signatures verified for residue operator",
                  state.domain.mpi_rank, verbose)
