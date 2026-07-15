import numpy as np

from pylabolt.utils.helpers import print_log
import pylabolt.parallel.cpu.compute_residues_kernels as\
    compute_residues_kernels_cpu


class ResidueOperator:
    def __init__(
        self,
        model,
        state,
        mpi_operator,
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
            mpi_operator.comm.Abort()

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
        for field_name in self.fields_list:
            if not hasattr(state.fields, field_name):
                raise ValueError(
                    field_name + " is not a valid field for" +
                    " residue computation"
                )
            setattr(
                self,
                field_name + "_old",
                np.zeros_like(getattr(state.fields, field_name))
            )
            field_shape = getattr(self, field_name + "_old").shape
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
        for item in self.fields_list:
            args = (
                state.domain.size,
                state.fields.solid,
                state.fields.ghost_node,
                getattr(state.fields, item),
                getattr(self, item + "_old"),
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

        print_log("Compiled residue operator",
                  state.domain.mpi_rank, verbose)

    def compute_residues_cpu(
        self,
        state,
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
                getattr(self, item + "_old")
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
        state
    ):
        """
        Compute residuals on GPU kernels
        Args:

        Returns:

        """
        pass

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
            # TODO: transfer residue fields to GPU

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
        for kernel_name in self.kernel_signatures:
            kernel = getattr(compute_residues_kernels_cpu, kernel_name)
            if (set(kernel.signatures) !=
                    self.kernel_signatures[kernel_name]):
                raise RuntimeError(
                    f"Developer error! {kernel_name} in"
                    f" residue operator compiled a new signature!"
                )

        print_log("Kernel signatures verified for residue operator",
                  state.domain.mpi_rank, verbose)
