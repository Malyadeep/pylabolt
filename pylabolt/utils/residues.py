import numpy as np

from pylabolt.utils.helpers import print_log
import pylabolt.parallel.cpu.compute_residues_kernels as\
    compute_residues_kernels_cpu


class ResidueOperator:
    def __init__(
        self,
        comm,
        model,
        state,
        backend,
        verbose=True
    ):
        """
        Compute residuals
        Attributes:

        """
        try:
            print_log("-" * 80, state.domain.mpi_rank, verbose)
            print_log("Setting up residue computation operator...\n",
                      state.domain.mpi_rank, verbose)
            self.setup_residue_operator(model, state)
            self.set_backend(backend)
            print_log("\nSetting up residue computation operator done!",
                      state.domain.mpi_rank, verbose)
            print_log("-" * 80, state.domain.mpi_rank, verbose)
        except Exception as e:
            print_log("-" * 80, state.domain.mpi_rank, verbose=True)
            print_log("FATAL ERROR!", state.domain.mpi_rank, verbose=True)
            print_log(str(e), state.domain.mpi_rank, verbose=True)
            comm.Abort()

    def setup_residue_operator(
        self,
        model,
        state,
        verbose=True
    ):
        """
        Setup Residue operator
        Args:

        Returns:

        """
        self.fields_list = model.residue_fields
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
                field_name:
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
        for item in self.residues:
            args = (
                state.domain.size,
                state.fields.solid,
                state.fields.ghost_node,
                getattr(state.fields, item),
                getattr(self, item + "_old"),
            )
            compile_args = backend.make_compile_args(args)
            if len(self.residues[item]) == 1:
                self.compute_residues_kernel_scalar(*compile_args)
            elif len(self.residues[item]) == 2:
                self.compute_residues_kernel_vector(*compile_args)
        print_log("Compiled compute residues operator",
                  state.domain.mpi_rank, verbose)

    def compute_residues_cpu(
        self,
        state
    ):
        """
        Compute residuals on CPU kernels
        Args:

        Returns:

        """
        for item in self.residues:
            args = (
                state.domain.size,
                state.fields.solid,
                state.fields.ghost_node,
                getattr(state.fields, item),
                getattr(self, item + "_old")
            )
            if len(self.residues[item]) == 1:
                self.compute_residues_kernel_scalar(*args)
            elif len(self.residues[item]) == 2:
                self.compute_residues_kernel_vector(*args)

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
        backend
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
