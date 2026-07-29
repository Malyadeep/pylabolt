from pylabolt.utils.helpers import print_log
import pylabolt.parallel.cpu.compute_fields_kernels as\
    compute_fields_kernels_cpu
import pylabolt.parallel.gpu.compute_fields_kernels as\
    compute_fields_kernels_gpu


class ComputeFieldsOperator:
    def __init__(
        self,
        model,
        state,
        verbose=True
    ):
        """
        Compute fields operator
        Attributes:

        """
        print_log("-" * 80, state.domain.mpi_rank, verbose)
        print_log("Setting up compute fields operator...",
                  state.domain.mpi_rank, verbose)
        self.model = model
        print_log("Setting up compute fields operator done!",
                  state.domain.mpi_rank, verbose)
        print_log("-" * 80, state.domain.mpi_rank, verbose)

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
        self.kernel_signatures = {}

        for moment_field in self.moment_fields:
            kernel = self.compute_fields_kernels[moment_field]
            args = self.compute_fields_args[moment_field]
            compile_args = backend.make_compile_args(args)
            if backend.backend_type == "cpu":
                kernel(*compile_args)
            elif backend.backend_type == "gpu":
                kernel[
                    backend.blocks, backend.threads_per_block
                ](*compile_args)
            self.kernel_signatures.update({
                kernel.__name__: set(kernel.signatures)
            })

        print_log("Compiled compute fields operator",
                  state.domain.mpi_rank, verbose)

    def compute_fields_cpu(
        self,
        state,
        backend,
        field=[]
    ):
        """
        Perform fields update on CPU kernels
        Args:

        Returns:

        """
        for current_field in field:
            kernel = self.compute_fields_kernels[current_field]
            args = self.compute_fields_args[current_field]
            kernel(*args)

    def compute_fields_gpu(
        self,
        state,
        backend,
        field=[]
    ):
        """
        Perform fields update on GPU kernels
        Args:

        Returns:

        """
        for current_field in field:
            kernel = self.compute_fields_kernels[current_field]
            args = self.compute_fields_args[current_field]
            kernel[
                backend.blocks, backend.threads_per_block
            ](*args)

    def set_kernel_args(
        self,
        state,
        backend,
        module,
        kernel_name,
        args_dict
    ):
        """
        Sets collision kernel and args based on configuration specified
        Args:

        Returns:

        """
        kernel = getattr(module, kernel_name)
        args = ()
        for key in args_dict:
            args_list = args_dict[key]
            attribute = getattr(state, key)
            if backend.backend_type == "cpu":
                key_args = tuple(
                    getattr(attribute, item) for item in args_list
                )
            elif backend.backend_type == "gpu":
                key_args = tuple(
                    getattr(attribute, item + "_device")
                    for item in args_list
                )
            args += key_args
        return kernel, args

    def set_backend(
        self,
        state,
        backend,
        verbose=True
    ):
        """
        Set backend for compute fields operator
        Args:

        Returns:

        """
        if backend.backend_type == "cpu":
            self.compute_fields = self.compute_fields_cpu
            compute_fields_kernels_module = compute_fields_kernels_cpu
        elif backend.backend_type == "gpu":
            self.compute_fields = self.compute_fields_gpu
            compute_fields_kernels_module = compute_fields_kernels_gpu

        self.compute_fields_config = self.model.compute_fields_config
        # args = self.model.get_compute_fields_args()
        self.compute_fields_type = self.compute_fields_config["type"]
        self.moment_fields = self.compute_fields_config["moment_fields"]

        self.compute_fields_kernels = {}
        self.compute_fields_args = {}
        args_container = ArgsContainer()
        for moment_field in self.moment_fields:
            args_dict = getattr(args_container, moment_field + "_args")
            kernel_name = moment_field + "_compute_" + self.compute_fields_type
            kernel, args = self.set_kernel_args(
                state,
                backend,
                compute_fields_kernels_module,
                kernel_name,
                args_dict
            )
            self.compute_fields_kernels.update({
                moment_field: kernel
            })
            self.compute_fields_args.update({
                moment_field: args
            })

        print_log("Backend set for compute fields operator",
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
            compute_fields_kernels_module = compute_fields_kernels_cpu
        elif backend.backend_type == "gpu":
            compute_fields_kernels_module = compute_fields_kernels_gpu
        for kernel_name in self.kernel_signatures:
            kernel = getattr(compute_fields_kernels_module, kernel_name)
            if (set(kernel.signatures) !=
                    self.kernel_signatures[kernel_name]):
                raise RuntimeError(
                    f"Developer error! {kernel_name} in"
                    f" compute fields operator compiled a new signature!"
                )

        print_log("Kernel signatures verified for compute fields operator",
                  state.domain.mpi_rank, verbose)


class ArgsContainer:
    def __init__(self):
        self.density_args = {
            "domain": ["size"],
            "lattice": ["no_of_directions"],
            "fields": ["solid", "ghost_node", "density", "pop_fluid_new"]
        }
        self.velocity_args = {
            "control": ["float_min"],
            "domain": ["size"],
            "lattice": ["cx", "cy", "no_of_directions"],
            "fields": [
                "solid", "ghost_node", "density", "velocity",
                "force_field", "pop_fluid_new"
            ]
        }
