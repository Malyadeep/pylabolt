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
        collision_operator,
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

        if state.fluid:
            compile_args = backend.make_compile_args(
                self.compute_fields_args_fluid
            )
            if backend.backend_type == "cpu":
                self.compute_fields_kernel_fluid(*compile_args)
            elif backend.backend_type == "gpu":
                self.compute_fields_kernel_fluid[
                    backend.blocks, backend.threads_per_block
                ](*compile_args)

            self.kernel_signatures.update({
                self.compute_fields_kernel_fluid.__name__:
                    set(self.compute_fields_kernel_fluid.signatures)
            })

        elif state.phase:
            compile_args = backend.make_compile_args(
                self.compute_fields_args_phase
            )
            if backend.backend_type == "cpu":
                self.compute_fields_kernel_phase(*compile_args)
            elif backend.backend_type == "gpu":
                self.compute_fields_kernel_phase[
                    backend.blocks, backend.threads_per_block
                ](*compile_args)

            self.kernel_signatures.update({
                self.compute_fields_kernel_fluid.__name__:
                    set(self.compute_fields_kernel_fluid.signatures)
            })

        print_log("Compiled compute fields operator",
                  state.domain.mpi_rank, verbose)

    def compute_fields_cpu(
        self,
        state,
        backend,
        fluid=False,
        phase=False
    ):
        """
        Perform fields update on CPU kernels
        Args:

        Returns:

        """
        if fluid:
            self.compute_fields_kernel_fluid(*self.compute_fields_args_fluid)

        if phase:
            self.compute_fields_kernel_phase(*self.compute_fields_args_phase)

    def compute_fields_gpu(
        self,
        state,
        backend,
        fluid=False,
        phase=False
    ):
        """
        Perform fields update on GPU kernels
        Args:

        Returns:

        """
        if fluid:
            self.compute_fields_kernel_fluid[
                backend.blocks, backend.threads_per_block
            ](*self.compute_fields_args_fluid)

        if phase:
            self.compute_fields_kernel_phase[
                backend.blocks, backend.threads_per_block
            ](*self.compute_fields_args_phase)

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

        self.compute_fields_type = self.model.compute_fields_type
        args = self.model.get_compute_fields_args()

        if state.fluid:
            if backend.backend_type == "cpu":
                self.compute_fields_args_fluid = tuple(
                    [state.control.float_min]
                )
            elif backend.backend_type == "gpu":
                self.compute_fields_args_fluid = tuple(
                    [state.control.float_min_device]
                )

            kernel_name = (
                self.compute_fields_type["fluid"]
            )
            self.compute_fields_kernel_fluid = getattr(
                compute_fields_kernels_module, kernel_name
            )
            args_fluid = args["fluid"]
            for key_no, key in enumerate(args_fluid):
                args_list = args_fluid[key]
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
                self.compute_fields_args_fluid += key_args

        if state.phase:
            kernel_name = (
                self.compute_fields_type["phase"]
            )
            self.streaming_kernel_phase = getattr(
                compute_fields_kernels_module, kernel_name
            )
            args_phase = args["phase"]
            if backend.backend_type == "cpu":
                self.compute_fields_args_phase = tuple(
                    [state.control.float_min]
                )
            elif backend.backend_type == "gpu":
                self.compute_fields_args_phase = tuple(
                    [state.control.float_min_device]
                )
            for key_no, key in enumerate(args_phase):
                args_list = args_phase[key]
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
                self.compute_fields_args_phase += key_args

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
