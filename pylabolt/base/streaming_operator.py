from pylabolt.utils.helpers import print_log
import pylabolt.parallel.cpu.streaming_kernels as streaming_kernels_cpu
# from pylabolt.parallel.gpu import streaming_kernels as streaming_kernels_gpu


class StreamingOperator:
    def __init__(
        self,
        model,
        state,
        verbose=True
    ):
        """
        Streaming operator
        Attributes:

        """
        print_log("-" * 80, state.domain.mpi_rank, verbose)
        print_log("Setting up streaming operator...",
                  state.domain.mpi_rank, verbose)
        self.model = model
        print_log("Setting up streaming operator done!",
                  state.domain.mpi_rank, verbose)
        print_log("-" * 80, state.domain.mpi_rank, verbose)

    def compile(
        self,
        state,
        backend,
        verbose=True
    ):
        """
        JIT compile streaming kernels
        Args:

        Returns:

        """
        self.kernel_signatures = {}
        if state.fluid:
            compile_args = backend.make_compile_args(self.streaming_args_fluid)
            self.streaming_kernel_fluid(*compile_args)
            self.kernel_signatures.update({
                self.streaming_kernel_fluid.__name__:
                    set(self.streaming_kernel_fluid.signatures)
            })

        elif state.phase:
            compile_args = backend.make_compile_args(self.streaming_args_phase)
            self.streaming_kernel_phase(*compile_args)
            self.kernel_signatures.update({
                self.streaming_kernel_phase.__name__:
                    set(self.streaming_kernel_phase.signatures)
            })

        print_log("Compiled streaming operator",
                  state.domain.mpi_rank, verbose)

    def stream_cpu(
        self,
        state,
        fluid=False,
        phase=False
    ):
        """
        Perform streaming operation on CPU kernels
        Args:

        Returns:

        """
        if fluid:
            self.streaming_kernel_fluid(*self.streaming_args_fluid)

        if phase:
            self.streaming_kernel_phase(*self.streaming_args_phase)

    def stream_gpu(
        self,
        state,
        fluid=False,
        phase=False
    ):
        """
        Perform streaming operation on GPU kernels
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
        Set backend for streaming operator
        Args:

        Returns:

        """
        if backend.backend_type == "cpu":
            self.stream = self.stream_cpu
        elif backend.backend_type == "gpu":
            self.stream = self.stream_gpu
            # TODO: transfer streaming operator attributes to device

        self.streaming_type = self.model.streaming_type
        args = self.model.get_streaming_args()

        if state.fluid:
            kernel_name = (
                self.streaming_type["fluid"] + "_kernel"
            )
            self.streaming_kernel_fluid = getattr(
                streaming_kernels_cpu, kernel_name
            )
            args_fluid = args["fluid"]
            self.streaming_args_fluid = ()
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
                self.streaming_args_fluid += key_args

        if state.phase:
            kernel_name = (
                self.streaming_type["phase"] + "_kernel"
            )
            self.streaming_kernel_phase = getattr(
                streaming_kernels_cpu, kernel_name
            )
            args_phase = args["phase"]
            self.streaming_args_phase = ()
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
                self.streaming_args_phase += key_args

        print_log("Backend set for streaming operator",
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
            kernel = getattr(streaming_kernels_cpu, kernel_name)
            if (set(kernel.signatures) !=
                    self.kernel_signatures[kernel_name]):
                raise RuntimeError(
                    f"Developer error! {kernel_name} in"
                    f" streaming operator compiled a new signature!"
                )

        print_log("Kernel signatures verified for streaming operator",
                  state.domain.mpi_rank, verbose)
