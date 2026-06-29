from pylabolt.utils.helpers import print_log
from pylabolt.parallel.cpu import streaming_kernels as streaming_kernels_cpu
# from pylabolt.parallel.gpu import streaming_kernels as streaming_kernels_gpu


class StreamingOperator:
    def __init__(
        self,
        model,
        state,
        backend,
        verbose=True
    ):
        """
        Streaming operator
        Attributes:

        """
        print_log("-" * 80, state.domain.mpi_rank, verbose)
        print_log("Setting up streaming operator...\n",
                  state.domain.mpi_rank, verbose)
        self.set_backend(model, state, backend)
        print_log("Setting up streaming operator done!",
                  state.domain.mpi_rank, verbose)
        print_log("-" * 80, state.domain.mpi_rank, verbose)

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
        model,
        state,
        backend
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

        self.streaming_type = model.streaming_type
        args = model.get_streaming_args()

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
