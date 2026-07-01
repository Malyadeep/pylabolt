from pylabolt.utils.helpers import print_log
from pylabolt.parallel.cpu import compute_fields_kernels as\
    compute_fields_kernels_cpu
# from pylabolt.parallel.gpu import streaming_kernels as streaming_kernels_gpu


class ComputeFieldsOperator:
    def __init__(
        self,
        model,
        state,
        collision_operator,
        backend,
        verbose=True
    ):
        """
        Compute fields operator
        Attributes:

        """
        print_log("-" * 80, state.domain.mpi_rank, verbose)
        print_log("Setting up compute fields operator...",
                  state.domain.mpi_rank, verbose)
        self.set_backend(model, state, collision_operator, backend)
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
        if state.fluid:
            compile_args = backend.make_compile_args(
                self.compute_fields_args_fluid
            )
            self.compute_fields_kernel_fluid(*compile_args)

        elif state.phase:
            compile_args = backend.make_compile_args(
                self.compute_fields_args_phase
            )
            self.compute_fields_kernel_phase(*compile_args)
        print_log("Compiled compute fields operator",
                  state.domain.mpi_rank, verbose)

    def compute_fields_cpu(
        self,
        state,
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
        fluid=False,
        phase=False
    ):
        """
        Perform fields update on GPU kernels
        Args:

        Returns:

        """
        pass

    def set_backend(
        self,
        model,
        state,
        collision_operator,
        backend
    ):
        """
        Set backend for compute fields operator
        Args:

        Returns:

        """
        if backend.backend_type == "cpu":
            self.compute_fields = self.compute_fields_cpu
        elif backend.backend_type == "gpu":
            self.compute_fields = self.compute_fields_gpu
            # TODO: transfer streaming operator attributes to device

        self.compute_fields_type = model.compute_fields_type
        args = model.get_compute_fields_args()

        if state.fluid:
            if collision_operator.forcing_fluid is not None:
                kernel_name = (
                    self.compute_fields_type["fluid"] + "_force"
                )
            else:
                kernel_name = (
                    self.compute_fields_type["fluid"] + "_no_force"
                )
            self.compute_fields_kernel_fluid = getattr(
                compute_fields_kernels_cpu, kernel_name
            )
            args_fluid = args["fluid"]
            self.compute_fields_args_fluid = ()
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
                self.streaming_type["phase"]
            )
            self.streaming_kernel_phase = getattr(
                compute_fields_kernels_cpu, kernel_name
            )
            args_phase = args["phase"]
            self.compute_fields_args_phase = ()
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
