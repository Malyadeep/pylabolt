import numpy as np

from pylabolt.utils.helpers import print_log
from pylabolt.parallel.cpu import collision_kernels as collision_kernels_cpu
# from pylabolt.parallel.gpu import collision_kernels as collision_kernels_gpu


class CollisionOperator:
    def __init__(
        self,
        comm,
        simulation,
        model,
        state,
        backend,
        verbose=True
    ):
        """
        Collision operator
        Attributes:

        """
        try:
            print_log("-" * 80, state.domain.mpi_rank, verbose)
            print_log("Setting up collision operator...\n",
                      state.domain.mpi_rank, verbose)
            if not hasattr(simulation, "collision_dict"):
                raise ValueError("fluid missing in collision_dict")
            self.collision_dict = simulation.collision_dict
            self.read_collision_dict(model, state, verbose=verbose)
            self.set_backend(model, state, backend)
            print_log("\nSetting up collision operator done!",
                      state.domain.mpi_rank, verbose)
            print_log("-" * 80, state.domain.mpi_rank, verbose)
        except Exception as e:
            print_log("-" * 80, state.domain.mpi_rank, verbose=True)
            print_log("FATAL ERROR!", state.domain.mpi_rank, verbose=True)
            print_log(str(e), state.domain.mpi_rank, verbose=True)
            comm.Abort()

    def read_collision_dict(
        self,
        model,
        state,
        verbose=True
    ):
        """
        Read collision dict and set collision kernels
        Args:

        Returns:

        """
        if state.fluid:
            if "fluid" not in self.collision_dict:
                raise ValueError("fluid missing in collision_dict")
            fluid_dict = self.collision_dict["fluid"]
            for key in ["model", "equilibrium", "forcing_model"]:
                if key not in fluid_dict:
                    raise ValueError(key + " missing in fluid: collision_dict")
            self.collision_fluid = fluid_dict["model"]
            self.equilibrium_fluid = fluid_dict["equilibrium"]
            self.forcing_fluid = fluid_dict["forcing_model"]
            if self.collision_fluid not in model.collision_models["fluid"]:
                raise ValueError(
                    "Unsupported fluid collision model: " +
                    self.collision_fluid +
                    "\nAvailable models: " + model.collision_models["fluid"]
                )
            if self.equilibrium_fluid not in model.equilibrium_models["fluid"]:
                raise ValueError(
                    "Unsupported fluid equilibrium model: " +
                    self.equilibrium_fluid +
                    "\nAvailable models: " + model.equilibrium_models["fluid"]
                )
            if self.forcing_fluid not in model.forcing_models["fluid"]:
                raise ValueError(
                    "Unsupported fluid forcing model: " + self.forcing_fluid +
                    "\nAvailable models: " + model.forcing_models["fluid"]
                )
            if self.forcing_fluid is None:
                self.gravity = np.zeros(2, dtype=state.control.precision)
                if "gravity" in fluid_dict:
                    print_log(
                        "WARNING! gravity ignored in fluid collision dict" +
                        " as forcing is set to None\n",
                        state.domain.mpi_rank,
                        verbose=verbose
                    )
            else:
                if "gravity" not in fluid_dict:
                    raise ValueError("gravity missing in fluid collision dict")
                self.gravity = fluid_dict["gravity"]
                if (not isinstance(self.gravity, list) and
                        len(self.gravity) == 2):
                    raise ValueError(
                        "gravity must be a list (gx, gy) in fluid" +
                        " collision dict"
                    )
                self.gravity = np.array(
                    self.gravity, dtype=state.control.precision
                )

            self.tau_fluid = state.transport.kin_visc *\
                state.lattice.inv_cs_2 + 0.5
            self.omega_fluid = 1 / self.tau_fluid
            if self.collision_fluid == "MRT":
                self.setup_MRT_params()
                self.collision_params = (
                    self.gravity,
                    self.M,
                    self.inv_M,
                    self.S,
                    self.pre_factor_mat,
                    self.inv_pre_factor_mat
                )
            elif self.collision_fluid == "BGK":
                self.collision_params = (
                    self.gravity,
                    self.omega_fluid
                )

        if state.phase:
            if "phase" not in self.collision_dict:
                raise ValueError("phase missing in collision_dict")
            phase_dict = self.collision_dict["phase"]
            if "model" not in phase_dict:
                raise ValueError("model missing in phase: collision_dict")
            self.model_phase = phase_dict["model"]
            if self.model_phase not in model.collision_models["phase"]:
                raise ValueError(
                    "Unsupported phase collision model: " + self.model_phase +
                    "\nAvailable models: " + model.collision_models["phase"]
                )
            if self.model_phase == "segregation":
                self.equilibrium_phase = None
            else:
                if "equilibrium" not in phase_dict:
                    raise ValueError(
                        "equilibrium missing in phase: collision_dict"
                    )
                self.equilibrium_phase = phase_dict["equilibrium"]
                if (self.equilibrium_phase not in
                        model.equilibrium_models["phase"]):
                    raise ValueError(
                        "Unsupported phase equilibrium model: " +
                        self.equilibrium_phase
                    )

        self.log_collision_operator_data(state, verbose=verbose)

    def setup_MRT_params(
        self,
        state
    ):
        """
        Sets up MRT matrices
        Args:

        Returns:

        """
        self.M = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [-4, -1, -1, -1, -1, 2, 2, 2, 2],
            [4, -2, -2, -2, -2, 1, 1, 1, 1],
            [0, 1, 0, -1, 0, 1, -1, -1, 1],
            [0, -2, 0, 2, 0, 1, -1, -1, 1],
            [0, 0, 1, 0, -1, 1, 1, -1, -1],
            [0, 0, -2, 0, 2, 1, 1, -1, -1],
            [0, 1, -1, 1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, -1, 1, -1]
        ], dtype=state.control.precision)
        self.inv_M = np.linalg.inv(self.M)
        self.S = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
             self.omega_fluid, self.omega_fluid],
            dtype=state.control.precision
        )
        self.pre_factor_mat = np.matmul(np.matmul(self.inv_M, self.S), self.M)
        self.inv_pre_factor_mat = np.linalg.inv(self.pre_factor_mat)

    def log_collision_operator_data(
        self,
        state,
        verbose=False
    ):
        """
        Prints user-defined collision-equilibrium model
        Args:

        Returns:

        """
        if state.fluid:
            print_log(
                f"{'Fluid collision model':<30}: {self.collision_fluid}",
                state.domain.mpi_rank, verbose=verbose
            )
            print_log(
                f"{'Fluid equilibrium model':<30}: {self.equilibrium_fluid}",
                state.domain.mpi_rank, verbose=verbose
            )
            print_log(
                f"{'Fluid forcing model':<30}: {str(self.forcing_fluid)}",
                state.domain.mpi_rank, verbose=verbose
            )
            print_log(
                f"{'Gravity':<30}: {self.gravity}",
                state.domain.mpi_rank, verbose=verbose
            )

        if state.phase:
            print_log(
                f"{'Phase collision model':<30}: {self.collision_phase}",
                state.domain.mpi_rank, verbose=verbose
            )
            print_log(
                f"{'Phase equilibrium model':<30}: {self.equilibrium_phase}",
                state.domain.mpi_rank, verbose=verbose
            )

    def collide_cpu(
        self,
        state,
        fluid=False,
        phase=False
    ):
        """
        Perform collision operation on CPU kernels
        Args:

        Returns:

        """
        if fluid:
            self.collision_kernel_fluid(*self.collision_args_fluid)

        if phase:
            self.collision_kernel_phase(*self.collision_args_phase)

    def collide_gpu(
        self,
        state,
        fluid=False,
        phase=False
    ):
        """
        Perform collision operation on GPU kernels
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
        Set backend for collision operator
        Args:

        Returns:

        """
        if backend.backend_type == "cpu":
            self.collide = self.collide_cpu
        elif backend.backend_type == "gpu":
            self.collide = self.collide_gpu
            # TODO: transfer collision operator attributes to device

        args = model.get_collision_args()

        if state.fluid:
            kernel_name = (
                self.collision_fluid + "_" +
                self.equilibrium_fluid + "_" +
                str(self.forcing_fluid)
            )
            self.collision_kernel_fluid = getattr(
                collision_kernels_cpu, kernel_name
            )
            args_fluid = args["fluid"]
            self.collision_args_fluid = ()
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
                self.collision_args_fluid += key_args
            self.collision_args_fluid += self.collision_params

        if state.phase:
            kernel_name = (
                self.collision_phase + "_" +
                self.equilibrium_phase + "_" +
                str(self.forcing_phase)
            )
            self.collision_kernel_phase = getattr(
                collision_kernels_cpu, kernel_name
            )
            args_phase = args["phase"]
            self.collision_args_phase = ()
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
                self.collision_args_phase += key_args
            self.collision_args_phase += self.collision_params
