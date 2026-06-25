import numpy as np

from pylabolt.utils.helpers import print_log
from pylabolt.parallel.cpu.obstacle_kernels import (
    compute_obstacle_boundary,
    compute_normals_circle,
    compute_normals_ellipse
)


class CollisionOperator:
    def __init__(
        self,
        comm,
        simulation,
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
            self.read_collision_dict(state, verbose=verbose)
            print_log("\nSetting up collision operator done!",
                      state.domain.mpi_rank, verbose)
            print_log("-" * 80, state.domain.mpi_rank, verbose)
        except Exception as e:
            print_log("-" * 80, state.domain.mpi_rank, verbose=True)
            print_log("FATAL ERROR!", state.domain.mpi_rank, verbose=True)
            print_log(str(e), state.domain.mpi_rank, verbose=True)
            comm.Abort()
        # self.set_backend(backend)

    def read_collision_dict(
        self,
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
            self.model_fluid = fluid_dict["model"]
            if self.model_fluid not in ["BGK", "MRT"]:
                raise ValueError(
                    "Unsupported fluid collision model: " + self.model_fluid +
                    "\nAvailable models: BGK, MRT"
                )
            self.equilibrium_fluid = fluid_dict["equilibrium"]
            if self.equilibrium_fluid not in [
                "density_based_second_order",
                "velocity_based_second_order"
            ]:
                raise ValueError(
                    "Unsupported fluid equilibrium model: " +
                    self.equilibrium_fluid +
                    "\nAvailable models: density_based_second_order, " +
                    "velocity_based_second_order"
                )
            self.forcing_fluid = fluid_dict["forcing_model"]
            if self.forcing_fluid not in [
                None,
                "guo_linear",
                "guo_second_order"
            ]:
                raise ValueError(
                    "Unsupported fluid forcing model: " +
                    self.forcing_fluid
                )
            if self.forcing_fluid is None:
                self.gravity = np.zeros(2, dtype=state.control.precision)
                if "gravity" in fluid_dict:
                    print_log(
                        "WARNING! gravity ignored in fluid collision dict" +
                        " as forcing is set to None",
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
        if state.phase:
            if "phase" not in self.collision_dict:
                raise ValueError("phase missing in collision_dict")
            phase_dict = self.collision_dict["phase"]
            if "model" not in phase_dict:
                raise ValueError("model missing in phase: collision_dict")
            self.model_phase = phase_dict["model"]
            if self.model_phase not in ["BGK", "segregation"]:
                raise ValueError(
                    "Unsupported phase collision model: " + self.model_fluid
                )
            if self.model_phase == "segregation":
                self.equilibrium_phase = None
            else:
                if "model" not in phase_dict:
                    raise ValueError(
                        "equilibrium missing in phase: collision_dict"
                    )
                self.equilibrium_phase = fluid_dict["equilibrium"]
                if self.equilibrium_phase not in [
                    "second_order"
                ]:
                    raise ValueError(
                        "Unsupported phase equilibrium model: " +
                        self.equilibrium_phase
                    )

    def collide_cpu(
        self,
        state
    ):
        """
        Perform collision operation on CPU kernels
        Args:

        Returns:

        """
        pass

    def set_backend(
        self,
        backend
    ):
        """
        Set backend for collision operator
        Args:

        Returns:

        """
        pass
        # if backend.backend_type == "cpu":
        #     self.find_obstacle_boundary_nodes =\
        #         self.find_obstacle_boundary_nodes_cpu
        #     self.find_obstacle_normals =\
        #         self.find_obstacle_normals_cpu
        # elif backend.backend_type == "gpu":
        #     self.find_obstacle_boundary_nodes =\
        #         self.find_obstacle_boundary_nodes_gpu
        #     self.find_obstacle_normals =\
        #         self.find_obstacle_normals_gpu
