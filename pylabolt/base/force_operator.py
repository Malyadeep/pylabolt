import numpy as np

from pylabolt.utils.helpers import print_log
import pylabolt.parallel.cpu.force_field_kernels as force_field_kernels_cpu
import pylabolt.parallel.gpu.force_field_kernels as force_field_kernels_gpu


class ForceOperator:
    def __init__(
        self,
        simulation,
        model,
        state,
        comm,
        collision_operator=None,
        phase_field_operator=None,
        verbose=True
    ):
        """
        Force computation and accumulation operator
        Attributes:

        """
        try:
            print_log("-" * 80, state.domain.mpi_rank, verbose)
            print_log("Setting up forcing operator...\n",
                      state.domain.mpi_rank, verbose)
            if not hasattr(simulation, "forcing_dict"):
                raise ValueError(
                    "forcing_dict not found in simulation.py file"
                )
            self.model = model
            self.collision_operator = collision_operator
            self.phase_field_operator = phase_field_operator
            self.forcing_dict = simulation.forcing_dict
            self.read_forcing_dict(state, verbose=verbose)
            print_log("\nSetting up forcing operator done!",
                      state.domain.mpi_rank, verbose)
            print_log("-" * 80, state.domain.mpi_rank, verbose)
        except Exception as e:
            print_log("-" * 80, state.domain.mpi_rank, verbose=True)
            print_log("FATAL ERROR!", state.domain.mpi_rank, verbose=True)
            print_log(str(e), state.domain.mpi_rank, verbose=True)
            comm.Abort()

    def read_forcing_dict(
        self,
        state,
        verbose=True
    ):
        """
        Read forcing dict
        Args:

        Returns:

        """
        if state.fluid:
            self.gravity = np.zeros(2, dtype=state.control.precision)
            if "gravity" not in self.forcing_dict:
                return
            else:
                if self.collision_operator.forcing_fluid is None:
                    print_log(
                        "WARNING! gravity ignored in forcing dict" +
                        " as forcing is set to None in collision dict\n",
                        state.domain.mpi_rank,
                        verbose=verbose
                    )
                else:
                    self.gravity = self.forcing_dict["gravity"]
                    if (not isinstance(self.gravity, list) and
                            len(self.gravity) == 2):
                        raise ValueError(
                            "gravity must be a list (gx, gy) in forcing dict"
                        )
                    self.gravity = np.array(
                        self.gravity, dtype=state.control.precision
                    )

        self.log_force_operator_data(state, verbose=verbose)

    def log_force_operator_data(
        self,
        state,
        verbose=False
    ):
        """
        Prints user-defined forcing dict data
        Args:

        Returns:

        """
        if state.fluid:
            print_log(
                f"{'Gravity':<30}: {self.gravity}",
                state.domain.mpi_rank, verbose=verbose
            )

    def compile(
        self,
        state,
        backend,
        verbose=True
    ):
        """
        JIT compile force kernels
        Args:

        Returns:

        """
        self.kernel_signatures = {}

        if state.fluid:
            compile_args = backend.make_compile_args(
                self.compute_gravity_force_args
            )
            if backend.backend_type == "cpu":
                self.compute_gravity_force(*compile_args)
            elif backend.backend_type == "gpu":
                self.compute_gravity_force[
                    backend.blocks, backend.threads_per_block
                ](*compile_args)

            self.kernel_signatures.update({
                self.compute_gravity_force.__name__:
                    set(self.compute_gravity_force.signatures)
            })

        if state.phase:
            pass

        print_log("Compiled force operator",
                  state.domain.mpi_rank, verbose)

    def compute_force_field_cpu(
        self,
        state,
        backend
    ):
        """
        Compute forces acting on the fluid using CPU kernels
        Args:

        Returns:

        """
        self.compute_gravity_force(*self.compute_gravity_force_args)

    def compute_force_field_gpu(
        self,
        state,
        backend
    ):
        """
        Compute forces acting on the fluid using GPU kernels
        Args:

        Returns:

        """
        self.compute_gravity_force[
            backend.blocks, backend.threads_per_block
        ](*self.compute_gravity_force_args)

    def set_backend(
        self,
        state,
        backend,
        verbose=True
    ):
        """
        Set backend for collision operator
        Args:

        Returns:

        """
        if backend.backend_type == "cpu":
            self.compute_force_field = self.compute_force_field_cpu
            force_field_kernels_module = force_field_kernels_cpu
            suffix = ""
        elif backend.backend_type == "gpu":
            self.compute_force_field = self.compute_force_field_gpu
            force_field_kernels_module = force_field_kernels_gpu
            self._device_attrs = ["gravity"]
            for arg_name in self._device_attrs:
                arg_device = backend.allocate_to_device(
                    getattr(self, arg_name)
                )
                setattr(self, arg_name + "_device", arg_device)
            suffix = "_device"

        if state.fluid:
            self.compute_gravity_force = getattr(
                force_field_kernels_module,
                "compute_gravity_force"
            )

            self.compute_gravity_force_args = (
                getattr(state.domain, "size" + suffix),
                getattr(state.fields, "solid" + suffix),
                getattr(state.fields, "ghost_node" + suffix),
                getattr(state.fields, "density" + suffix),
                getattr(state.fields, "force_field" + suffix),
                getattr(self, "gravity" + suffix)
            )

        if state.phase:
            pass

        print_log("Backend set for force operator",
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
            force_field_kernels_module = force_field_kernels_cpu
        elif backend.backend_type == "gpu":
            force_field_kernels_module = force_field_kernels_gpu
        for kernel_name in self.kernel_signatures:
            kernel = getattr(force_field_kernels_module, kernel_name)
            if (set(kernel.signatures) !=
                    self.kernel_signatures[kernel_name]):
                raise RuntimeError(
                    f"Developer error! {kernel_name} in"
                    f" force operator compiled a new signature!"
                )
        print_log("Kernel signatures verified for force operator",
                  state.domain.mpi_rank, verbose)
