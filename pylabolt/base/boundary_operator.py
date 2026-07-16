from pylabolt.utils.helpers import print_log
import pylabolt.parallel.cpu.fluid_boundary_kernels as fluid_kernels_cpu
import pylabolt.parallel.cpu.phase_boundary_kernels as phase_kernels_cpu
import pylabolt.parallel.gpu.fluid_boundary_kernels as fluid_kernels_gpu
import pylabolt.parallel.gpu.phase_boundary_kernels as phase_kernels_gpu


class BoundaryOperator:
    def __init__(
        self,
        model,
        state,
        verbose=True
    ):
        """
        Boundary condition operator
        Attributes:

        """
        print_log("-" * 80, state.domain.mpi_rank, verbose)
        print_log("Setting up boundary condition operator...",
                  state.domain.mpi_rank, verbose)
        self.model = model
        print_log("Setting up boundary condition operator done!",
                  state.domain.mpi_rank, verbose)
        print_log("-" * 80, state.domain.mpi_rank, verbose)

    def compile(
        self,
        state,
        backend,
        verbose=True
    ):
        """
        JIT compile boundary condition kernels
        Args:

        Returns:

        """
        self.kernel_signatures = {}
        if state.fluid:
            self.kernel_signatures.update({"fluid": {}})
            for itr in range(len(self.boundary_kernels_fluid)):
                if self.boundary_kernels_fluid[itr] is not None:
                    compile_args = backend.make_compile_args(
                        self.boundary_args_fluid[itr]
                    )
                    self.boundary_kernels_fluid[itr](*compile_args)
                    self.kernel_signatures["fluid"].update({
                        self.boundary_kernels_fluid[itr].__name__:
                            set(self.boundary_kernels_fluid[itr].signatures)
                    })

        elif state.phase:
            self.kernel_signatures.update({"phase": {}})
            for itr in range(len(self.boundary_kernels_phase)):
                if self.boundary_kernels_phase[itr] is not None:
                    compile_args = backend.make_compile_args(
                        self.boundary_args_phase[itr]
                    )
                    self.boundary_kernels_phase[itr](*compile_args)
                    self.kernel_signatures["phase"].update({
                        self.boundary_kernels_phase[itr].__name__:
                            set(self.boundary_kernels_phase[itr].signatures)
                    })

        print_log("Compiled boundary operator", state.domain.mpi_rank, verbose)

    def set_boundary_cpu(
        self,
        state,
        fluid=False,
        phase=False
    ):
        """
        Apply boundary condition on CPU kernels
        Args:

        Returns:

        """
        if fluid:
            for itr in range(len(self.boundary_kernels_fluid)):
                if self.boundary_kernels_fluid[itr] is not None:
                    self.boundary_kernels_fluid[itr](
                        *self.boundary_args_fluid[itr]
                    )

        if phase:
            for itr in range(len(self.boundary_kernels_phase)):
                if self.boundary_kernels_phase[itr] is not None:
                    self.boundary_kernels_phase[itr](
                        *self.boundary_args_phase[itr]
                    )

    def set_boundary_gpu(
        self,
        state,
        fluid=False,
        phase=False
    ):
        """
        Apply boundary condition on GPU kernels
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
        Set backend for boundary condition operator
        Args:

        Returns:

        """
        if backend.backend_type == "cpu":
            self.set_boundary = self.set_boundary_cpu
        elif backend.backend_type == "gpu":
            self.set_boundary = self.set_boundary_gpu
            # TODO: transfer collision operator attributes to device

        self.boundary_condition_type = self.model.boundary_condition_type

        if state.fluid:
            self.boundary_kernels_fluid = []
            self.boundary_args_fluid = []
            setup_boundary_fluid = SetupBoundaryFluid(
                self.boundary_condition_type["fluid"],
                backend
            )
            for boundary_element in state.boundary.boundary_elements:
                if boundary_element.type_fluid != "periodic":
                    setup_function = getattr(
                        setup_boundary_fluid,
                        "setup_" + boundary_element.type_fluid
                    )
                    kernel, args = setup_function(state, boundary_element)
                    self.boundary_kernels_fluid.append(kernel)
                    self.boundary_args_fluid.append(args)
                else:
                    self.boundary_kernels_fluid.append(None)
                    self.boundary_args_fluid.append(None)

        if state.phase:
            self.boundary_kernels_phase = []
            self.boundary_args_phase = []
            setup_boundary_phase = SetupBoundaryPhase(
                self.boundary_condition_type["phase"],
                backend
            )
            for boundary_element in state.boundary.boundary_elements:
                if boundary_element.type_fluid != "periodic":
                    setup_function = getattr(
                        setup_boundary_phase,
                        "setup_" + boundary_element.type_phase
                    )
                    kernel, args = setup_function(state, boundary_element)
                    self.boundary_kernels_phase.append(kernel)
                    self.boundary_args_phase.append(args)
                else:
                    self.boundary_kernels_phase.append(None)
                    self.boundary_args_phase.append(None)

        print_log("Backend set for boundary operator",
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
        if state.fluid:
            for kernel_name in self.kernel_signatures["fluid"]:
                kernel = getattr(fluid_kernels_cpu, kernel_name)
                if (set(kernel.signatures) !=
                        self.kernel_signatures["fluid"][kernel_name]):
                    raise RuntimeError(
                        f"Developer error! {kernel_name}: fluid in"
                        f" boundary operator compiled a new signature!"
                    )

        if state.phase:
            for kernel_name in self.kernel_signatures["phase"]:
                kernel = getattr(phase_kernels_cpu, kernel_name)
                if (set(kernel.signatures) !=
                        self.kernel_signatures["phase"][kernel_name]):
                    raise RuntimeError(
                        f"Developer error! {kernel_name}: phase in"
                        f" boundary operator compiled a new signature!"
                    )

        print_log("Kernel signatures verified for boundary operator",
                  state.domain.mpi_rank, verbose)


def make_args(args_dict, state, boundary_element, backend_suffix):
    """
    Make kernel args tuple from given args_dict
    Args:

    Returns:

    """
    args = ()
    for item in args_dict:
        if item == "state":
            for sub_item in args_dict["state"]:
                attr = getattr(state, sub_item)
                key_args = tuple(
                    getattr(attr, attr_name + backend_suffix)
                    for attr_name in args_dict["state"][sub_item]
                )
                args += key_args
        elif item == "boundary_element":
            key_args = tuple(
                getattr(boundary_element, attr_name + backend_suffix)
                for attr_name in args_dict["boundary_element"]
            )
            args += key_args
    return args


class SetupBoundaryFluid:
    def __init__(self, boundary_condition_type_fluid, backend):
        self.boundary_condition_type_fluid = boundary_condition_type_fluid
        self.backend_type = backend.backend_type
        if self.backend_type == "cpu":
            self.backend_suffix = ""
        elif self.backend_type == "gpu":
            self.backend_suffix = "_device"

    def setup_bounce_back(self, state, boundary_element):
        if self.backend_type == "cpu":
            kernel = getattr(fluid_kernels_cpu, "bounce_back")
        elif self.backend_type == "gpu":
            kernel = getattr(fluid_kernels_gpu, "bounce_back")
        args_dict = {
            "state": {
                "fields": ["solid", "pop_fluid", "pop_fluid_new"]
            },
            "boundary_element": ["boundary_nodes", "out_list", "inv_list"]
        }
        args = make_args(
            args_dict,
            state,
            boundary_element,
            self.backend_suffix
        )
        return kernel, args

    def setup_fixed_velocity(self, state, boundary_element):
        if self.backend_type == "cpu":
            kernel = getattr(
                fluid_kernels_cpu,
                "fixed_velocity_" + self.boundary_condition_type_fluid
            )
        elif self.backend_type == "gpu":
            kernel = getattr(
                fluid_kernels_gpu,
                "fixed_velocity_" + self.boundary_condition_type_fluid
            )
        if self.boundary_condition_type_fluid == "density_based":
            args_dict = {
                "state": {
                    "lattice": ["cx", "cy", "weights", "inv_cs_2"],
                    "fields": ["solid", "density", "pop_fluid",
                               "pop_fluid_new"]
                },
                "boundary_element": [
                    "boundary_nodes", "out_list", "inv_list",
                    "vector_fluid"
                ]
            }
        else:
            args_dict = {
                "state": {
                    "lattice": ["cx", "cy", "weights", "inv_cs_2"],
                    "fields": ["solid", "density", "pop_fluid",
                               "pop_fluid_new"]
                },
                "boundary_element": [
                    "boundary_nodes", "out_list", "inv_list",
                    "vector_fluid"
                ]
            }
        args = make_args(
            args_dict,
            state,
            boundary_element,
            self.backend_suffix
        )
        return kernel, args

    def setup_fixed_pressure(self, state, boundary_element):
        if self.backend_type == "cpu":
            kernel = getattr(
                fluid_kernels_cpu,
                "fixed_pressure_" + self.boundary_condition_type_fluid
            )
        elif self.backend_type == "gpu":
            kernel = getattr(
                fluid_kernels_gpu,
                "fixed_pressure_" + self.boundary_condition_type_fluid
            )
        args_dict = {
            "state": {
                "domain": ["shape"],
                "lattice": ["cx", "cy", "weights", "inv_cs_2", "inv_cs_4"],
                "fields": ["solid", "velocity", "pop_fluid",
                           "pop_fluid_new"]
            },
            "boundary_element": [
                "boundary_nodes", "out_list", "inv_list",
                "scalar_fluid", "surface_normals"
            ]
        }
        args = make_args(
            args_dict,
            state,
            boundary_element,
            self.backend_suffix
        )
        return kernel, args


class SetupBoundaryPhase:
    def __init__(self, boundary_condition_type_phase, backend):
        self.boundary_condition_type_phase = boundary_condition_type_phase
        self.backend_type = backend.backend_type

    def setup_bounce_back(self, state, boundary_element):
        if self.backend_type == "cpu":
            kernel = getattr(phase_kernels_cpu, "bounce_back")
        elif self.backend_type == "gpu":
            kernel = getattr(phase_kernels_gpu, "bounce_back")
        args_dict = {
            "state": {
                "fields": ["solid", "pop_phase", "pop_phase_new"]
            },
            "boundary_element": [
                "boundary_nodes", "out_list", "inv_list"
            ]
        }
        args = make_args(
            args_dict,
            state,
            boundary_element,
            self.backend_suffix
        )
        return kernel, args
