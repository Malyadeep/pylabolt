from pylabolt.utils.helpers import print_log
from pylabolt.parallel.cpu import fluid_boundary_kernels as fluid_kernels_cpu
from pylabolt.parallel.cpu import phase_boundary_kernels as phase_kernels_cpu
# from pylabolt.parallel.gpu import collision_kernels as collision_kernels_gpu


class BoundaryOperator:
    def __init__(
        self,
        model,
        state,
        backend,
        verbose=True
    ):
        """
        Boundary condition operator
        Attributes:

        """
        print_log("-" * 80, state.domain.mpi_rank, verbose)
        print_log("Setting up boundary condition operator...\n",
                  state.domain.mpi_rank, verbose)
        self.boundary_conditions = []
        self.boundary_args = []
        self.set_backend(model, state, backend)
        print_log("\nSetting up boundary condition operator done!",
                  state.domain.mpi_rank, verbose)
        print_log("-" * 80, state.domain.mpi_rank, verbose)

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
        model,
        state,
        backend
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

        self.boundary_condition_type = model.boundary_condition_type

        if state.fluid:
            self.boundary_kernels_fluid = []
            self.boundary_args_fluid = []
            setup_boundary_fluid = SetupBoundaryFluid(
                self.boundary_condition_type["fluid"]
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
                self.boundary_condition_type["phase"]
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


class SetupBoundaryFluid:
    def __init__(self, boundary_condition_type_fluid):
        self.boundary_condition_type_fluid = boundary_condition_type_fluid

    def setup_bounce_back(self, state, boundary_element):
        kernel = getattr(fluid_kernels_cpu, "bounce_back")
        args = (
            boundary_element.boundary_nodes,
            boundary_element.out_list,
            boundary_element.inv_list,
            state.fields.solid,
            state.fields.pop_fluid,
            state.fields.pop_fluid_new
        )
        return kernel, args

    def setup_fixed_velocity(self, state, boundary_element):
        kernel = getattr(
            fluid_kernels_cpu,
            "fixed_velocity_" + self.boundary_condition_type_fluid
        )
        if self.boundary_condition_type_fluid == "density_based":
            args = (
                state.domain.shape,
                state.lattice.cx,
                state.lattice.cy,
                state.lattice.weights,
                state.lattice.inv_cs_2,
                boundary_element.boundary_nodes,
                boundary_element.out_list,
                boundary_element.inv_list,
                state.fields.solid,
                state.fields.density,
                state.fields.velocity,
                state.fields.pop_fluid,
                state.fields.pop_fluid_new
            )
        elif self.boundary_condition_type_fluid == "no_density_based":
            args = (
                state.domain.shape,
                state.lattice.cx,
                state.lattice.cy,
                state.lattice.weights,
                state.lattice.inv_cs_2,
                boundary_element.boundary_nodes,
                boundary_element.out_list,
                boundary_element.inv_list,
                state.fields.solid,
                state.fields.velocity,
                state.fields.pop_fluid,
                state.fields.pop_fluid_new
            )
        return kernel, args

    def setup_fixed_pressure(self, state, boundary_element):
        kernel = getattr(
            fluid_kernels_cpu,
            "fixed_pressure_" + self.boundary_condition_type_fluid
        )
        if self.boundary_condition_type_fluid == "density_based":
            args = (
                state.domain.shape,
                state.lattice.cx,
                state.lattice.cy,
                state.lattice.weights,
                state.lattice.inv_cs_2,
                state.lattice.inv_cs_4,
                boundary_element.boundary_nodes,
                boundary_element.out_list,
                boundary_element.inv_list,
                boundary_element.surface_normals,
                state.fields.solid,
                state.fields.density,
                state.fields.velocity,
                state.fields.pop_fluid,
                state.fields.pop_fluid_new
            )
        elif self.boundary_condition_type_fluid == "no_density_based":
            args = (
                state.domain.shape,
                state.lattice.cx,
                state.lattice.cy,
                state.lattice.weights,
                state.lattice.inv_cs_2,
                state.lattice.inv_cs_4,
                boundary_element.boundary_nodes,
                boundary_element.out_list,
                boundary_element.inv_list,
                boundary_element.surface_normals,
                state.fields.solid,
                state.fields.pressure,
                state.fields.velocity,
                state.fields.pop_fluid,
                state.fields.pop_fluid_new
            )
        return kernel, args


class SetupBoundaryPhase:
    def __init__(self, boundary_condition_type_phase):
        self.boundary_condition_type_phase = boundary_condition_type_phase

    def setup_bounce_back(self, state, boundary_element):
        kernel = getattr(phase_kernels_cpu, "bounce_back")
        args = (
            boundary_element.boundary_nodes,
            boundary_element.out_list,
            boundary_element.inv_list,
            state.fields.solid,
            state.fields.pop_phase,
            state.fields.pop_phase_new
        )
        return kernel, args
