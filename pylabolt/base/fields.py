import numpy as np


class Fields:
    def __init__(
        self,
        control,
        lattice,
        domain,
        fluid=False,  # True if fluid physics is solved
        phase=False,  # True if multiphase physics is solved
        scalar=False  # True if scalar transport is solved
    ):
        """
        Container for time evolving fields of the system
        Attributes (common):
            solid: (Nx * Ny, 2) int array
            fluid_boundary: (Nx * Ny) bool array
            solid_boundary: (Nx * Ny) bool array
            ghost_node: (Nx * Ny) bool array
            periodic_boundary: (Nx * Ny) bool array
        Attributes (fluid = True):
            velocity: (Nx * Ny, 2) float array
            density: (Nx * Ny) float array
            pressure: (Nx * Ny) float array
            force_field: (Nx * Ny, 2) float array
            pop_fluid: (Nx * Ny, lattice.no_of_directions) float array
            pop_fluid_new: (Nx * Ny, lattice.no_of_directions) float array
            pop_fluid_eq: (Nx * Ny, lattice.no_of_directions) float array
        Attributes (phase = True):
            phase_field: (Nx * Ny) float array
            grad_phase_field: (Nx * Ny, 2) float array
            lap_phase_field: (Nx * Ny) float array
            surface_tension_force: (Nx * Ny, 2) float array
            pressure_corr_force: (Nx * Ny, 2) float array
            viscous_corr_force: (Nx * Ny, 2) float array
            curvature: (Nx * Ny) float array
            phase_field_solid: (Nx * Ny) float array
            pop_phase: (Nx * Ny, lattice.no_of_directions) float array
            pop_phase_new: (Nx * Ny, lattice.no_of_directions) float array
            pop_phase_eq: (Nx * Ny, lattice.no_of_directions) float array
        Attributes (scalar = True):
            --TODO-- no scalar field solver present yet
        """
        self.fluid = fluid
        self.phase = phase
        self.scalar = scalar
        dim = len(domain.shape)
        # ------- Geometry ------- #
        self.solid = self.allocate_memory(domain.size, np.bool_)
        self.solid_id = self.allocate_memory(domain.size, int)
        self.solid_id[:] = -1
        self.solid_boundary = self.allocate_memory(domain.size, np.bool_)
        self.fluid_boundary = self.allocate_memory(domain.size, np.bool_)
        self.surface_normals = self.allocate_memory(
            domain.size,
            control.precision,
            components=dim
        )
        self.ghost_node = self.allocate_memory(domain.size, np.bool_)
        self.init_ghost_nodes(domain)
        self.periodic_boundary = self.allocate_memory(domain.size, np.bool_)
        # ------- Fluid physics ------- #
        if self.fluid:
            self.velocity = self.allocate_memory(
                domain.size,
                control.precision,
                components=dim
            )
            self.density = self.allocate_memory(
                domain.size,
                control.precision
            )
            self.pressure = self.allocate_memory(
                domain.size,
                control.precision
            )
            self.force_field = self.allocate_memory(
                domain.size,
                control.precision,
                components=dim
            )
            self.pop_fluid = self.allocate_memory(
                domain.size,
                control.precision,
                components=lattice.no_of_directions
            )
            self.pop_fluid_new = self.allocate_memory(
                domain.size,
                control.precision,
                components=lattice.no_of_directions
            )
        # ------- Phase physics ------- #
        if self.phase:
            self.phase_field = self.allocate_memory(
                domain.size,
                control.precision
            )
            self.grad_phase_field = self.allocate_memory(
                domain.size,
                control.precision,
                components=dim
            )
            self.lap_phase_field = self.allocate_memory(
                domain.size,
                control.precision
            )
            self.surface_tension_force = self.allocate_memory(
                domain.size,
                control.precision,
                components=dim
            )
            self.pressure_corr_force = self.allocate_memory(
                domain.size,
                control.precision,
                components=dim
            )
            self.viscous_corr_force = self.allocate_memory(
                domain.size,
                control.precision,
                components=dim
            )
            self.curvature = self.allocate_memory(
                domain.size,
                control.precision
            )
            self.phase_field_solid = self.allocate_memory(
                domain.size,
                control.precision
            )
            self.pop_phase = self.allocate_memory(
                domain.size,
                control.precision,
                components=lattice.no_of_directions
            )
            self.pop_phase_new = self.allocate_memory(
                domain.size,
                control.precision,
                components=lattice.no_of_directions
            )
        # ------- Phase physics ------- #
        if self.scalar:
            # --TODO-- no scalar field solver present yet
            pass

    def allocate_memory(
        self,
        size,
        dtype,
        components=None
    ):
        """
        Allocates memory and returns numpy array
        Args;
            size: int - domain size (Nx * Ny)
            dtype: data-type
            components: int, default=None
        Returns:
            array (size, components), dtype=dtype
        """
        if components is None:
            return np.zeros((size,), dtype=dtype)
        else:
            return np.zeros((size, components), dtype=dtype)

    def init_ghost_nodes(self, domain):
        """
        Initializes ghost node markers
        Args:
            domain: domain object
        Returns:
            None
        """
        Nx, Ny = domain.shape
        for ind in range(domain.size):
            i = ind // domain.shape[1]
            j = ind - i * domain.shape[1]
            if i == 0 or j == 0 or i == Nx - 1 or j == Ny - 1:
                self.ghost_node[ind] = True

    def set_backend(
        self,
        backend
    ):
        """
        Configure backend attributes for fields object
        Args:

        Returns:

        """
        if backend.backend_type == "gpu":
            self._device_attrs = [
                "solid",
                "solid_id",
                "solid_boundary",
                "fluid_boundary",
                "ghost_node",
                "periodic_boundary"
            ]
            if self.fluid:
                self._device_attrs.extend([
                    "velocity",
                    "density",
                    "pressure",
                    "force_field",
                    "pop_fluid",
                    "pop_fluid_new"
                ])
            if self.phase:
                self._device_attrs.extend([
                    "phase_field",
                    "grad_phase_field",
                    "lap_phase_field",
                    "surface_tension_force",
                    "pressure_corr_force",
                    "viscous_corr_force",
                    "curvature",
                    "phase_field_solid",
                    "pop_phase",
                    "pop_phase_new"
                ])
            for arg_name in self._device_attrs:
                arg_device = backend.allocate_to_device(
                    getattr(self, arg_name)
                )
                setattr(self, arg_name + "_device", arg_device)
