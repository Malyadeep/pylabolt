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
        dim = len(domain.shape)
        # ------- Geometry ------- #
        self.solid = self.allocate_memory(domain.size, np.int32, components=2)
        self.solid_boundary = self.allocate_memory(domain.size, np.bool_)
        self.fluid_boundary = self.allocate_memory(domain.size, np.bool_)
        self.ghost_node = self.allocate_memory(domain.size, np.bool_)
        self.init_ghost_nodes(domain)
        self.periodic_boundary = self.allocate_memory(domain.size, np.bool_)
        # ------- Fluid physics ------- #
        if fluid is True:
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
            self.pop_fluid_eq = self.allocate_memory(
                domain.size,
                control.precision,
                components=lattice.no_of_directions
            )
        # ------- Phase physics ------- #
        if phase is True:
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
            self.pop_phase_eq = self.allocate_memory(
                domain.size,
                control.precision,
                components=lattice.no_of_directions
            )
        # ------- Phase physics ------- #
        if scalar is True:
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
            size: np.int32 - domain size (Nx * Ny)
            dtype: data-type
            components: np.int32, default=None
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
