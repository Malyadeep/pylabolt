import numpy as np
from types import SimpleNamespace


def make_control():
    return SimpleNamespace(
        precision=np.float64
    )


def make_mesh(global_shape):
    if 1 in global_shape:
        dimensions = 1
    else:
        dimensions = 2
    return SimpleNamespace(
        grid_global_shape=np.array(global_shape, dtype=np.int32),
        dimensions=dimensions
    )


def make_decompose_dict(nx, ny):
    return {
        "nx": nx,
        "ny": ny
    }


def make_domain(Nx, Ny, mpi_rank):
    return SimpleNamespace(
        mpi_rank=mpi_rank,
        size=Nx * Ny,
        Nx=Nx,
        Ny=Ny,
        shape=np.array([Nx, Ny], dtype=np.int32),
        offset=np.array([0, 0])
    )


def make_lattice(no_of_directions):
    return SimpleNamespace(
        no_of_directions=no_of_directions
    )


class DummyFields:
    def __init__(self, domain, control):
        self.velocity = np.zeros((domain.size, 2), dtype=control.precision)
        self.pressure = np.zeros(domain.size, dtype=control.precision)
        self.density = np.zeros(domain.size, dtype=control.precision)
        self.phase_field = np.zeros(domain.size, dtype=control.precision)
        self.ghost_node = np.zeros(domain.size, dtype=np.bool_)
        for ind in range(domain.size):
            i = ind // domain.shape[1]
            j = ind - i * domain.shape[1]
            if (i == 0 or j == 0 or i == domain.shape[0] - 1 or
                    j == domain.shape[1] - 1):
                self.ghost_node[ind] = True


def make_fields(domain, control):
    return DummyFields(domain, control)


class DummyComm:
    def __init__(self, mpi_rank, mpi_size):
        self.mpi_size = mpi_size
        self.mpi_rank = mpi_rank

    def Get_rank(self):
        return self.mpi_rank

    def Get_size(self):
        return self.mpi_size


def make_comm(mpi_size, mpi_rank):
    return DummyComm(mpi_size, mpi_rank)


def make_simulation(
    control_dict=None,
    mesh_dict=None,
    lattice_dict=None,
    decompose_dict=None,
    boundary_dict=None,
    initial_fields_dict=None
):
    data = {}
    if control_dict is not None:
        data["control_dict"] = control_dict
    if mesh_dict is not None:
        data["mesh_dict"] = mesh_dict
    if decompose_dict is not None:
        data["decompose_dict"] = decompose_dict
    if boundary_dict is not None:
        data["boundary_dict"] = boundary_dict
    if lattice_dict is not None:
        data["lattice_dict"] = lattice_dict
    if initial_fields_dict is not None:
        data["initial_fields_dict"] = initial_fields_dict
    return SimpleNamespace(**data)


class BoundaryDictsFluid:
    def __init__(self):
        self.sample_dict = {
            "options": {},
            "check": {
                "fluid": {
                    "type": "bounce_back"
                },
                "segments": [
                    [[0, 20], [20, 20]],
                    [[0, 0], [20, 0]],
                    [[0, 0], [0, 20]],
                    [[20, 0], [20, 20]]
                ]
            }
        }

    def get_bounce_back_dict(self):
        sample_dict = self.sample_dict
        sample_dict["check"]["fluid"]["type"] = "bounce_back"
        return sample_dict

    def get_fixed_velocity_dict(self):
        sample_dict = self.sample_dict
        sample_dict["check"]["fluid"]["type"] = "fixed_velocity"
        sample_dict["check"]["fluid"]["value"] = [1e-3, 1e-5]
        return sample_dict

    def get_fixed_pressure_dict(self):
        sample_dict = self.sample_dict
        sample_dict["check"]["fluid"]["type"] = "fixed_pressure"
        sample_dict["check"]["fluid"]["value"] = 0.35
        return sample_dict

    def get_periodic_dict(self, Nx, Ny, orientation="vertical"):
        sample_dict = self.sample_dict
        sample_dict["check"]["fluid"]["type"] = "periodic"
        if orientation == "vertical":
            sample_dict["check"]["segments"] = [
                [[0, 0], [0, Ny - 1]],
                [[Nx - 1, 0], [Nx - 1, Ny - 1]]
            ]
        elif orientation == "horizontal":
            sample_dict["check"]["segments"] = [
                [[0, 0], [Nx - 1, 0]],
                [[0, Ny - 1], [Nx - 1, Ny - 1]]
            ]
        return sample_dict


class BoundaryDictsPhase:
    def __init__(self):
        self.sample_dict = {
            "options": {},
            "check": {
                "phase": {
                    "type": "bounce_back"
                },
                "segments": [
                    [[0, 20], [20, 20]],
                    [[0, 0], [20, 0]],
                    [[0, 0], [0, 20]],
                    [[20, 0], [20, 20]]
                ]
            }
        }

    def get_bounce_back_dict(self):
        sample_dict = self.sample_dict
        sample_dict["check"]["phase"]["type"] = "bounce_back"
        return sample_dict

    def get_fixed_value_dict(self):
        sample_dict = self.sample_dict
        sample_dict["check"]["phase"]["type"] = "fixed_value"
        sample_dict["check"]["phase"]["value"] = 0.35
        return sample_dict

    def get_periodic_dict(self, Nx, Ny, orientation="vertical"):
        sample_dict = self.sample_dict
        sample_dict["check"]["phase"]["type"] = "periodic"
        if orientation == "vertical":
            sample_dict["check"]["segments"] = [
                [[0, 0], [0, Ny - 1]],
                [[Nx - 1, 0], [Nx - 1, Ny - 1]]
            ]
        elif orientation == "horizontal":
            sample_dict["check"]["segments"] = [
                [[0, 0], [Nx - 1, 0]],
                [[0, Ny - 1], [Nx - 1, Ny - 1]]
            ]
        return sample_dict
