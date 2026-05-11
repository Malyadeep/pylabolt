import numpy as np
import pytest
import re

from pylabolt.backend.domain import Domain, local_to_global, global_to_local


"""
Base fixtures
"""


@pytest.fixture(
    params=[(50, 40), (70, 65)]
)
def global_shape(request):
    return np.array(request.param, dtype=np.int32)


@pytest.fixture(
    params=[(1, 1), (2, 1), (1, 2), (3, 2), (2, 3), (3, 3)]
)
def decompose(request):
    nx, ny = request.param
    return {"nx": nx, "ny": ny}


"""
Derived fixtures
"""


@pytest.fixture
def dummy_simulation(decompose):
    class Simulation:
        decompose_dict = decompose
    return Simulation()


@pytest.fixture
def dummy_mesh(global_shape):
    class Mesh:
        def __init__(self):
            self.grid_global_shape = global_shape
    return Mesh()


@pytest.fixture
def make_comm():
    class DummyComm:
        def __init__(self, mpi_rank, mpi_size):
            self.mpi_size = mpi_size
            self.mpi_rank = mpi_rank

        def Get_rank(self):
            return self.mpi_rank

        def Get_size(self):
            return self.mpi_size

    return DummyComm


@pytest.fixture
def make_domain(dummy_simulation, dummy_mesh, make_comm):
    def _make(mpi_rank, mpi_size):
        comm = make_comm(mpi_rank, mpi_size)
        return Domain(
            dummy_simulation,
            dummy_mesh,
            comm,
            verbose=False
        )
    return _make


"""
Invalid decompsition tests
"""


def test_missing_decompose_dict(dummy_mesh, make_comm):
    mesh = dummy_mesh
    comm = make_comm(1, 0)

    class Simulation:
        pass

    mssg = "decompose_dict not found in simulation.py file"
    with pytest.raises(ValueError, match=mssg):
        Domain(
            Simulation(),
            mesh,
            comm,
            verbose=False
        )


def test_missing_nx_ny(dummy_mesh, make_comm):
    mesh = dummy_mesh
    comm = make_comm(1, 0)

    class Simulation:
        decompose_dict = {
            "ny": 2
        }

    mssg = "nx or ny missing decompose_dict"
    with pytest.raises(ValueError, match=mssg):
        Domain(
            Simulation(),
            mesh,
            comm,
            verbose=False
        )


def test_nx_ny_size_mismatch(dummy_mesh, make_comm):
    mesh = dummy_mesh
    comm = make_comm(4, 0)

    class Simulation:
        decompose_dict = {
            "nx": 3,
            "ny": 2
        }

    mssg = "invalid domain decomposition. " +\
        "nx * ny not equal to total no.of MPI processes"
    with pytest.raises(ValueError, match=re.escape(mssg)):
        Domain(
            Simulation(),
            mesh,
            comm,
            verbose=False
        )


"""
Test decomposition consistency
"""


def test_decompose_logic(make_domain, dummy_simulation, dummy_mesh):
    nx = dummy_simulation.decompose_dict["nx"]
    ny = dummy_simulation.decompose_dict["ny"]
    mpi_size = nx * ny

    for mpi_rank in range(mpi_size):
        domain = make_domain(mpi_rank, mpi_size)

        assert domain.mpi_rank == mpi_rank
        assert domain.mpi_size == mpi_size
        assert domain.Nx_proc == nx
        assert domain.Ny_proc == ny

        i_proc_exp = mpi_rank // ny
        j_proc_exp = mpi_rank - ny * i_proc_exp
        assert domain.i_proc == i_proc_exp
        assert domain.j_proc == j_proc_exp

        offset = np.zeros(len(dummy_mesh.grid_global_shape), dtype=np.int32)
        if i_proc_exp != nx - 1:
            Nx_rank_exp = np.ceil(
                dummy_mesh.grid_global_shape[0] / nx
            )
        else:
            Nx_rank_exp = dummy_mesh.grid_global_shape[0] -\
                i_proc_exp * np.ceil(
                    dummy_mesh.grid_global_shape[0] / nx
                )
        Nx_rank_exp = int(Nx_rank_exp)
        offset[0] = i_proc_exp * np.ceil(
            dummy_mesh.grid_global_shape[0] / nx
        )
        if j_proc_exp != ny - 1:
            Ny_rank_exp = np.ceil(
                dummy_mesh.grid_global_shape[1] / ny
            )
            Ny_rank_exp = int(Ny_rank_exp)
        else:
            Ny_rank_exp = dummy_mesh.grid_global_shape[1] -\
                j_proc_exp * np.ceil(
                    dummy_mesh.grid_global_shape[1] / ny
                )
        Ny_rank_exp = int(Ny_rank_exp)
        offset[1] = j_proc_exp * np.ceil(
            dummy_mesh.grid_global_shape[1] / ny
        )
        Nx_pad_exp = Nx_rank_exp + 2
        Ny_pad_exp = Ny_rank_exp + 2
        shape_exp = np.array([Nx_pad_exp, Ny_pad_exp])
        size_exp = np.prod(shape_exp)
        assert Nx_rank_exp == domain.Nx_rank
        assert Ny_rank_exp == domain.Ny_rank
        assert Nx_pad_exp == domain.Nx_pad
        assert Ny_pad_exp == domain.Ny_pad
        assert size_exp == domain.size
        assert np.all(shape_exp == domain.shape)


def test_domain_coverage(make_domain, dummy_simulation, dummy_mesh):
    nx = dummy_simulation.decompose_dict["nx"]
    ny = dummy_simulation.decompose_dict["ny"]
    mpi_size = nx * ny

    coverage = np.zeros(
        dummy_mesh.grid_global_shape, dtype=np.int32
    )
    for mpi_rank in range(mpi_size):
        domain = make_domain(mpi_rank, mpi_size)

        assert domain.Nx_rank > 0
        assert domain.Ny_rank > 0
        x0, y0 = domain.offset
        x1 = x0 + domain.Nx_rank
        y1 = y0 + domain.Ny_rank
        coverage[x0:x1, y0:y1] += 1
    assert np.all(coverage == 1)


def test_local_global_mapping(make_domain, dummy_simulation):
    nx = dummy_simulation.decompose_dict["nx"]
    ny = dummy_simulation.decompose_dict["ny"]
    mpi_size = nx * ny

    for mpi_rank in range(mpi_size):
        domain = make_domain(mpi_rank, mpi_size)
        offset = domain.offset
        for i in range(domain.Nx_rank):
            for j in range(domain.Ny_rank):
                # local - global - local
                i_g, j_g = local_to_global(i, j, offset)
                i_l, j_l = global_to_local(i_g, j_g, offset)
                assert i == i_l
                assert j == j_l

                # check global indices
                assert i_g == i + offset[0]
                assert j_g == j + offset[1]

                # recheck local to global
                i_g2, j_g2 = local_to_global(i_l, j_l, offset)
                assert i_g == i_g2
                assert j_g == j_g2
