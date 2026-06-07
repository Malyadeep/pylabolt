import numpy as np
import pytest
import re

from pylabolt.parallel.domain import Domain, local_to_global, global_to_local
from factories import make_decompose_dict, make_mesh, make_comm, make_simulation

"""
Base fixtures
"""


GLOBAL_SHAPES = [
    (50, 40),
    (70, 65)
]

DECOMPOSE_PARAMS = [
    (1, 1),
    (2, 1),
    (1, 2),
    (3, 2),
    (2, 3),
    (3, 3)
]

COMBINED_PARAMS = [
    (*d, s)
    for d in DECOMPOSE_PARAMS
    for s in GLOBAL_SHAPES
]


"""
Derived fixtures
"""


def make_domain(
    mpi_rank,
    mpi_size,
    dummy_simulation,
    dummy_mesh
):
    comm = make_comm(mpi_rank, mpi_size)
    return Domain(
        dummy_simulation,
        dummy_mesh,
        comm,
        verbose=False
    )


"""
Invalid decompsition tests
"""


def test_missing_decompose_dict():
    mesh = make_mesh((10, 10))
    comm = make_comm(1, 0)
    simulation = make_simulation()

    mssg = "decompose_dict not found in simulation.py file"
    with pytest.raises(ValueError, match=mssg):
        Domain(
            simulation,
            mesh,
            comm,
            verbose=False
        )


def test_missing_nx_ny():
    mesh = make_mesh((10, 10))
    comm = make_comm(1, 0)

    decompose_dict = {
        "ny": 2
    }
    simulation = make_simulation(decompose_dict=decompose_dict)

    mssg = "nx or ny missing decompose_dict"
    with pytest.raises(ValueError, match=mssg):
        Domain(
            simulation,
            mesh,
            comm,
            verbose=False
        )


def test_nx_ny_size_mismatch():
    mesh = make_mesh((10, 10))
    comm = make_comm(4, 0)

    decompose_dict = make_decompose_dict(3, 2)
    simulation = make_simulation(decompose_dict=decompose_dict)

    mssg = "invalid domain decomposition. " +\
        "nx * ny not equal to total no.of MPI processes"
    with pytest.raises(ValueError, match=re.escape(mssg)):
        Domain(
            simulation,
            mesh,
            comm,
            verbose=False
        )


"""
Test decomposition consistency
"""


@pytest.mark.parametrize(
    "nx, ny, global_shape",
    COMBINED_PARAMS
)
def test_decompose_logic(nx, ny, global_shape):
    mpi_size = nx * ny
    dummy_mesh = make_mesh(global_shape)
    decompose_dict = make_decompose_dict(nx, ny)
    simulation = make_simulation(decompose_dict=decompose_dict)

    for mpi_rank in range(mpi_size):
        domain = make_domain(
            mpi_rank,
            mpi_size,
            simulation,
            dummy_mesh
        )

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


@pytest.mark.parametrize(
    "nx, ny, global_shape",
    COMBINED_PARAMS
)
def test_domain_coverage(nx, ny, global_shape):
    mpi_size = nx * ny
    dummy_mesh = make_mesh(global_shape)
    decompose_dict = make_decompose_dict(nx, ny)
    simulation = make_simulation(decompose_dict=decompose_dict)

    coverage = np.zeros(
        dummy_mesh.grid_global_shape, dtype=np.int32
    )
    for mpi_rank in range(mpi_size):
        domain = make_domain(
            mpi_rank,
            mpi_size,
            simulation,
            dummy_mesh
        )

        assert domain.Nx_rank > 0
        assert domain.Ny_rank > 0
        x0, y0 = domain.offset
        x1 = x0 + domain.Nx_rank
        y1 = y0 + domain.Ny_rank
        coverage[x0:x1, y0:y1] += 1
    assert np.all(coverage == 1)


@pytest.mark.parametrize(
    "nx, ny, global_shape",
    COMBINED_PARAMS
)
def test_local_global_mapping(nx, ny, global_shape):
    mpi_size = nx * ny
    dummy_mesh = make_mesh(global_shape)
    decompose_dict = make_decompose_dict(nx, ny)
    simulation = make_simulation(decompose_dict=decompose_dict)

    for mpi_rank in range(mpi_size):
        domain = make_domain(
            mpi_rank,
            mpi_size,
            simulation,
            dummy_mesh
        )
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
