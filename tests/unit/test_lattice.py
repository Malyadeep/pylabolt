import numpy as np
import pytest

from pylabolt.base.lattice import Lattice
from factories import make_control, make_mesh, make_simulation


"""
Lattice test
"""


def test_missing_lattice_dict():
    control = make_control()
    mesh = make_mesh((10, 10))

    simulation = make_simulation()

    mssg = "lattice_dict not found in simulation.py file"
    with pytest.raises(ValueError, match=mssg):
        Lattice(
            simulation,
            control,
            mesh,
            0,
            verbose=False
        )


def test_missing_lattice_type():
    control = make_control()
    mesh = make_mesh((10, 10))

    simulation = make_simulation(lattice_dict={})

    mssg = "lattice_type missing in lattice_dict"
    with pytest.raises(ValueError, match=mssg):
        Lattice(
            simulation,
            control,
            mesh,
            0,
            verbose=False
        )


@pytest.mark.parametrize(
    "global_shape, lattice_type",
    [[(10, 20), "D1Q3"],
     [(10, 1), "D2Q9"]]
)
def test_lattice_mesh_mismatch(global_shape, lattice_type):
    control = make_control()
    mesh = make_mesh(global_shape)

    lattice_dict = {
        "lattice_type": lattice_type
    }

    simulation = make_simulation(lattice_dict=lattice_dict)

    mssg = "grid dimensions and lattice type are incompatible"
    with pytest.raises(ValueError, match=mssg):
        Lattice(
            simulation,
            control,
            mesh,
            0,
            verbose=False
        )


def test_unsupported_lattice_type():
    control = make_control()
    mesh = make_mesh((10, 10))

    lattice_dict = {
        "lattice_type": "D1Q9"
    }

    simulation = make_simulation(lattice_dict=lattice_dict)

    mssg = "Unsupported lattice type"
    with pytest.raises(ValueError, match=mssg):
        Lattice(
            simulation,
            control,
            mesh,
            0,
            verbose=False
        )


def test_d2q9_entries():
    control = make_control()
    mesh = make_mesh((10, 10))

    lattice_dict = {
        "lattice_type": "D2Q9"
    }

    simulation = make_simulation(lattice_dict=lattice_dict)

    lattice = Lattice(
        simulation,
        control,
        mesh,
        0,
        verbose=False
    )

    cs = control.precision(1/np.sqrt(3))
    assert np.isclose(cs, lattice.cs)
    assert (lattice.cs.dtype == control.precision)
    cs_2 = cs * cs
    assert np.isclose(cs_2, lattice.cs_2)
    inv_cs_2 = 1 / cs_2
    assert np.isclose(inv_cs_2, lattice.inv_cs_2)
    inv_cs_4 = inv_cs_2 * inv_cs_2
    assert np.isclose(inv_cs_4, lattice.inv_cs_4)
    cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
    cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
    assert np.all(cx == lattice.cx)
    assert np.all(cy == lattice.cy)
    weights = np.array(
        [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36],
        dtype=control.precision
    )
    assert np.all(weights == lattice.weights)
    inv_list = np.array(
        [0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32
    )
    assert np.all(inv_list == lattice.inv_list)
    no_of_directions = np.int32(9)
    assert no_of_directions == lattice.no_of_directions


def test_d1q3_entries():
    control = make_control()
    mesh = make_mesh((10, 1))

    lattice_dict = {
        "lattice_type": "D1Q3"
    }

    simulation = make_simulation(lattice_dict=lattice_dict)

    lattice = Lattice(
        simulation,
        control,
        mesh,
        0,
        verbose=False
    )

    cs = control.precision(1/np.sqrt(3))
    assert np.isclose(cs, lattice.cs)
    assert (lattice.cs.dtype == control.precision)
    cs_2 = cs * cs
    assert np.isclose(cs_2, lattice.cs_2)
    inv_cs_2 = 1 / cs_2
    assert np.isclose(inv_cs_2, lattice.inv_cs_2)
    inv_cs_4 = inv_cs_2 * inv_cs_2
    assert np.isclose(inv_cs_4, lattice.inv_cs_4)
    cx = np.array([0, 1, -1], dtype=np.int32)
    cy = np.array([0, 0, 0], dtype=np.int32)
    assert np.all(cx == lattice.cx)
    assert np.all(cy == lattice.cy)
    weights = np.array([2/3, 1/6, 1/6], dtype=control.precision)
    assert np.all(weights == lattice.weights)
    inv_list = np.array([0, 2, 1], dtype=np.int32)
    assert np.all(inv_list == lattice.inv_list)
    no_of_directions = np.int32(3)
    assert no_of_directions == lattice.no_of_directions
