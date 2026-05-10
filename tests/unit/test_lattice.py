import numpy as np
import pytest

from pylabolt.base.lattice import Lattice


"""
Dummy classes
"""


class DummyControl:
    def __init__(self):
        self.precision = np.float64


class DummyMesh:
    def __init__(self, dimensions=1):
        self.dimensions = dimensions


"""
Lattice test
"""


def test_missing_lattice_dict():
    control = DummyControl()
    mesh = DummyMesh(dimensions=1)

    class Simulation:
        pass

    mssg = "lattice_dict not found in simulation.py file"
    with pytest.raises(ValueError, match=mssg):
        Lattice(
            Simulation(),
            control,
            mesh,
            0,
            verbose=False
        )


def test_missing_lattice_type():
    control = DummyControl()
    mesh = DummyMesh(dimensions=1)

    class Simulation:
        lattice_dict = {}

    mssg = "lattice_type missing in lattice_dict"
    with pytest.raises(ValueError, match=mssg):
        Lattice(
            Simulation(),
            control,
            mesh,
            0,
            verbose=False
        )


def test_lattice_mesh_mismatch():
    control = DummyControl()
    mesh = DummyMesh(dimensions=1)

    class Simulation:
        lattice_dict = {
            "lattice_type": "D2Q9"
        }

    mssg = "grid dimensions and lattice type are incompatible"
    with pytest.raises(ValueError, match=mssg):
        Lattice(
            Simulation(),
            control,
            mesh,
            0,
            verbose=False
        )

    control = DummyControl()
    mesh = DummyMesh(dimensions=2)

    class Simulation:
        lattice_dict = {
            "lattice_type": "D1Q3"
        }

    mssg = "grid dimensions and lattice type are incompatible"
    with pytest.raises(ValueError, match=mssg):
        Lattice(
            Simulation(),
            control,
            mesh,
            0,
            verbose=False
        )


def test_unsupported_lattice_type():
    control = DummyControl()
    mesh = DummyMesh(dimensions=2)

    class Simulation:
        lattice_dict = {
            "lattice_type": "D1Q9"
        }

    mssg = "Unsupported lattice type"
    with pytest.raises(ValueError, match=mssg):
        Lattice(
            Simulation(),
            control,
            mesh,
            0,
            verbose=False
        )


def test_d2q9_entries():
    control = DummyControl()
    mesh = DummyMesh(dimensions=2)

    class Simulation:
        lattice_dict = {
            "lattice_type": "D2Q9"
        }

    lattice = Lattice(
            Simulation(),
            control,
            mesh,
            0,
            verbose=False
        )

    cs = control.precision(1/np.sqrt(3))
    assert np.isclose(cs, lattice.cs)
    assert (lattice.cs.dtype == control.precision)
    cs2 = cs * cs
    assert np.isclose(cs2, lattice.cs2)
    cs_2 = 1 / cs2
    assert np.isclose(cs_2, lattice.cs_2)
    cs_4 = cs_2 * cs_2
    assert np.isclose(cs_4, lattice.cs_4)
    c_x = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
    c_y = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
    assert np.all(c_x == lattice.c_x)
    assert np.all(c_y == lattice.c_y)
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
    control = DummyControl()
    mesh = DummyMesh(dimensions=1)

    class Simulation:
        lattice_dict = {
            "lattice_type": "D1Q3"
        }

    lattice = Lattice(
            Simulation(),
            control,
            mesh,
            0,
            verbose=False
        )

    cs = control.precision(1/np.sqrt(3))
    assert np.isclose(cs, lattice.cs)
    assert (lattice.cs.dtype == control.precision)
    cs2 = cs * cs
    assert np.isclose(cs2, lattice.cs2)
    cs_2 = 1 / cs2
    assert np.isclose(cs_2, lattice.cs_2)
    cs_4 = cs_2 * cs_2
    assert np.isclose(cs_4, lattice.cs_4)
    c_x = np.array([0, 1, -1], dtype=np.int32)
    c_y = np.array([0, 0, 0], dtype=np.int32)
    assert np.all(c_x == lattice.c_x)
    assert np.all(c_y == lattice.c_y)
    weights = np.array([2/3, 1/6, 1/6], dtype=control.precision)
    assert np.all(weights == lattice.weights)
    inv_list = np.array([0, 2, 1], dtype=np.int32)
    assert np.all(inv_list == lattice.inv_list)
    no_of_directions = np.int32(3)
    assert no_of_directions == lattice.no_of_directions
