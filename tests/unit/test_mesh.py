import numpy as np
import pytest
import re

from pylabolt.base.mesh import Mesh


"""
Mesh test
"""


def test_missing_mesh_dict():
    class Simulation:
        pass

    mssg = "mesh_dict not found in simulation.py file"
    with pytest.raises(ValueError, match=mssg):
        Mesh(
            Simulation(),
            0,
            verbose=False
        )


def test_missing_grid():
    class Simulation:
        mesh_dict = {}

    mssg = "grid missing in mesh_dict"
    with pytest.raises(ValueError, match=mssg):
        Mesh(
            Simulation(),
            0,
            verbose=False
        )


def test_invalid_grid():
    class Simulation:
        mesh_dict = {"grid": 0}

    mssg = "grid entry in mesh_dict must be a list [Nx, Ny]"
    with pytest.raises(ValueError, match=re.escape(mssg)):
        Mesh(
            Simulation(),
            0,
            verbose=False
        )

    class Simulation:
        mesh_dict = {"grid": [1, 2, 3]}
    with pytest.raises(ValueError, match=re.escape(mssg)):
        Mesh(
            Simulation(),
            0,
            verbose=False
        )


def test_invalid_grid_size():
    class Simulation:
        mesh_dict = {"grid": [0, 1]}

    mssg = "grid dimensions cannot be zero"
    with pytest.raises(ValueError, match=mssg):
        Mesh(
            Simulation(),
            0,
            verbose=False
        )

    class Simulation:
        mesh_dict = {"grid": [1, 1]}

    mssg = "grid specified is a point"
    with pytest.raises(ValueError, match=mssg):
        Mesh(
            Simulation(),
            0,
            verbose=False
        )


def test_grid_dimensions():
    class Simulation:
        mesh_dict = {"grid": [1, 25]}

    mesh = Mesh(
        Simulation(),
        0,
        verbose=False
    )

    assert mesh.dimensions == 1

    class Simulation:
        mesh_dict = {"grid": [30, 25]}

    mesh = Mesh(
        Simulation(),
        0,
        verbose=False
    )

    assert mesh.dimensions == 2


def test_grid_shape_size():
    class Simulation:
        mesh_dict = {"grid": [30, 25]}

    mesh = Mesh(
        Simulation(),
        0,
        verbose=False
    )

    assert np.all(mesh.grid_global_shape == np.array([30, 25]))
    assert mesh.grid_global_size == np.prod(np.array([30, 25]))
