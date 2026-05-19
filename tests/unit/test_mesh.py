import numpy as np
import pytest
import re

from pylabolt.base.mesh import Mesh
from factories import make_simulation


"""
Mesh test
"""


def test_missing_mesh_dict():
    simulation = make_simulation()

    mssg = "mesh_dict not found in simulation.py file"
    with pytest.raises(ValueError, match=mssg):
        Mesh(
            simulation,
            0,
            verbose=False
        )


def test_missing_grid():
    simulation = make_simulation(mesh_dict={})

    mssg = "grid missing in mesh_dict"
    with pytest.raises(ValueError, match=mssg):
        Mesh(
            simulation,
            0,
            verbose=False
        )


def test_invalid_grid():
    simulation = make_simulation(mesh_dict={"grid": 0})

    mssg = "grid entry in mesh_dict must be a list [Nx, Ny]"
    with pytest.raises(ValueError, match=re.escape(mssg)):
        Mesh(
            simulation,
            0,
            verbose=False
        )

    simulation = make_simulation(mesh_dict={"grid": [1, 2, 3]})
    with pytest.raises(ValueError, match=re.escape(mssg)):
        Mesh(
            simulation,
            0,
            verbose=False
        )


def test_invalid_grid_size():
    simulation = make_simulation(mesh_dict={"grid": [0, 1]})

    mssg = "grid dimensions cannot be zero"
    with pytest.raises(ValueError, match=mssg):
        Mesh(
            simulation,
            0,
            verbose=False
        )

    simulation = make_simulation(mesh_dict={"grid": [1, 1]})

    mssg = "grid specified is a point"
    with pytest.raises(ValueError, match=mssg):
        Mesh(
            simulation,
            0,
            verbose=False
        )


def test_grid_dimensions():
    simulation = make_simulation(mesh_dict={"grid": [1, 25]})

    mesh = Mesh(
        simulation,
        0,
        verbose=False
    )

    assert mesh.dimensions == 1

    simulation = make_simulation(mesh_dict={"grid": [30, 25]})

    mesh = Mesh(
        simulation,
        0,
        verbose=False
    )

    assert mesh.dimensions == 2


def test_grid_shape_size():
    simulation = make_simulation(mesh_dict={"grid": [30, 25]})

    mesh = Mesh(
        simulation,
        0,
        verbose=False
    )

    assert np.all(mesh.grid_global_shape == np.array([30, 25]))
    assert mesh.grid_global_size == np.prod(np.array([30, 25]))
