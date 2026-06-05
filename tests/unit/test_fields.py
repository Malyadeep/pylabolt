import numpy as np
import pytest

from pylabolt.base.fields import Fields
from factories import make_control, make_lattice, make_domain


""" Fixtures """


@pytest.fixture
def setup_env():
    control = make_control()
    lattice = make_lattice(9)
    domain = make_domain(5, 5, 0)
    return control, lattice, domain


"""
Geometry components
"""


def test_geometry_fields(setup_env):
    control, lattice, domain = setup_env

    fields = Fields(
        control,
        lattice,
        domain,
        fluid=False,
        phase=False,
        scalar=False
    )

    fields_check = {
        "solid": [(domain.size,), np.bool_, False],
        "solid_id": [(domain.size,), np.int32, -1],
        "fluid_boundary": [(domain.size,), np.bool_, False],
        "solid_boundary": [(domain.size,), np.bool_, False],
        "ghost_node": [(domain.size,), np.bool_, False],
        "periodic_boundary": [(domain.size,), np.bool_, False]
    }
    for field_name in fields_check.keys():
        assert hasattr(fields, field_name)
        assert (
            getattr(fields, field_name).shape ==
            fields_check[field_name][0]
        )
        assert (
            getattr(fields, field_name).dtype ==
            fields_check[field_name][1]
        )
        if field_name != "ghost_node":
            assert np.all(
                getattr(fields, field_name) ==
                fields_check[field_name][2]
            )


def test_ghost_node_init(setup_env):
    control, lattice, domain = setup_env

    fields = Fields(
        control,
        lattice,
        domain,
        fluid=False,
        phase=False,
        scalar=False
    )

    expected_result = np.zeros(domain.size, dtype=np.bool_)
    for i in range(domain.Nx):
        for j in range(domain.Ny):
            if i == 0 or j == 0 or i == domain.Nx - 1 or j == domain.Ny - 1:
                ind = i * domain.Ny + j
                expected_result[ind] = True

    assert np.all(fields.ghost_node == expected_result)


"""
Fluid components
"""


def test_fluid_fields(setup_env):
    control, lattice, domain = setup_env

    fields = Fields(
        control,
        lattice,
        domain,
        fluid=True,
        phase=False,
        scalar=False
    )

    fields_check = {
        "velocity": [(domain.size, 2), control.precision, 0],
        "density": [(domain.size,), control.precision, 0],
        "pressure": [(domain.size,), control.precision, 0],
        "force_field": [(domain.size, 2), control.precision, 0],
        "pop_fluid": [
            (domain.size, lattice.no_of_directions), control.precision, 0
        ],
        "pop_fluid_new": [
            (domain.size, lattice.no_of_directions), control.precision, 0
        ],
        "pop_fluid_eq": [
            (domain.size, lattice.no_of_directions), control.precision, 0
        ]
    }
    for field_name in fields_check.keys():
        assert hasattr(fields, field_name)
        assert (
            getattr(fields, field_name).shape ==
            fields_check[field_name][0]
        )
        assert (
            getattr(fields, field_name).dtype ==
            fields_check[field_name][1]
        )
        assert np.all(
            getattr(fields, field_name) ==
            fields_check[field_name][2]
        )


"""
Phase components
"""


def test_phase_fields(setup_env):
    control, lattice, domain = setup_env

    fields = Fields(
        control,
        lattice,
        domain,
        fluid=True,
        phase=True,
        scalar=False
    )

    fields_check = {
        "phase_field": [(domain.size,), control.precision, 0],
        "grad_phase_field": [(domain.size, 2), control.precision, 0],
        "lap_phase_field": [(domain.size,), control.precision, 0],
        "surface_tension_force": [(domain.size, 2), control.precision, 0],
        "pressure_corr_force": [(domain.size, 2), control.precision, 0],
        "viscous_corr_force": [(domain.size, 2), control.precision, 0],
        "curvature": [(domain.size,), control.precision, 0],
        "phase_field_solid": [(domain.size,), control.precision, 0],
        "pop_phase": [
            (domain.size, lattice.no_of_directions), control.precision, 0
        ],
        "pop_phase_new": [
            (domain.size, lattice.no_of_directions), control.precision, 0
        ],
        "pop_phase_eq": [
            (domain.size, lattice.no_of_directions), control.precision, 0
        ]
    }
    for field_name in fields_check.keys():
        assert hasattr(fields, field_name)
        assert (
            getattr(fields, field_name).shape ==
            fields_check[field_name][0]
        )
        assert (
            getattr(fields, field_name).dtype ==
            fields_check[field_name][1]
        )
        assert np.all(
            getattr(fields, field_name) ==
            fields_check[field_name][2]
        )
