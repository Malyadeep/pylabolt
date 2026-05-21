import pytest
import numpy as np
import re


from pylabolt.base.boundary import Boundary, BoundaryElement
from factories import (
    make_mesh,
    make_domain,
    make_control,
    make_simulation,
    BoundaryDictsFluid
)


@pytest.fixture
def setup_env():
    mesh = make_mesh((15, 11))
    domain = make_domain(17, 13, 0)
    control = make_control()
    return mesh, domain, control


@pytest.fixture
def boundary_dicts_fluid():
    return BoundaryDictsFluid()


def build_boundary(
    simulation,
    setup_env,
    fluid=False,
    phase=False,
    scalar=False
):
    mesh, domain, control = setup_env
    return Boundary(
        simulation,
        mesh,
        domain,
        control,
        fluid=fluid,
        phase=phase,
        scalar=scalar,
        verbose=False
    )


def test_missing_boundary_dict(setup_env):
    simulation = make_simulation()
    mssg = "boundary_dict not found in simulation.py file"
    with pytest.raises(ValueError, match=mssg):
        build_boundary(simulation, setup_env)


class TestBoundaryOptions:
    def test_missing_options(self, setup_env):
        simulation = make_simulation(boundary_dict={})
        mssg = "options missing in boundary_dict"
        with pytest.raises(ValueError, match=mssg):
            build_boundary(simulation, setup_env)

    def test_default_boundary_options(self, setup_env):
        boundary_dict = {
            "options": {}
        }
        simulation = make_simulation(boundary_dict=boundary_dict)
        boundary = build_boundary(simulation, setup_env)
        assert boundary.compute_forces is False

    def test_specified_boundary_options(self, setup_env):
        boundary_dict = {
            "options": {"compute_forces": True}
        }
        simulation = make_simulation(boundary_dict=boundary_dict)
        boundary = build_boundary(simulation, setup_env)
        assert boundary.compute_forces is True

        boundary_dict = {
            "options": {"compute_forces": 0}
        }
        simulation = make_simulation(boundary_dict=boundary_dict)
        mssg = "compute_forces must be a bool:" +\
            " True/False (default: False)"
        with pytest.raises(ValueError, match=re.escape(mssg)):
            build_boundary(simulation, setup_env)


class TestSegmentValidation:
    def test_missing_segment(self, setup_env):
        boundary_dict = {
            "options": {},
            "check": {}
        }
        simulation = make_simulation(boundary_dict=boundary_dict)
        mssg = "segments missing in boundary: check"
        with pytest.raises(ValueError, match=mssg):
            build_boundary(simulation, setup_env)

    def test_segment_is_list(self, setup_env):
        boundary_dict = {
            "options": {},
            "check": {"segments": 1}
        }
        simulation = make_simulation(boundary_dict=boundary_dict)
        mssg = "segments must be a list object: check"
        with pytest.raises(ValueError, match=mssg):
            build_boundary(simulation, setup_env)

    def test_empty_segment(self, setup_env):
        boundary_dict = {
            "options": {},
            "check": {"segments": []}
        }
        simulation = make_simulation(boundary_dict=boundary_dict)
        mssg = "segments cannot be an empty list: check"
        with pytest.raises(ValueError, match=mssg):
            build_boundary(simulation, setup_env)

    def test_segment_structure(self, setup_env):
        boundary_dict = {
            "options": {},
            "check": {
                "segments": [
                    []
                ]
            }
        }
        entries = (
            1,      # check each segment is a list
            [1],     # check each segment has length 2
            [1, 2],  # check each segment consists of list
            [[1, 2], [2, 1, 3]],   # check each coordinate has length 2
            [[1, 2, 3], [2, 3]],   # check each coordinate has length 2
        )
        simulation = make_simulation(boundary_dict=boundary_dict)
        for entry in entries:
            simulation.boundary_dict["check"]["segments"][0] = entry
            mssg = "segment must have structure [[x1, y1], [x2, y2]]: check"
            with pytest.raises(ValueError, match=re.escape(mssg)):
                build_boundary(simulation, setup_env)

    def test_segment_ordering(self, setup_env):
        boundary_dict = {
            "options": {},
            "check": {
                "segments": [
                    []
                ]
            }
        }
        entries = (
            [[0, 5], [5, 0]],
            [[6, 4], [1, 10]],
            [[7, 8], [1, 0]]
        )
        simulation = make_simulation(boundary_dict=boundary_dict)
        for entry in entries:
            simulation.boundary_dict["check"]["segments"][0] = entry
            mssg = "segment must satisfy x2 >= x1 and y2 >= y1: check"
            with pytest.raises(ValueError, match=re.escape(mssg)):
                build_boundary(simulation, setup_env)

    def test_segment_alignment(self, setup_env):
        boundary_dict = {
            "options": {},
            "check": {
                "segments": [
                    [[1, 2], [6, 10]]
                ]
            }
        }
        simulation = make_simulation(boundary_dict=boundary_dict)
        mssg = "segment must be axis-aligned (horizontal or vertical): check"
        with pytest.raises(ValueError, match=re.escape(mssg)):
            build_boundary(simulation, setup_env)


class TestFluidBoundaryDicts:
    def test_missing_fluid(self, setup_env, boundary_dicts_fluid):
        boundary_dict = boundary_dicts_fluid.sample_dict
        boundary_dict["check"].pop("fluid")
        simulation = make_simulation(boundary_dict=boundary_dict)
        mssg = "fluid missing in boundary: check"
        with pytest.raises(ValueError, match=mssg):
            build_boundary(simulation, setup_env, fluid=True)

    def test_missing_fluid_type(self, setup_env, boundary_dicts_fluid):
        boundary_dict = boundary_dicts_fluid.sample_dict
        boundary_dict["check"]["fluid"].pop("type")
        simulation = make_simulation(boundary_dict=boundary_dict)
        mssg = "type missing in fluid section of boundary: check"
        with pytest.raises(ValueError, match=mssg):
            build_boundary(simulation, setup_env, fluid=True)

    def test_unsupported_fluid_bc(self, setup_env, boundary_dicts_fluid):
        boundary_dict = boundary_dicts_fluid.sample_dict
        boundary_dict["check"]["fluid"]["type"] = "other"
        simulation = make_simulation(boundary_dict=boundary_dict)
        mssg = "Unsupported boundary condition for fluid: other"
        with pytest.raises(ValueError, match=mssg):
            build_boundary(simulation, setup_env, fluid=True)

    def test_fixed_velocity_fluid(self, setup_env, boundary_dicts_fluid):
        boundary_dict = boundary_dicts_fluid.get_fixed_velocity_dict()
        boundary_dict["check"]["fluid"].pop("value")
        simulation = make_simulation(boundary_dict=boundary_dict)
        mssg = "value missing in fluid section of boundary: check"
        with pytest.raises(ValueError, match=mssg):
            build_boundary(simulation, setup_env, fluid=True)

        boundary_dict = boundary_dicts_fluid.get_fixed_velocity_dict()
        values = [[1, 2, 3], 0.5, [1], "foo"]
        mssg = "value for fixed_velocity must be a list (ux, uy): check"
        for value in values:
            boundary_dict["check"]["fluid"]["value"] = value
            simulation = make_simulation(boundary_dict=boundary_dict)
            with pytest.raises(ValueError, match=re.escape(mssg)):
                build_boundary(simulation, setup_env, fluid=True)

    def test_fixed_pressure_fluid(self, setup_env, boundary_dicts_fluid):
        boundary_dict = boundary_dicts_fluid.get_fixed_pressure_dict()
        boundary_dict["check"]["fluid"].pop("value")
        simulation = make_simulation(boundary_dict=boundary_dict)
        mssg = "value missing in fluid section of boundary: check"
        with pytest.raises(ValueError, match=mssg):
            build_boundary(simulation, setup_env, fluid=True)

        boundary_dict = boundary_dicts_fluid.get_fixed_pressure_dict()
        values = [[1, 2, 3], [1], "foo"]
        mssg = "value for fixed_pressure must be a float or int: check"
        for value in values:
            boundary_dict["check"]["fluid"]["value"] = value
            simulation = make_simulation(boundary_dict=boundary_dict)
            with pytest.raises(ValueError, match=mssg):
                build_boundary(simulation, setup_env, fluid=True)
