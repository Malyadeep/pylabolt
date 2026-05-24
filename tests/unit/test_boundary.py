import pytest
import numpy as np
import re


from pylabolt.base.boundary import Boundary, BoundaryElement
from factories import (
    make_mesh,
    make_domain,
    make_control,
    make_simulation,
    BoundaryDictsFluid,
    BoundaryDictsPhase
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


@pytest.fixture
def boundary_dicts_phase():
    return BoundaryDictsPhase()


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

    def test_valid_fluid_bc(self, setup_env, boundary_dicts_fluid):
        mesh, domain, control = setup_env
        boundary_dict = boundary_dicts_fluid.sample_dict
        simulation = make_simulation(boundary_dict=boundary_dict)
        boundary_types = [
            "bounce_back",
            "fixed_velocity",
            "fixed_pressure"
        ]
        boundary_values = [
            None,
            [0.5, 0.5],
            0.33
        ]
        orientations = ["horizontal", "horizontal", "vertical", "vertical"]
        locations = ["top", "bottom", "left", "right"]
        for itr, boundary_type in enumerate(boundary_types):
            simulation.boundary_dict["check"]["fluid"]["type"] = boundary_type
            simulation.boundary_dict["check"]["fluid"]["value"] =\
                boundary_values[itr]
            Nx, Ny = mesh.grid_global_shape
            simulation.boundary_dict["check"]["segments"] = [
                [[0, Ny - 1], [Nx - 1, Ny - 1]],
                [[0, 0], [Nx - 1, 0]],
                [[0, 0], [0, Ny - 1]],
                [[Nx - 1, 0], [Nx - 1, Ny - 1]]
            ]
            boundary_scalar_value = control.precision(0)
            boundary_vector_value = np.zeros(2, dtype=control.precision)
            if boundary_values[itr] is not None:
                if isinstance(boundary_values[itr], list):
                    boundary_vector_value = np.array(
                        boundary_values[itr], dtype=control.precision
                    )
                elif isinstance(boundary_values[itr], float):
                    boundary_scalar_value = control.precision(
                        boundary_values[itr]
                    )
            boundary = build_boundary(simulation, setup_env, fluid=True)
            assert (
                len(boundary.boundary_elements) ==
                len(simulation.boundary_dict["check"]["segments"])
            )
            for number in range(len(boundary.boundary_elements)):
                assert (
                    boundary.boundary_elements[number].type_fluid ==
                    boundary_type
                )
                assert (
                    boundary.boundary_elements[number].orientation ==
                    orientations[number]
                )
                assert (
                    boundary.boundary_elements[number].location ==
                    locations[number]
                )
                assert (
                    boundary.boundary_elements[number].name ==
                    "check_" + str(number)
                )
                assert np.all(
                    np.array(
                        simulation.boundary_dict["check"]["segments"][number]
                    ) == boundary.boundary_elements[number].segment
                )
                assert np.all(
                    boundary_vector_value ==
                    boundary.boundary_elements[number].vector_fluid
                )
                assert np.isclose(
                    boundary_scalar_value,
                    boundary.boundary_elements[number].scalar_fluid
                )


class TestPhaseBoundaryDicts:
    def test_missing_phase(self, setup_env, boundary_dicts_phase):
        boundary_dict = boundary_dicts_phase.sample_dict
        boundary_dict["check"].pop("phase")
        simulation = make_simulation(boundary_dict=boundary_dict)
        mssg = "phase missing in boundary: check"
        with pytest.raises(ValueError, match=mssg):
            build_boundary(simulation, setup_env, phase=True)

    def test_missing_phase_type(self, setup_env, boundary_dicts_phase):
        boundary_dict = boundary_dicts_phase.sample_dict
        boundary_dict["check"]["phase"].pop("type")
        simulation = make_simulation(boundary_dict=boundary_dict)
        mssg = "type missing in phase section of boundary: check"
        with pytest.raises(ValueError, match=mssg):
            build_boundary(simulation, setup_env, phase=True)

    def test_unsupported_phase_bc(self, setup_env, boundary_dicts_phase):
        boundary_dict = boundary_dicts_phase.sample_dict
        boundary_dict["check"]["phase"]["type"] = "other"
        simulation = make_simulation(boundary_dict=boundary_dict)
        mssg = "Unsupported boundary condition for phase: other"
        with pytest.raises(ValueError, match=mssg):
            build_boundary(simulation, setup_env, phase=True)

    def test_fixed_value_phase(self, setup_env, boundary_dicts_phase):
        boundary_dict = boundary_dicts_phase.get_fixed_value_dict()
        boundary_dict["check"]["phase"].pop("value")
        simulation = make_simulation(boundary_dict=boundary_dict)
        mssg = "value missing in phase section of boundary: check"
        with pytest.raises(ValueError, match=mssg):
            build_boundary(simulation, setup_env, phase=True)

        boundary_dict = boundary_dicts_phase.get_fixed_value_dict()
        values = [[1, 2, 3], [1], "foo"]
        mssg = "value for fixed_value must be a float or int: check"
        for value in values:
            boundary_dict["check"]["phase"]["value"] = value
            simulation = make_simulation(boundary_dict=boundary_dict)
            with pytest.raises(ValueError, match=mssg):
                build_boundary(simulation, setup_env, phase=True)

    def test_valid_phase_bc(self, setup_env, boundary_dicts_phase):
        mesh, domain, control = setup_env
        boundary_dict = boundary_dicts_phase.sample_dict
        simulation = make_simulation(boundary_dict=boundary_dict)
        boundary_types = [
            "bounce_back",
            "fixed_value"
        ]
        boundary_values = [
            None,
            0.33
        ]
        orientations = ["horizontal", "horizontal", "vertical", "vertical"]
        locations = ["top", "bottom", "left", "right"]
        for itr, boundary_type in enumerate(boundary_types):
            simulation.boundary_dict["check"]["phase"]["type"] = boundary_type
            simulation.boundary_dict["check"]["phase"]["value"] =\
                boundary_values[itr]
            Nx, Ny = mesh.grid_global_shape
            simulation.boundary_dict["check"]["segments"] = [
                [[0, Ny - 1], [Nx - 1, Ny - 1]],
                [[0, 0], [Nx - 1, 0]],
                [[0, 0], [0, Ny - 1]],
                [[Nx - 1, 0], [Nx - 1, Ny - 1]]
            ]
            boundary_scalar_value = control.precision(0)
            boundary_vector_value = np.zeros(2, dtype=control.precision)
            if boundary_values[itr] is not None:
                if isinstance(boundary_values[itr], float):
                    boundary_scalar_value = control.precision(
                        boundary_values[itr]
                    )
            boundary = build_boundary(simulation, setup_env, phase=True)
            assert (
                len(boundary.boundary_elements) ==
                len(simulation.boundary_dict["check"]["segments"])
            )
            for number in range(len(boundary.boundary_elements)):
                assert (
                    boundary.boundary_elements[number].type_phase ==
                    boundary_type
                )
                assert (
                    boundary.boundary_elements[number].orientation ==
                    orientations[number]
                )
                assert (
                    boundary.boundary_elements[number].location ==
                    locations[number]
                )
                assert (
                    boundary.boundary_elements[number].name ==
                    "check_" + str(number)
                )
                assert np.all(
                    np.array(
                        simulation.boundary_dict["check"]["segments"][number]
                    ) == boundary.boundary_elements[number].segment
                )
                assert np.all(
                    boundary_vector_value ==
                    boundary.boundary_elements[number].vector_phase
                )
                assert np.isclose(
                    boundary_scalar_value,
                    boundary.boundary_elements[number].scalar_phase
                )


class TestPeriodicValidation:
    def test_invalid_no_of_segments(self, setup_env, boundary_dicts_fluid):
        mesh, domain, control = setup_env
        Nx, Ny = mesh.grid_global_shape
        boundary_dict = boundary_dicts_fluid.get_periodic_dict(Nx, Ny)
        simulation = make_simulation(boundary_dict=boundary_dict)
        segments = [
            [[0, Ny - 1], [Nx - 1, Ny - 1]],
            [[0, 0], [Nx - 1, 0]],
            [[0, 0], [0, Ny - 1]],
            [[Nx - 1, 0], [Nx - 1, Ny - 1]]
        ]
        simulation.boundary_dict["check"]["segments"] = []
        mssg = "For periodic boundary, each segment should be" +\
            " followed by a corresponding periodic pair: check"
        for itr in range(4):
            simulation.boundary_dict["check"]["segments"].append(
                segments[itr]
            )
            if itr != 1:
                with pytest.raises(ValueError, match=mssg):
                    build_boundary(simulation, setup_env, fluid=True)

    def test_segment_orientation(self, setup_env, boundary_dicts_fluid):
        mesh, domain, control = setup_env
        Nx, Ny = mesh.grid_global_shape
        boundary_dict = boundary_dicts_fluid.get_periodic_dict(
            Nx,
            Ny,
            orientation="vertical"
        )
        simulation = make_simulation(boundary_dict=boundary_dict)
        simulation.boundary_dict["check"]["segments"][1] =\
            [[0, 0], [Nx - 1, 0]]
        mssg = "invalid periodic pair - different orientation: check"

        with pytest.raises(ValueError, match=mssg):
            build_boundary(simulation, setup_env, fluid=True)
        boundary_dict = boundary_dicts_fluid.get_periodic_dict(
            Nx,
            Ny,
            orientation="horizontal"
        )
        simulation = make_simulation(boundary_dict=boundary_dict)
        simulation.boundary_dict["check"]["segments"][1] =\
            [[0, 0], [0, Ny - 1]]
        with pytest.raises(ValueError, match=mssg):
            build_boundary(simulation, setup_env, fluid=True)

    def test_segment_length(self, setup_env, boundary_dicts_fluid):
        mesh, domain, control = setup_env
        Nx, Ny = mesh.grid_global_shape
        boundary_dict = boundary_dicts_fluid.get_periodic_dict(
            Nx,
            Ny,
            orientation="vertical"
        )
        simulation = make_simulation(boundary_dict=boundary_dict)
        simulation.boundary_dict["check"]["segments"] = [
            [[0, 0], [0, Ny - 1]],
            [[Nx - 1, 0], [Nx - 1, Ny - 3]]
        ]
        mssg = "invalid periodic pair - unequal length segments: check"

        with pytest.raises(ValueError, match=mssg):
            build_boundary(simulation, setup_env, fluid=True)
        boundary_dict = boundary_dicts_fluid.get_periodic_dict(
            Nx,
            Ny,
            orientation="horizontal"
        )
        simulation = make_simulation(boundary_dict=boundary_dict)
        simulation.boundary_dict["check"]["segments"] = [
            [[0, 0], [Nx - 1, 0]],
            [[0, Ny - 1], [Nx - 3, Ny - 1]]
        ]
        with pytest.raises(ValueError, match=mssg):
            build_boundary(simulation, setup_env, fluid=True)

    def test_connection_horizontal(self, setup_env, boundary_dicts_fluid):
        mesh, domain, control = setup_env
        Nx, Ny = mesh.grid_global_shape
        boundary_dict = boundary_dicts_fluid.get_periodic_dict(
            Nx,
            Ny,
            orientation="horizontal"
        )
        simulation = make_simulation(boundary_dict=boundary_dict)
        simulation.boundary_dict["check"]["segments"] = [
            [[2, 0], [Nx - 1, 0]],
            [[0, Ny - 1], [Nx - 3, Ny - 1]]
        ]
        mssg = "horizontal periodic boundaries must span" +\
            " same x-range: check"
        with pytest.raises(ValueError, match=mssg):
            build_boundary(simulation, setup_env, fluid=True)

        simulation.boundary_dict["check"]["segments"] = [
            [[0, 2], [Nx - 1, 2]],
            [[0, Ny - 1], [Nx - 1, Ny - 1]]
        ]
        mssg = "horizontal periodic boundaries must connect" +\
            " top-bottom boundaries: check"
        with pytest.raises(ValueError, match=mssg):
            build_boundary(simulation, setup_env, fluid=True)

    def test_connection_vertical(self, setup_env, boundary_dicts_fluid):
        mesh, domain, control = setup_env
        Nx, Ny = mesh.grid_global_shape
        boundary_dict = boundary_dicts_fluid.get_periodic_dict(
            Nx,
            Ny,
            orientation="vertical"
        )
        simulation = make_simulation(boundary_dict=boundary_dict)
        simulation.boundary_dict["check"]["segments"] = [
            [[0, 2], [0, Ny - 1]],
            [[Nx - 1, 0], [Nx - 1, Ny - 3]]
        ]
        mssg = "vertical periodic boundaries must span" +\
            " same y-range: check"
        with pytest.raises(ValueError, match=mssg):
            build_boundary(simulation, setup_env, fluid=True)

        simulation.boundary_dict["check"]["segments"] = [
            [[2, 0], [2, Ny - 1]],
            [[Nx - 1, 0], [Nx - 1, Ny - 1]]
        ]
        mssg = "vertical periodic boundaries must connect" +\
            " left-right boundaries: check"
        with pytest.raises(ValueError, match=mssg):
            build_boundary(simulation, setup_env, fluid=True)
