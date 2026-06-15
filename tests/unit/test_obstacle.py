import pytest
import numpy as np
from copy import deepcopy
import re

from pylabolt.base.obstacle import Obstacle
from factories import (
    make_comm,
    make_mesh,
    make_control,
    make_fields,
    make_domain,
    make_simulation,
    make_boundary
)


def get_circle_dict():
    return {
        "options": {},
        "cylinder": {
            "type": "circle",
            "radius": 20,
            "center": [100/4 - 0.5, 100/4 - 0.5],
            "density": 1,
            "static": False,
            "periodicity": "none",
            "solid_motion_dict": {
                "type": "fixed_velocity",
                "degree_of_freedom": "both",
                "linear_velocity": [1e-3, 2e-4],
                "angular_velocity": 5e-5
            }
        }
    }


def get_ellipse_dict():
    return {
        "options": {},
        "ellipsoid": {
            "type": "ellipse",
            "semi_major_axis": 30,
            "semi_minor_axis": 15,
            "inclination_angle": 60,
            "center": [3 * 100/2 - 0.5, 3 * 100/4 - 0.5],
            "density": 1,
            "static": False,
            "periodicity": "none",
            "solid_motion_dict": {
                "type": "fixed_velocity",
                "degree_of_freedom": "both",
                "linear_velocity": [5e-3, 7e-4],
                "angular_velocity": 2e-5
            }
        }
    }


@pytest.fixture
def setup_env():
    mesh = make_mesh((100, 100))
    domain = make_domain(102, 102, 0)
    control = make_control()
    fields = make_fields(domain, control)
    boundary = make_boundary()
    return mesh, domain, control, fields, boundary


def build_obstacle(
    simulation,
    setup_env,
    fluid=False,
    phase=False,
    scalar=False
):
    mesh, domain, control, fields, boundary = setup_env
    return Obstacle(
        simulation,
        mesh,
        domain,
        control,
        fields,
        boundary,
        fluid=fluid,
        phase=phase,
        scalar=scalar,
        verbose=False
    )


def test_missing_obstacle_dict(setup_env):
    simulation = make_simulation()
    mssg = "obstacle_dict not found in simulation.py file"
    with pytest.raises(ValueError, match=mssg):
        build_obstacle(simulation, setup_env)


def test_missing_obstacle_type(setup_env):
    simulation = make_simulation(obstacle_dict={
        "options": {},
        "cylinder": {}
    })
    mssg = "type missing in obstacle: cylinder"
    with pytest.raises(ValueError, match=mssg):
        build_obstacle(simulation, setup_env)


def test_unsupported_obstacle_type(setup_env):
    simulation = make_simulation(obstacle_dict={
        "options": {},
        "cylinder": {"type": "none"}
    })
    mssg = "Unsupported obstacle type: none"
    with pytest.raises(ValueError, match=mssg):
        build_obstacle(simulation, setup_env)


class TestObstacleOptions:
    def test_missing_options(self, setup_env):
        simulation = make_simulation(obstacle_dict={})
        mssg = "options missing in obstacle_dict"
        with pytest.raises(ValueError, match=mssg):
            build_obstacle(simulation, setup_env)

    def test_default_obstacle_options(self, setup_env):
        obstacle_dict = {
            "options": {}
        }
        simulation = make_simulation(obstacle_dict=obstacle_dict)
        obstacle = build_obstacle(simulation, setup_env)
        assert obstacle.compute_forces is False
        assert obstacle.compute_torque is False
        assert np.all(obstacle.ref_point_torque == 0)
        assert obstacle.write_obstacle_data is False
        assert obstacle.write_interval == 1

    def mutate_options_dict(self, obstacle_dict, key_name, value, setup_env):
        obstacle_dict_1 = deepcopy(obstacle_dict)
        obstacle_dict_1["options"][key_name] = value
        simulation = make_simulation(obstacle_dict=obstacle_dict_1)
        build_obstacle(simulation, setup_env)

    def test_illegal_dtype_obstacle_options(self, setup_env):
        obstacle_dict = {
            "options": {
                "compute_forces": True,
                "compute_torque": True,
                "ref_point_torque": [45, 20],
                "write_obstacle_data": {
                    "interval": 100
                }
            }
        }

        mssg = "compute_forces must be bool: True/False (default: False)"
        with pytest.raises(ValueError, match=re.escape(mssg)):
            self.mutate_options_dict(
                obstacle_dict, "compute_forces", 1, setup_env
            )

        mssg = "compute_torque must be bool: True/False (default: False)"
        with pytest.raises(ValueError, match=re.escape(mssg)):
            self.mutate_options_dict(
                obstacle_dict, "compute_torque", 1, setup_env
            )

        mssg = "ref_point_torque must be list: (x, y)"
        with pytest.raises(ValueError, match=re.escape(mssg)):
            self.mutate_options_dict(
                obstacle_dict, "ref_point_torque", 1, setup_env
            )

        obstacle_dict_1 = deepcopy(obstacle_dict)
        obstacle_dict_1["options"]["write_obstacle_data"]["interval"] =\
            -1
        simulation = make_simulation(obstacle_dict=obstacle_dict_1)
        mssg = "interval must be int and > 0 in obstacle_dict options"
        with pytest.raises(ValueError, match=re.escape(mssg)):
            build_obstacle(simulation, setup_env)

    def test_specified_obstacle_options(self, setup_env):
        obstacle_dict = {
            "options": {
                "compute_forces": True,
                "compute_torque": True,
                "ref_point_torque": [45, 20],
                "write_obstacle_data": {
                    "interval": 100
                }
            }
        }
        simulation = make_simulation(obstacle_dict=obstacle_dict)
        obstacle = build_obstacle(simulation, setup_env)
        assert obstacle.compute_forces is True
        assert obstacle.compute_torque is True
        assert np.allclose(obstacle.ref_point_torque, np.array([45, 20]))
        assert obstacle.write_obstacle_data is True
        assert obstacle.write_interval == 100


class TestObstacleCircle:
    def test_missing_entries_circle(self, setup_env):
        keys_to_remove = [
            "radius",
            "center",
            "density",
            "static",
            "solid_motion_dict"
        ]

        for key in keys_to_remove:
            obstacle_dict = get_circle_dict()
            del obstacle_dict["cylinder"][key]
            simulation = make_simulation(obstacle_dict=obstacle_dict)
            mssg = key + " missing in obstacle: cylinder"
            with pytest.raises(ValueError, match=re.escape(mssg)):
                build_obstacle(simulation, setup_env)

        keys_to_remove_solid_motion = [
            "type",
            "degree_of_freedom",
            "linear_velocity",
            "angular_velocity"
        ]

        for key in keys_to_remove_solid_motion:
            obstacle_dict = get_circle_dict()
            del obstacle_dict["cylinder"]["solid_motion_dict"][key]
            simulation = make_simulation(obstacle_dict=obstacle_dict)
            mssg = key + " missing in solid_motion_dict: cylinder"
            with pytest.raises(ValueError, match=re.escape(mssg)):
                build_obstacle(simulation, setup_env)

    def test_illegal_dtype_circle(self, setup_env):
        illegal_dtype_map = {
            "radius": [15, 23],
            "center": 45,
            "density": True,
            "static": 5
        }

        mssgs = [
            "radius must be float or int for obstacle type circle: cylinder",
            "center must be list for obstacle type circle: cylinder",
            "density must be float or int for obstacle type circle: cylinder",
            "static must be True/False for obstacle type circle: cylinder",
        ]

        for key_no, key in enumerate(list(illegal_dtype_map.keys())):
            obstacle_dict = get_circle_dict()
            obstacle_dict["cylinder"][key] = illegal_dtype_map[key]
            simulation = make_simulation(obstacle_dict=obstacle_dict)
            with pytest.raises(ValueError, match=re.escape(mssgs[key_no])):
                build_obstacle(simulation, setup_env)

    def test_illegal_solid_motion_entries(self, setup_env):
        illegal_entry_map = {
            "type": "none",
            "degree_of_freedom": "all",
            "linear_velocity": 25,
            "angular_velocity": [2, 1]
        }

        mssgs = [
            "Unsupported motion type: none in obstacle: cylinder",
            "Unsupported degree of freedom: all in obstacle: cylinder",
            "linear_velocity must be list [ux, uy]: cylinder",
            "angular_velocity must be int/float: cylinder"
        ]

        for key_no, key in enumerate(list(illegal_entry_map.keys())):
            obstacle_dict = get_circle_dict()
            obstacle_dict["cylinder"]["solid_motion_dict"][key] =\
                illegal_entry_map[key]
            simulation = make_simulation(obstacle_dict=obstacle_dict)
            with pytest.raises(ValueError, match=re.escape(mssgs[key_no])):
                build_obstacle(simulation, setup_env)

    def test_valid_circle_entries(self, setup_env):
        obstacle_dict = get_circle_dict()
        simulation = make_simulation(obstacle_dict=obstacle_dict)
        obstacle = build_obstacle(simulation, setup_env)
        obstacle_circle = obstacle.obstacles[0]

        assert obstacle_circle.id == 0
        assert obstacle_circle.name == "cylinder"
        assert obstacle_circle.type == obstacle_dict["cylinder"]["type"]
        assert np.allclose(
            obstacle_circle.density, obstacle_dict["cylinder"]["density"]
        )
        assert np.allclose(
            obstacle_circle.center,
            np.array(obstacle_dict["cylinder"]["center"], dtype=np.float64)
        )
        assert np.allclose(
            obstacle_circle.radius, obstacle_dict["cylinder"]["radius"]
        )
        assert obstacle_circle.static == obstacle_dict["cylinder"]["static"]
        assert obstacle_circle.motion_type ==\
            obstacle_dict["cylinder"]["solid_motion_dict"]["type"]
        assert obstacle_circle.degree_of_freedom ==\
            obstacle_dict["cylinder"]["solid_motion_dict"]["degree_of_freedom"]
        assert np.allclose(
            obstacle_circle.linear_velocity,
            np.array(obstacle_dict["cylinder"]["solid_motion_dict"]
                     ["linear_velocity"], dtype=np.float64)
        )
        assert np.allclose(
            obstacle_circle.angular_velocity,
            obstacle_dict["cylinder"]["solid_motion_dict"]["angular_velocity"]
        )
