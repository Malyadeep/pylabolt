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
            "center": [3 * 100/4 - 0.5, 3 * 100/4 - 0.5],
            "density": 1,
            "static": False,
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

    def test_static_obstacle_entries(self, setup_env):
        obstacle_dict = get_circle_dict()
        obstacle_dict["cylinder"]["static"] = True
        del obstacle_dict["cylinder"]["solid_motion_dict"]
        simulation = make_simulation(obstacle_dict=obstacle_dict)
        obstacle = build_obstacle(simulation, setup_env)
        obstacle_circle = obstacle.obstacles[0]

        assert obstacle_circle.motion_type is None
        assert obstacle_circle.degree_of_freedom is None
        assert np.allclose(
            obstacle_circle.linear_velocity,
            np.zeros(2, dtype=np.float64)
        )
        assert np.allclose(obstacle_circle.angular_velocity, 0)

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
        assert np.allclose(obstacle_circle.inclination_angle, 0)
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

    def test_circle_nodes_initialization(self, setup_env):
        obstacle_dict = get_circle_dict()
        simulation = make_simulation(obstacle_dict=obstacle_dict)
        obstacle = build_obstacle(simulation, setup_env)
        mesh, domain, control, fields, boundary = setup_env
        obstacle_circle = obstacle.obstacles[0]
        Nx, Ny = mesh.grid_global_shape[0], mesh.grid_global_shape[1]
        Nx_pad, Ny_pad = Nx + 2, Ny + 2
        domain_size = Nx_pad * Ny_pad
        solid_check = np.zeros(domain_size, dtype=np.bool_)
        solid_id_check = np.zeros(domain_size, dtype=np.int32)
        solid_id_check[:] = -1
        velocity_check = np.zeros((domain_size, 2), dtype=np.float64)
        for ind in range(domain_size):
            i = ind // Ny_pad
            j = ind % Ny_pad
            rx = i - (obstacle_circle.center[0] + 1)
            ry = j - (obstacle_circle.center[1] + 1)
            rx_min = rx
            ry_min = ry
            # ------------- All boundaries periodic -------------
            if np.abs(rx + Nx) < np.abs(rx_min):
                rx_min = rx + Nx
            if np.abs(rx - Nx) < np.abs(rx_min):
                rx_min = rx - Nx
            if np.abs(ry + Ny) < np.abs(ry_min):
                ry_min = ry + Ny
            if np.abs(ry - Ny) < np.abs(ry_min):
                ry_min = ry - Ny
            # ------------- All boundaries periodic -------------
            dist = np.sqrt(rx_min * rx_min + ry_min * ry_min)
            if dist <= obstacle_circle.radius and not fields.ghost_node[ind]:
                solid_check[ind] = True
                solid_id_check[ind] = obstacle_circle.id
                velocity_check[ind, 0] = obstacle_circle.linear_velocity[0] -\
                    obstacle_circle.angular_velocity * ry
                velocity_check[ind, 1] = obstacle_circle.linear_velocity[1] +\
                    obstacle_circle.angular_velocity * rx

        assert np.all(fields.solid == solid_check)
        assert np.all(fields.solid_id == solid_id_check)
        assert np.allclose(fields.velocity, velocity_check)


class TestObstacleEllipse:
    def test_missing_entries_ellipse(self, setup_env):
        keys_to_remove = [
            "semi_major_axis",
            "semi_minor_axis",
            "center",
            "inclination_angle",
            "density",
            "static",
            "solid_motion_dict"
        ]

        for key in keys_to_remove:
            obstacle_dict = get_ellipse_dict()
            del obstacle_dict["ellipsoid"][key]
            simulation = make_simulation(obstacle_dict=obstacle_dict)
            mssg = key + " missing in obstacle: ellipsoid"
            with pytest.raises(ValueError, match=re.escape(mssg)):
                build_obstacle(simulation, setup_env)

        keys_to_remove_solid_motion = [
            "type",
            "degree_of_freedom",
            "linear_velocity",
            "angular_velocity"
        ]

        for key in keys_to_remove_solid_motion:
            obstacle_dict = get_ellipse_dict()
            del obstacle_dict["ellipsoid"]["solid_motion_dict"][key]
            simulation = make_simulation(obstacle_dict=obstacle_dict)
            mssg = key + " missing in solid_motion_dict: ellipsoid"
            with pytest.raises(ValueError, match=re.escape(mssg)):
                build_obstacle(simulation, setup_env)

    def test_illegal_dtype_ellipse(self, setup_env):
        illegal_dtype_map = {
            "semi_major_axis": [15, 23],
            "semi_minor_axis": [15, 23],
            "center": 45,
            "inclination_angle": [15, 23],
            "density": True,
            "static": 5
        }

        mssgs = [
            "semi_major_axis must be float or int for obstacle type " +
            "ellipse: ellipsoid",
            "semi_minor_axis must be float or int for obstacle type " +
            "ellipse: ellipsoid",
            "center must be list for obstacle type ellipse: ellipsoid",
            "inclination_angle must be float or int for obstacle" +
            " type ellipse representing angle in degrees: ellipsoid",
            "density must be float or int for obstacle type " +
            "ellipse: ellipsoid",
            "static must be True/False for obstacle type ellipse: ellipsoid",
        ]

        for key_no, key in enumerate(list(illegal_dtype_map.keys())):
            obstacle_dict = get_ellipse_dict()
            obstacle_dict["ellipsoid"][key] = illegal_dtype_map[key]
            simulation = make_simulation(obstacle_dict=obstacle_dict)
            with pytest.raises(ValueError, match=re.escape(mssgs[key_no])):
                build_obstacle(simulation, setup_env)

    def test_static_obstacle_entries(self, setup_env):
        obstacle_dict = get_ellipse_dict()
        obstacle_dict["ellipsoid"]["static"] = True
        del obstacle_dict["ellipsoid"]["solid_motion_dict"]
        simulation = make_simulation(obstacle_dict=obstacle_dict)
        obstacle = build_obstacle(simulation, setup_env)
        obstacle_circle = obstacle.obstacles[0]

        assert obstacle_circle.motion_type is None
        assert obstacle_circle.degree_of_freedom is None
        assert np.allclose(
            obstacle_circle.linear_velocity,
            np.zeros(2, dtype=np.float64)
        )
        assert np.allclose(obstacle_circle.angular_velocity, 0)

    def test_illegal_solid_motion_entries(self, setup_env):
        illegal_entry_map = {
            "type": "none",
            "degree_of_freedom": "all",
            "linear_velocity": 25,
            "angular_velocity": [2, 1]
        }

        mssgs = [
            "Unsupported motion type: none in obstacle: ellipsoid",
            "Unsupported degree of freedom: all in obstacle: ellipsoid",
            "linear_velocity must be list [ux, uy]: ellipsoid",
            "angular_velocity must be int/float: ellipsoid"
        ]

        for key_no, key in enumerate(list(illegal_entry_map.keys())):
            obstacle_dict = get_ellipse_dict()
            obstacle_dict["ellipsoid"]["solid_motion_dict"][key] =\
                illegal_entry_map[key]
            simulation = make_simulation(obstacle_dict=obstacle_dict)
            with pytest.raises(ValueError, match=re.escape(mssgs[key_no])):
                build_obstacle(simulation, setup_env)

    def test_valid_ellipse_entries(self, setup_env):
        obstacle_dict = get_ellipse_dict()
        simulation = make_simulation(obstacle_dict=obstacle_dict)
        obstacle = build_obstacle(simulation, setup_env)
        obstacle_ellipse = obstacle.obstacles[0]

        assert obstacle_ellipse.id == 0
        assert obstacle_ellipse.name == "ellipsoid"
        assert obstacle_ellipse.type == obstacle_dict["ellipsoid"]["type"]
        assert np.allclose(
            obstacle_ellipse.density, obstacle_dict["ellipsoid"]["density"]
        )
        assert np.allclose(
            obstacle_ellipse.center,
            np.array(obstacle_dict["ellipsoid"]["center"], dtype=np.float64)
        )
        assert np.allclose(
            obstacle_ellipse.semi_major_axis,
            obstacle_dict["ellipsoid"]["semi_major_axis"]
        )
        assert np.allclose(
            obstacle_ellipse.semi_minor_axis,
            obstacle_dict["ellipsoid"]["semi_minor_axis"]
        )
        assert np.allclose(
            obstacle_ellipse.inclination_angle,
            obstacle_dict["ellipsoid"]["inclination_angle"] * np.pi / 180
        )
        assert np.allclose(
            obstacle_ellipse.cos_alpha,
            np.cos(obstacle_dict["ellipsoid"]["inclination_angle"] *
                   np.pi / 180)
        )
        assert np.allclose(
            obstacle_ellipse.sin_alpha,
            np.sin(obstacle_dict["ellipsoid"]["inclination_angle"] *
                   np.pi / 180)
        )
        assert obstacle_ellipse.static == obstacle_dict["ellipsoid"]["static"]
        assert obstacle_ellipse.motion_type ==\
            obstacle_dict["ellipsoid"]["solid_motion_dict"]["type"]
        assert (
            obstacle_ellipse.degree_of_freedom ==
            obstacle_dict["ellipsoid"]["solid_motion_dict"]
            ["degree_of_freedom"]
        )
        assert np.allclose(
            obstacle_ellipse.linear_velocity,
            np.array(obstacle_dict["ellipsoid"]["solid_motion_dict"]
                     ["linear_velocity"], dtype=np.float64)
        )
        assert np.allclose(
            obstacle_ellipse.angular_velocity,
            obstacle_dict["ellipsoid"]["solid_motion_dict"]["angular_velocity"]
        )

    def test_ellipse_nodes_initialization(self, setup_env):
        obstacle_dict = get_ellipse_dict()
        simulation = make_simulation(obstacle_dict=obstacle_dict)
        obstacle = build_obstacle(simulation, setup_env)
        mesh, domain, control, fields, boundary = setup_env
        obstacle_ellipse = obstacle.obstacles[0]
        Nx, Ny = mesh.grid_global_shape[0], mesh.grid_global_shape[1]
        Nx_pad, Ny_pad = Nx + 2, Ny + 2
        domain_size = Nx_pad * Ny_pad
        solid_check = np.zeros(domain_size, dtype=np.bool_)
        solid_id_check = np.zeros(domain_size, dtype=np.int32)
        solid_id_check[:] = -1
        velocity_check = np.zeros((domain_size, 2), dtype=np.float64)
        for ind in range(domain_size):
            i = ind // Ny_pad
            j = ind % Ny_pad
            rx = i - (obstacle_ellipse.center[0] + 1)
            ry = j - (obstacle_ellipse.center[1] + 1)
            rx_min = rx
            ry_min = ry
            # ------------- All boundaries periodic -------------
            if np.abs(rx + Nx) < np.abs(rx_min):
                rx_min = rx + Nx
            if np.abs(rx - Nx) < np.abs(rx_min):
                rx_min = rx - Nx
            if np.abs(ry + Ny) < np.abs(ry_min):
                ry_min = ry + Ny
            if np.abs(ry - Ny) < np.abs(ry_min):
                ry_min = ry - Ny
            # ------------- All boundaries periodic -------------
            x = (rx_min * obstacle_ellipse.cos_alpha +
                 ry_min * obstacle_ellipse.sin_alpha)
            y = (-rx_min * obstacle_ellipse.sin_alpha +
                 ry_min * obstacle_ellipse.cos_alpha)
            scaled_dist = (
                (x * x) / (
                    obstacle_ellipse.semi_major_axis *
                    obstacle_ellipse.semi_major_axis
                ) +
                (y * y) / (
                    obstacle_ellipse.semi_minor_axis *
                    obstacle_ellipse.semi_minor_axis
                )
            )
            if scaled_dist <= 1 and not fields.ghost_node[ind]:
                solid_check[ind] = True
                solid_id_check[ind] = obstacle_ellipse.id
                velocity_check[ind, 0] = obstacle_ellipse.linear_velocity[0] -\
                    obstacle_ellipse.angular_velocity * ry_min
                velocity_check[ind, 1] = obstacle_ellipse.linear_velocity[1] +\
                    obstacle_ellipse.angular_velocity * rx_min

        assert np.all(fields.solid == solid_check)
        assert np.all(fields.solid_id == solid_id_check)
        assert np.allclose(fields.velocity, velocity_check)


class TestMultipleObstacle:
    def test_multiple_obstacle(self, setup_env):
        obstacle_dict_circle = get_circle_dict()
        obstacle_dict_ellipse = get_ellipse_dict()
        obstacle_dict = {
            "options": {},
            "cylinder": obstacle_dict_circle["cylinder"],
            "ellipsoid": obstacle_dict_ellipse["ellipsoid"]
        }
        simulation = make_simulation(obstacle_dict=obstacle_dict)
        obstacle = build_obstacle(simulation, setup_env)

        assert len(obstacle.obstacles) == 2
        assert obstacle.obstacles[0].id == 0
        assert obstacle.obstacles[1].id == 1

        assert obstacle.obstacles[0].type == "circle"
        assert obstacle.obstacles[1].type == "ellipse"

    def test_multiple_obstacle_node_initialization(self, setup_env):
        obstacle_dict_circle = get_circle_dict()
        obstacle_dict_ellipse = get_ellipse_dict()
        obstacle_dict = {
            "options": {},
            "cylinder": obstacle_dict_circle["cylinder"],
            "ellipsoid": obstacle_dict_ellipse["ellipsoid"]
        }
        simulation = make_simulation(obstacle_dict=obstacle_dict)
        obstacle = build_obstacle(simulation, setup_env)

        mesh, domain, control, fields, boundary = setup_env
        Nx, Ny = mesh.grid_global_shape[0], mesh.grid_global_shape[1]
        Nx_pad, Ny_pad = Nx + 2, Ny + 2
        domain_size = Nx_pad * Ny_pad
        solid_check = np.zeros(domain_size, dtype=np.bool_)
        solid_id_check = np.zeros(domain_size, dtype=np.int32)
        solid_id_check[:] = -1
        velocity_check = np.zeros((domain_size, 2), dtype=np.float64)

        obstacle_circle = obstacle.obstacles[0]
        obstacle_ellipse = obstacle.obstacles[1]
        for ind in range(domain_size):
            i = ind // Ny_pad
            j = ind % Ny_pad
            rx = i - (obstacle_ellipse.center[0] + 1)
            ry = j - (obstacle_ellipse.center[1] + 1)
            rx_min = rx
            ry_min = ry
            # ------------- All boundaries periodic -------------
            if np.abs(rx + Nx) < np.abs(rx_min):
                rx_min = rx + Nx
            if np.abs(rx - Nx) < np.abs(rx_min):
                rx_min = rx - Nx
            if np.abs(ry + Ny) < np.abs(ry_min):
                ry_min = ry + Ny
            if np.abs(ry - Ny) < np.abs(ry_min):
                ry_min = ry - Ny
            # ------------- All boundaries periodic -------------
            x = (rx_min * obstacle_ellipse.cos_alpha +
                 ry_min * obstacle_ellipse.sin_alpha)
            y = (-rx_min * obstacle_ellipse.sin_alpha +
                 ry_min * obstacle_ellipse.cos_alpha)
            scaled_dist = (
                (x * x) / (
                    obstacle_ellipse.semi_major_axis *
                    obstacle_ellipse.semi_major_axis
                ) +
                (y * y) / (
                    obstacle_ellipse.semi_minor_axis *
                    obstacle_ellipse.semi_minor_axis
                )
            )
            if scaled_dist <= 1 and not fields.ghost_node[ind]:
                solid_check[ind] = True
                solid_id_check[ind] = obstacle_ellipse.id
                velocity_check[ind, 0] = obstacle_ellipse.linear_velocity[0] -\
                    obstacle_ellipse.angular_velocity * ry_min
                velocity_check[ind, 1] = obstacle_ellipse.linear_velocity[1] +\
                    obstacle_ellipse.angular_velocity * rx_min

            rx = i - (obstacle_circle.center[0] + 1)
            ry = j - (obstacle_circle.center[1] + 1)
            rx_min = rx
            ry_min = ry
            # ------------- All boundaries periodic -------------
            if np.abs(rx + Nx) < np.abs(rx_min):
                rx_min = rx + Nx
            if np.abs(rx - Nx) < np.abs(rx_min):
                rx_min = rx - Nx
            if np.abs(ry + Ny) < np.abs(ry_min):
                ry_min = ry + Ny
            if np.abs(ry - Ny) < np.abs(ry_min):
                ry_min = ry - Ny
            # ------------- All boundaries periodic -------------
            scaled_dist = np.sqrt(
                rx_min * rx_min + ry_min * ry_min
            ) / obstacle_circle.radius
            if scaled_dist <= 1 and not fields.ghost_node[ind]:
                solid_check[ind] = True
                solid_id_check[ind] = obstacle_circle.id
                velocity_check[ind, 0] = obstacle_circle.linear_velocity[0] -\
                    obstacle_circle.angular_velocity * ry_min
                velocity_check[ind, 1] = obstacle_circle.linear_velocity[1] +\
                    obstacle_circle.angular_velocity * rx_min

        assert np.all(fields.solid == solid_check)
        assert np.all(fields.solid_id == solid_id_check)
        assert np.allclose(fields.velocity, velocity_check)
