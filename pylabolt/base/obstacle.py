import numpy as np

from pylabolt.utils.helpers import print_log
from pylabolt.parallel.domain import local_to_global
from pylabolt.parallel.cpu.obstacle_kernels import (
    construct_circle,
    construct_ellipse
)


class Obstacle:
    def __init__(
        self,
        simulation,
        mesh,
        domain,
        control,
        fields,
        boundary,
        fluid=False,
        phase=False,
        scalar=False,
        verbose=True
    ):
        """
        Container for obstacle information
        Attributes:

        """
        print_log("-" * 80, domain.mpi_rank, verbose)
        print_log("Setting up obstacles...\n",
                  domain.mpi_rank, verbose)
        if not hasattr(simulation, "obstacle_dict"):
            raise ValueError(
                "obstacle_dict not found in simulation.py file"
            )
        self.obstacle_dict = simulation.obstacle_dict
        self.compute_forces = False
        self.compute_torque = False
        self.ref_point_torque = np.zeros(2, dtype=control.precision)
        self.write_obstacle_data = False
        self.write_interval = 1
        self.obstacles = []

        # ------- Read and initialize obstacles ------- #
        self.read_obstacle_dict(
            mesh,
            domain,
            control,
            fields,
            boundary,
            fluid=fluid,
            phase=phase,
            scalar=scalar,
            verbose=verbose
        )

        print_log("Setting up obstacles done!",
                  domain.mpi_rank, verbose)
        print_log("-" * 80, domain.mpi_rank, verbose)

    def read_obstacle_dict(
        self,
        mesh,
        domain,
        control,
        fields,
        boundary,
        fluid=False,
        phase=False,
        scalar=False,
        verbose=True
    ):
        """
        Reads obstacle_dict and creates obstacles
        Args:

        Returns:

        """
        if "options" not in self.obstacle_dict:
            raise ValueError("options missing in obstacle_dict")
        options_dict = self.obstacle_dict["options"]
        self.read_options_dict(
            options_dict,
            domain,
            control,
            verbose
        )
        key_list = list(self.obstacle_dict.keys())
        key_list.remove("options")
        for obstacle_id, key in enumerate(key_list):
            user_obstacle_name = key
            user_obstacle_dict = self.obstacle_dict[key]
            self.read_user_obstacle_dict(
                obstacle_id,
                user_obstacle_name,
                user_obstacle_dict,
                domain,
                control,
                mesh,
                fields,
                boundary,
                fluid=fluid,
                phase=phase,
                scalar=scalar,
                verbose=verbose
            )

    def read_options_dict(
        self,
        options_dict,
        domain,
        control,
        verbose
    ):
        """
        Reads and sets options for obstacle_dict
        Args:

        Returns:

        """
        print_log("Setting obstacle options", domain.mpi_rank, verbose)
        if "compute_forces" in options_dict:
            self.compute_forces = options_dict["compute_forces"]
            if not isinstance(self.compute_forces, (bool, np.bool_)):
                raise ValueError(
                    "compute_forces must be bool:" +
                    " True/False (default: False)"
                )
        print_log(
            "compute_forces: " + str(self.compute_forces),
            domain.mpi_rank,
            verbose=verbose
        )
        if "compute_torque" in options_dict:
            self.compute_torque = options_dict["compute_torque"]
            if not isinstance(self.compute_torque, (bool, np.bool_)):
                raise ValueError(
                    "compute_torque must be bool:" +
                    " True/False (default: False)"
                )
        print_log(
            "compute_torque: " + str(self.compute_torque),
            domain.mpi_rank,
            verbose=verbose
        )
        if "ref_point_torque" in options_dict:
            if self.compute_torque is False:
                print_log(
                    "WARNING! ref_point_torque ignored in "
                    "obstacle_dict options as compute_torque is False",
                    domain.mpi_rank,
                    verbose=verbose
                )
            else:
                ref_point_torque = options_dict["ref_point_torque"]
                if not isinstance(ref_point_torque, list):
                    raise ValueError(
                        "ref_point_torque must be list: (x, y)"
                    )
                self.ref_point_torque = np.array(
                    ref_point_torque, dtype=control.precision
                )
        print_log(
            "ref_point_torque: " + str(self.ref_point_torque),
            domain.mpi_rank,
            verbose=verbose
        )
        if "write_obstacle_data" in options_dict:
            self.write_obstacle_data = True
            temp_dict = options_dict["write_obstacle_data"]
            if "interval" not in temp_dict:
                raise ValueError("interval missing obstacle_dict options!")
            self.write_interval = temp_dict["interval"]
            if not (isinstance(self.write_interval, int) and
                    self.write_interval > 0):
                raise ValueError(
                    "interval must be int and > 0 in obstacle_dict options"
                )
        print_log(
            "write_obstacle_data: " + str(self.write_obstacle_data),
            domain.mpi_rank,
            verbose=verbose
        )
        if self.write_obstacle_data:
            print_log(
                "write_interval: " + str(self.write_interval),
                domain.mpi_rank,
                verbose=verbose
            )
        print_log("Obstacle options set\n", domain.mpi_rank, verbose)

    def read_user_obstacle_dict(
        self,
        obstacle_id,
        user_obstacle_name,
        user_obstacle_dict,
        domain,
        control,
        mesh,
        fields,
        boundary,
        fluid=False,
        phase=False,
        scalar=False,
        verbose=True
    ):
        """
        Reads user-defined obstacle dictionaries and
        creates obstacle elements for each boundaries
        Args:

        Returns:

        """
        if "type" not in user_obstacle_dict:
            raise ValueError(
                "type missing in obstacle: " + user_obstacle_name
            )
        obstacle_type = user_obstacle_dict["type"]

        if obstacle_type == "circle":
            obstacle = Circle(
                obstacle_id,
                user_obstacle_name,
                user_obstacle_dict,
                control,
                domain,
                mesh,
                fields,
                boundary
            )
        elif obstacle_type == "ellipse":
            obstacle = Ellipse(
                obstacle_id,
                user_obstacle_name,
                user_obstacle_dict,
                control,
                domain,
                mesh,
                fields,
                boundary
            )
        elif obstacle_type == "custom":
            pass
            # TODO: creation of custom obstacle from user-defined
            # function or txt file
            # obstacle = create_custom_obstacle()
        else:
            raise ValueError("Unsupported obstacle type: " + obstacle_type)

        self.obstacles.append(obstacle)

        self.log_obstacle_data(
            obstacle,
            domain,
            verbose=verbose
        )

    def log_obstacle_data(
        self,
        obstacle,
        domain,
        verbose=True
    ):
        """
        Prints user-defined obstacle data to std-out after validation
        Args:

        Returns:

        """
        for key, value in obstacle.properties.items():
            print_log(f"{key:20s}: {value}", domain.mpi_rank, verbose)
        if obstacle.static is False:
            for key, value in obstacle.motion_properties.items():
                print_log(f"{key:20s}: {value}", domain.mpi_rank, verbose)

        print_log("\n", domain.mpi_rank, verbose)


class Circle:
    def __init__(
        self,
        obstacle_id,
        user_obstacle_name,
        user_obstacle_dict,
        control,
        domain,
        mesh,
        fields,
        boundary
    ):
        """
        Container object for obstacle type: circle
        Attributes:

        """
        self.id = obstacle_id
        self.name = user_obstacle_name
        self.type = "circle"
        self.solid_boundary_nodes = []
        self.fluid_boundary_nodes = []
        self.surface_normals_solid = []
        self.surface_normals_fluid = []
        self.forces = np.zeros(2, dtype=control.precision)
        self.torque = control.precision(0)

        if "radius" not in user_obstacle_dict:
            raise ValueError(
                "radius missing in obstacle: " + user_obstacle_name
            )
        radius = user_obstacle_dict["radius"]
        if type(radius) not in (float, int):
            raise ValueError(
                "radius must be float or int for obstacle type circle: " +
                user_obstacle_name
            )
        self.radius = control.precision(radius)

        if "center" not in user_obstacle_dict:
            raise ValueError(
                "center missing in obstacle: " + user_obstacle_name
            )
        center = user_obstacle_dict["center"]
        if not isinstance(center, list):
            raise ValueError(
                "center must be list for obstacle type circle: " +
                user_obstacle_name
            )
        self.center = np.array(center, dtype=control.precision)

        self.inclination_angle = control.precision(0)

        if "density" not in user_obstacle_dict:
            raise ValueError(
                "density missing in obstacle: " + user_obstacle_name
            )
        density = user_obstacle_dict["density"]
        if type(density) not in (float, int):
            raise ValueError(
                "density must be float or int for obstacle type circle: " +
                user_obstacle_name
            )
        self.density = control.precision(density)

        if "static" not in user_obstacle_dict:
            raise ValueError(
                "static missing in obstacle: " + user_obstacle_name
            )
        static = user_obstacle_dict["static"]
        if not isinstance(static, (bool, np.bool_)):
            raise ValueError(
                "static must be True/False for obstacle type circle: " +
                user_obstacle_name
            )
        self.static = static

        if self.static:
            self.motion_type = None
            self.degree_of_freedom = None
            self.linear_velocity = np.zeros(2, dtype=control.precision)
            self.angular_velocity = control.precision(0)
        else:
            if "solid_motion_dict" not in user_obstacle_dict:
                raise ValueError(
                    "solid_motion_dict missing in obstacle: " +
                    user_obstacle_name
                )
            solid_motion_dict = user_obstacle_dict["solid_motion_dict"]
            for key in [
                "type",
                "degree_of_freedom",
                "linear_velocity",
                "angular_velocity"
            ]:
                if key not in solid_motion_dict:
                    raise ValueError(
                        key + " missing in solid_motion_dict: " +
                        user_obstacle_name
                    )
            self.motion_type = solid_motion_dict["type"]
            if self.motion_type not in ["fixed_velocity", "calculated"]:
                raise ValueError(
                    "Unsupported motion type: " + self.motion_type +
                    " in obstacle: " + user_obstacle_name
                )
            self.degree_of_freedom = solid_motion_dict["degree_of_freedom"]
            if self.degree_of_freedom not in [
                "rotation",
                "translation",
                "both"
            ]:
                raise ValueError(
                    "Unsupported degree of freedom: " + self.degree_of_freedom
                    + " in obstacle: " + user_obstacle_name
                )
            self.linear_velocity = solid_motion_dict["linear_velocity"]
            if not isinstance(self.linear_velocity, list):
                raise ValueError(
                    "linear_velocity must be list [ux, uy]: " +
                    user_obstacle_name
                )
            self.linear_velocity = np.array(
                self.linear_velocity, dtype=control.precision
            )
            self.angular_velocity = solid_motion_dict["angular_velocity"]
            if not isinstance(self.angular_velocity, (int, float)):
                raise ValueError(
                    "angular_velocity must be int/float: " +
                    user_obstacle_name
                )
            self.angular_velocity = np.array(
                self.angular_velocity, dtype=control.precision
            )

        self.reconstruct_args = (
            self.center,
            self.radius
        )

        self.set_solid_nodes(
            fields,
            boundary,
            domain,
            mesh
        )

    def set_solid_nodes(
        self,
        fields,
        boundary,
        domain,
        mesh,
    ):
        """
        Sets solid node values for obstacle type circle
        Args:

        Returns:

        """
        offset = domain.offset
        for ind in range(domain.size):
            if not fields.ghost_node[ind]:
                i = ind // domain.shape[1]
                j = ind - i * domain.shape[1]
                i_global, j_global = local_to_global(
                    i - 1, j - 1, offset
                )
                inside_solid, rx, ry = construct_circle(
                    i_global,
                    j_global,
                    mesh.grid_global_shape,
                    boundary.x_periodic,
                    boundary.y_periodic,
                    *self.reconstruct_args
                )
                if inside_solid:
                    fields.solid[ind] = True
                    fields.solid_id[ind] = self.id
                    fields.velocity[ind, 0] = self.linear_velocity[0] -\
                        self.angular_velocity * ry
                    fields.velocity[ind, 1] = self.linear_velocity[1] +\
                        self.angular_velocity * rx

    @property
    def properties(self):
        return {
            "obstacle id": self.id,
            "obstacle name": self.name,
            "obstacle type": self.type,
            "density": self.density,
            "center": [float(item) for item in self.center],
            "radius": self.radius,
            "static obstacle": self.static
        }

    @property
    def motion_properties(self):
        return {
            "motion_type": self.motion_type,
            "degree_of_freedom": self.degree_of_freedom,
            "linear_velocity": [float(item) for item in self.linear_velocity],
            "angular_velocity": self.angular_velocity
        }


class Ellipse:
    def __init__(
        self,
        obstacle_id,
        user_obstacle_name,
        user_obstacle_dict,
        control,
        domain,
        mesh,
        fields,
        boundary
    ):
        """
        Container object for obstacle type: circle
        Attributes:

        """
        self.id = obstacle_id
        self.name = user_obstacle_name
        self.type = "ellipse"
        self.solid_boundary_nodes = []
        self.fluid_boundary_nodes = []
        self.surface_normals_solid = []
        self.surface_normals_fluid = []
        self.forces = np.zeros(2, dtype=control.precision)
        self.torque = control.precision(0)

        if "semi_major_axis" not in user_obstacle_dict:
            raise ValueError(
                "semi_major_axis missing in obstacle: " + user_obstacle_name
            )
        semi_major_axis = user_obstacle_dict["semi_major_axis"]
        if type(semi_major_axis) not in (float, int):
            raise ValueError(
                "semi_major_axis must be float or int for obstacle" +
                " type ellipse: " + user_obstacle_name
            )
        self.semi_major_axis = control.precision(semi_major_axis)

        if "semi_minor_axis" not in user_obstacle_dict:
            raise ValueError(
                "semi_minor_axis missing in obstacle: " + user_obstacle_name
            )
        semi_minor_axis = user_obstacle_dict["semi_minor_axis"]
        if type(semi_minor_axis) not in (float, int):
            raise ValueError(
                "semi_minor_axis must be float or int for obstacle" +
                " type ellipse: " + user_obstacle_name
            )
        self.semi_minor_axis = control.precision(semi_minor_axis)

        if "inclination_angle" not in user_obstacle_dict:
            raise ValueError(
                "inclination_angle missing in obstacle: " + user_obstacle_name
            )
        inclination_angle = user_obstacle_dict["inclination_angle"]
        if type(semi_minor_axis) not in (float, int):
            raise ValueError(
                "inclination_angle must be float or int for obstacle" +
                " type ellipse representing angle in degrees: " +
                user_obstacle_name
            )
        self.inclination_angle = control.precision(inclination_angle) *\
            np.pi / 180
        self.cos_alpha = np.cos(self.inclination_angle)
        self.sin_alpha = np.sin(self.inclination_angle)

        if "center" not in user_obstacle_dict:
            raise ValueError(
                "center missing in obstacle: " + user_obstacle_name
            )
        center = user_obstacle_dict["center"]
        if not isinstance(center, list):
            raise ValueError(
                "center must be list for obstacle type circle: " +
                user_obstacle_name
            )
        self.center = np.array(center, dtype=control.precision)

        if "density" not in user_obstacle_dict:
            raise ValueError(
                "density missing in obstacle: " + user_obstacle_name
            )
        density = user_obstacle_dict["density"]
        if type(density) not in (float, int):
            raise ValueError(
                "density must be float or int for obstacle type circle: " +
                user_obstacle_name
            )
        self.density = control.precision(density)

        if "static" not in user_obstacle_dict:
            raise ValueError(
                "static missing in obstacle: " + user_obstacle_name
            )
        static = user_obstacle_dict["static"]
        if not isinstance(static, (bool, np.bool_)):
            raise ValueError(
                "static must be True/False for obstacle type circle: " +
                user_obstacle_name
            )
        self.static = static

        if self.static:
            self.motion_type = None
            self.degree_of_freedom = None
            self.linear_velocity = np.zeros(2, dtype=control.precision)
            self.angular_velocity = control.precision(0)
        else:
            if "solid_motion_dict" not in user_obstacle_dict:
                raise ValueError(
                    "solid_motion_dict missing in obstacle: " +
                    user_obstacle_name
                )
            solid_motion_dict = user_obstacle_dict["solid_motion_dict"]
            for key in [
                "type",
                "degree_of_freedom",
                "linear_velocity",
                "angular_velocity"
            ]:
                if key not in solid_motion_dict:
                    raise ValueError(
                        key + " missing in solid_motion_dict: " +
                        user_obstacle_name
                    )
            self.motion_type = solid_motion_dict["type"]
            if self.motion_type not in ["fixed_velocity", "calculated"]:
                raise ValueError(
                    "Unsupported motion type: " + self.motion_type +
                    " in obstacle: " + user_obstacle_name
                )
            self.degree_of_freedom = solid_motion_dict["degree_of_freedom"]
            if self.degree_of_freedom not in [
                "rotation",
                "translation",
                "both"
            ]:
                raise ValueError(
                    "Unsupported degree of freedom: " + self.motion_type +
                    " in obstacle: " + user_obstacle_name
                )
            self.linear_velocity = solid_motion_dict["linear_velocity"]
            if not isinstance(self.linear_velocity, list):
                raise ValueError(
                    "linear_velocity must be list [ux, uy]: " +
                    user_obstacle_name
                )
            self.linear_velocity = np.array(
                self.linear_velocity, dtype=control.precision
            )
            self.angular_velocity = solid_motion_dict["angular_velocity"]
            if not isinstance(self.angular_velocity, (int, float)):
                raise ValueError(
                    "angular_velocity must be int/float: " +
                    user_obstacle_name
                )
            self.angular_velocity = np.array(
                self.angular_velocity, dtype=control.precision
            )

        self.reconstruct_args = (
            self.center,
            self.semi_major_axis,
            self.semi_minor_axis,
            self.cos_alpha,
            self.sin_alpha
        )

        self.set_solid_nodes(
            fields,
            boundary,
            domain,
            mesh,
        )

    def set_solid_nodes(
        self,
        fields,
        boundary,
        domain,
        mesh,
    ):
        """
        Sets solid node values for obstacle type circle
        Args:

        Returns:

        """
        offset = domain.offset
        for ind in range(domain.size):
            if not fields.ghost_node[ind]:
                i = ind // domain.shape[1]
                j = ind - i * domain.shape[1]
                i_global, j_global = local_to_global(
                    i - 1, j - 1, offset
                )
                inside_solid, rx, ry = construct_ellipse(
                    i_global,
                    j_global,
                    mesh.grid_global_shape,
                    boundary.x_periodic,
                    boundary.y_periodic,
                    *self.reconstruct_args
                )
                if inside_solid:
                    fields.solid[ind] = True
                    fields.solid_id[ind] = self.id
                    fields.velocity[ind, 0] = self.linear_velocity[0] -\
                        self.angular_velocity * ry
                    fields.velocity[ind, 1] = self.linear_velocity[1] +\
                        self.angular_velocity * rx

    @property
    def properties(self):
        return {
            "obstacle id": self.id,
            "obstacle name": self.name,
            "obstacle type": self.type,
            "density": self.density,
            "center": [float(item) for item in self.center],
            "semi-major axis": self.semi_major_axis,
            "semi-minor axis": self.semi_minor_axis,
            "inclination angle": np.rad2deg(self.inclination_angle),
            "static obstacle": self.static
        }

    @property
    def motion_properties(self):
        return {
            "motion_type": self.motion_type,
            "degree_of_freedom": self.degree_of_freedom,
            "linear_velocity": [float(item) for item in self.linear_velocity],
            "angular_velocity": self.angular_velocity
        }


def create_custom_obstacle():
    pass
