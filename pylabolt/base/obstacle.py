import numpy as np

from pylabolt.utils.IO import print_log
from pylabolt.backend.domain import global_to_local


class Obstacle:
    def __init__(
        self,
        simulation,
        mesh,
        domain,
        control,
        fields,
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
        self.compute_forces = False
        self.ref_point_torque = np.zeros(2, dtype=control.precision)
        self.obstacles = []

        # ------- Read and initialize obstacles elements ------- #
        self.read_obstacle_dict(
            mesh,
            domain,
            control,
            fields,
            fluid=fluid,
            phase=phase,
            scalar=scalar,
            verbose=verbose
        )

    def read_obstacle_dict(
        self,
        mesh,
        domain,
        control,
        fields,
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
        for key_no, key in enumerate(self.obstacle_dict.keys()):
            if key == "options":
                continue
            user_obstacle_name = key
            user_obstacle_dict = self.obstacle_dict[key]
            # self.read_user_obstacle_dict(
            #     user_obstacle_name,
            #     user_obstacle_dict,
            #     mesh,
            #     domain,
            #     control,
            #     fields,
            #     fluid=fluid,
            #     phase=phase,
            #     scalar=scalar,
            #     verbose=verbose
            # )

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
                    "compute_forces must be a bool:" +
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
                    "compute_forces must be a bool:" +
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
                        "ref_point_torque must be a list: (x, y)"
                    )
                self.ref_point_torque = np.array(
                    ref_point_torque, dtype=control.precision
                )
        print_log(
            "ref_point_torque: " + str(self.ref_point_torque),
            domain.mpi_rank,
            verbose=verbose
        )
        print_log("Obstacle options set\n", domain.mpi_rank, verbose)

    def read_user_obstacle_dict(
        self,
        user_obstacle_name,
        user_obstacle_dict,
        mesh,
        domain,
        control,
        fields,
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
            obstacle = create_circle()
        elif obstacle_type == "ellipse":
            obstacle = create_ellipse()
        elif obstacle_type == "custom":
            obstacle = create_custom_obstacle()
        else:
            raise ValueError("Unsupported obstacle type")

        self.obstacles.append(obstacle)

        # self.log_obstacle_data(
        #     user_obstacle_name,
        #     segments,
        #     orientations,
        #     locations,
        #     domain,
        #     fluid_config=fluid_config,
        #     phase_config=phase_config,
        #     verbose=verbose
        # )

    def log_obstacle_data(
        self,
        user_obstacle_name,
        segments,
        orientations,
        locations,
        domain,
        fluid_config=None,
        phase_config=None,
        verbose=True
    ):
        """
        Prints user-defined obstacle data to std-out after validation
        Args:

        Returns:

        """
        print_log("obstacle name: " + user_obstacle_name,
                  domain.mpi_rank, verbose)
        if fluid_config["type"] is not None:
            print_log("fluid obstacle condition:" + fluid_config["type"],
                      domain.mpi_rank, verbose)
            if fluid_config["scalar_value"] is not None:
                print_log("fluid value:" + str(fluid_config["scalar_value"]),
                          domain.mpi_rank, verbose)
            if fluid_config["vector_value"] is not None:
                print_log("fluid value:" + str(fluid_config["vector_value"]),
                          domain.mpi_rank, verbose)
        if phase_config["type"] is not None:
            print_log("phase obstacle condition:" + phase_config["type"],
                      domain.mpi_rank, verbose)
            if phase_config["scalar_value"] is not None:
                print_log("phase value:" + str(phase_config["scalar_value"]),
                          domain.mpi_rank, verbose)
            if phase_config["vector_value"] is not None:
                print_log("phase value:" + str(phase_config["vector_value"]),
                          domain.mpi_rank, verbose)
        print_log("segments:", domain.mpi_rank, verbose)
        for segment_no in range(len(segments)):
            print_log(
                "segment_" + str(segment_no + 1) + ":" +
                str(segments[segment_no]).ljust(30) + "| orientation: " +
                orientations[segment_no].ljust(5) + " | location: " +
                str(locations[segment_no]),
                domain.mpi_rank,
                verbose
            )
        print_log("\n", domain.mpi_rank, verbose)


def create_circle():
    pass


def create_ellipse():
    pass


def create_custom_obstacle():
    pass
