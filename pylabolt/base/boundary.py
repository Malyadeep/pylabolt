import numpy as np

from pylabolt.utils.IO import print_log
from pylabolt.backend.domain import global_to_local


class BoundaryElement:
    def __init__(
        self,
        boundary_name,
        segment,
        segment_no,
        orientation,
        location,
        domain,
        control,
        fluid_config=None,
        phase_config=None
    ):
        """
        Container for each boundary element data
        Attributes:

        """
        # TODO: make this class dimension aware: for the moment only 2D
        self.name = boundary_name + "_" + str(int(segment_no))
        self.orientation = orientation
        self.location = location
        self.segment = np.array(segment)
        if fluid_config is not None:
            self.type_fluid = fluid_config["type"]
            self.scalar_fluid = control.precision(0)
            self.vector_fluid = np.zeros(2)
            if fluid_config["scalar_value"] is not None:
                self.scalar_fluid = control.precision(
                    fluid_config["scalar_value"]
                )
            if fluid_config["vector_value"] is not None:
                self.scalar_fluid = np.array(
                    fluid_config["vector_value"], dtype=control.precision
                )
        if phase_config is not None:
            self.type_phase = phase_config["type"]
            self.scalar_phase = control.precision(0)
            self.vector_phase = np.zeros(2)
            if phase_config["scalar_value"] is not None:
                self.scalar_phase = control.precision(
                    phase_config["scalar_value"]
                )
            if phase_config["vector_value"] is not None:
                self.vector_phase = np.array(
                    phase_config["vector_value"], dtype=control.precision
                )
        if self.location == "bottom":
            self.surface_normals = np.array([0, 1], dtype=control.precision)
            self.out_list = np.array([4, 7, 8], dtype=np.int32)
            self.inv_list = np.array([2, 5, 6], dtype=np.int32)
        elif self.location == "top":
            self.surface_normals = np.array([0, -1], dtype=control.precision)
            self.out_list = np.array([2, 5, 6], dtype=np.int32)
            self.inv_list = np.array([4, 7, 8], dtype=np.int32)
        elif self.location == "left":
            self.surface_normals = np.array([1, 0], dtype=control.precision)
            self.out_list = np.array([3, 6, 7], dtype=np.int32)
            self.inv_list = np.array([1, 8, 5], dtype=np.int32)
        elif self.location == "right":
            self.surface_normals = np.array([-1, 0], dtype=control.precision)
            self.out_list = np.array([1, 8, 5], dtype=np.int32)
            self.inv_list = np.array([3, 6, 7], dtype=np.int32)
        # TODO: add rank specific boundary node allocation
        self.boundary_nodes = self.allocate_boundary_nodes(
            segment,
            domain
        )

    def allocate_boundary_nodes(
        self,
        segment,
        domain
    ):
        """
        Allocates nodes at the boundary to the corresponding rank
        Args:
            segment: list
            domain: domain object
        Returns:
            boundary_nodes: int array
        """
        boundary_nodes = []
        x_min, y_min = segment[0]
        x_max, y_max = segment[1]
        if self.orientation == "horizontal":
            j_global = y_min
            for i_global in range(x_min, x_max + 1):  # includes [x_min,x_max]
                i, j = global_to_local(i_global, j_global, domain.offset)
                if (i < 0 or j < 0 or
                        i >= (domain.shape[0] - 2) or
                        j >= (domain.shape[1] - 2)):
                    pass
                else:
                    ind = (i + 1) * domain.shape[1] + (j + 1)
                    boundary_nodes.append(ind)
            boundary_nodes = np.array(boundary_nodes, dtype=np.int64)
            return boundary_nodes
        if self.orientation == "vertical":
            i_global = x_min
            for j_global in range(y_min, y_max + 1):  # includes [y_min,y_max]
                i, j = global_to_local(i_global, j_global, domain.offset)
                if (i < 0 or j < 0 or
                        i >= (domain.shape[0] - 2) or
                        j >= (domain.shape[1] - 2)):
                    pass
                else:
                    ind = (i + 1) * domain.shape[1] + (j + 1)
                    boundary_nodes.append(ind)
            boundary_nodes = np.array(boundary_nodes, dtype=np.int64)
            return boundary_nodes


class Boundary:
    def __init__(
        self,
        simulation,
        mesh,
        domain,
        control,
        fluid=False,
        phase=False,
        scalar=False,
        verbose=True
    ):
        """
        Container for simulation domain boundary information
        Attributes:

        """
        print_log("-" * 80, domain.mpi_rank, verbose)
        print_log("Setting up domain boundaries...\n",
                  domain.mpi_rank, verbose)
        if not hasattr(simulation, "boundary_dict"):
            raise ValueError(
                "boundary_dict not found in simulation.py file"
            )
        self.boundary_dict = simulation.boundary_dict
        self.compute_forces = False
        self.boundary_elements = []

        # ------- Read and initialize boundary elements ------- #
        self.read_boundary_dict(
            mesh,
            domain,
            control,
            fluid=fluid,
            phase=phase,
            scalar=scalar,
            verbose=verbose
        )

    def read_boundary_dict(
        self,
        mesh,
        domain,
        control,
        fluid=False,
        phase=False,
        scalar=False,
        verbose=True
    ):
        """
        Reads boundary_dict and creates boundary elements
        Args:

        Returns:

        """
        if "options" not in self.boundary_dict:
            raise ValueError("options missing in boundary_dict")
        options_dict = self.boundary_dict["options"]
        self.read_options_dict(
            options_dict,
            domain,
            verbose
        )
        for key_no, key in enumerate(self.boundary_dict.keys()):
            if key == "options":
                continue
            user_boundary_name = key
            user_boundary_dict = self.boundary_dict[key]
            self.read_user_boundary_dict(
                user_boundary_name,
                user_boundary_dict,
                mesh,
                domain,
                control,
                fluid=fluid,
                phase=phase,
                scalar=scalar,
                verbose=verbose
            )

    def read_options_dict(
        self,
        options_dict,
        domain,
        verbose
    ):
        """
        Reads and sets options for boundary_dict
        Args:

        Returns:

        """
        print_log("Setting boundary options", domain.mpi_rank, verbose)
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
        print_log("Boundary options set\n", domain.mpi_rank, verbose)

    def read_user_boundary_dict(
        self,
        user_boundary_name,
        user_boundary_dict,
        mesh,
        domain,
        control,
        fluid=False,
        phase=False,
        scalar=False,
        verbose=True
    ):
        """
        Reads user-defined boundary dictionaries and
        creates boundary elements for each boundaries
        Args:

        Returns:

        """
        fluid_config = {
            "type": None,
            "scalar_value": None,
            "vector_value": None
        }
        phase_config = {
            "type": None,
            "scalar_value": None,
            "vector_value": None
        }
        if "entity" not in user_boundary_dict:
            # TODO: add something like entity later
            # raise ValueError(
            #     "entity missing in boundary: " + user_boundary_name
            # )
            pass
        # ------- Validate segments entry ------- #
        if "segments" not in user_boundary_dict:
            raise ValueError(
                "segments missing in boundary: " + user_boundary_name
            )
        segments = user_boundary_dict["segments"]
        if not isinstance(segments, list):
            raise ValueError(
                "segments must be a list object: " + user_boundary_name
            )
        if len(segments) == 0:
            raise ValueError(
                "segments cannot be an empty list: " + user_boundary_name
            )
        for segment in segments:
            if not (
                isinstance(segment, list) and
                len(segment) == 2 and
                all(isinstance(pt, list) and len(pt) == 2 for pt in segment)
            ):
                raise ValueError(
                    "segment must have structure [[x1, y1], [x2, y2]]: " +
                    user_boundary_name
                )

            (x1, y1), (x2, y2) = segment

            # enforce ordering
            if not (x2 >= x1 and y2 >= y1):
                raise ValueError(
                    "segment must satisfy x2 >= x1 and y2 >= y1: " +
                    user_boundary_name
                )

            # enforce axis-aligned constraint
            if not (x1 == x2 or y1 == y2):
                raise ValueError(
                    "segment must be axis-aligned (horizontal or vertical): " +
                    user_boundary_name
                )

        # ------- Validate fluid dict ------- #
        if fluid:
            if "fluid" not in user_boundary_dict:
                raise ValueError(
                    "fluid missing in boundary: " + user_boundary_name
                )
            fluid_dict = user_boundary_dict["fluid"]
            if "type" not in fluid_dict:
                raise ValueError(
                    "type missing in fluid section of boundary: "
                    + user_boundary_name
                )
            boundary_type = fluid_dict["type"]
            supported_fluid_bcs = [
                "bounce_back",
                "fixed_velocity",
                "fixed_pressure",
                "periodic"
            ]
            if boundary_type not in supported_fluid_bcs:
                raise ValueError(
                    "Unsupported boundary condition for fluid: " +
                    boundary_type
                )
            fluid_config["type"] = boundary_type
            if boundary_type == "fixed_velocity":
                if "value" not in fluid_dict:
                    raise ValueError(
                        "value missing in fluid section of boundary: "
                        + user_boundary_name
                    )
                boundary_value = fluid_dict["value"]
                if (not isinstance(boundary_value, list) or
                        len(boundary_value) != 2):
                    raise ValueError(
                        "value for fixed_velocity must be a list (ux, uy)"
                        + user_boundary_name
                    )
                fluid_config["vector_value"] =\
                    np.array(boundary_value, dtype=control.precision)
            elif boundary_type == "fixed_pressure":
                if "value" not in fluid_dict:
                    raise ValueError(
                        "value missing in fluid section of boundary: "
                        + user_boundary_name
                    )
                boundary_value = fluid_dict["value"]
                if (not isinstance(boundary_value, int) and
                        not isinstance(boundary_value, float)):
                    raise ValueError(
                        "value for fixed_pressure must be a float or int"
                        + user_boundary_name
                    )
                fluid_config["scalar_value"] =\
                    control.precision(boundary_value)
            elif boundary_type == "periodic":
                self.validate_periodic_boundary(
                    user_boundary_name,
                    segments,
                    mesh
                )

        # ------- Validate phase dict ------- #
        if phase:
            if "phase" not in user_boundary_dict:
                raise ValueError(
                    "phase missing in boundary: " + user_boundary_name
                )
            phase_dict = user_boundary_dict["phase"]
            if "type" not in phase_dict:
                raise ValueError(
                    "type missing in phase section of boundary: "
                    + user_boundary_name
                )
            boundary_type = fluid_dict["type"]
            supported_phase_bcs = [
                "bounce_back",
                "fixed_value",
                "periodic"
            ]
            if boundary_type not in supported_phase_bcs:
                raise ValueError(
                    "Unsupported boundary condition for phase: " +
                    boundary_type
                )
            phase_config["type"] = boundary_type
            if boundary_type == "fixed_value":
                if "value" not in phase_dict:
                    raise ValueError(
                        "value missing in phase section of boundary: "
                        + user_boundary_name
                    )
                boundary_value = phase_dict["value"]
                if (not isinstance(boundary_value, int) and
                        not isinstance(boundary_value, float)):
                    raise ValueError(
                        "value for fixed_value must be a float or int"
                        + user_boundary_name
                    )
                phase_config["scalar_value"] =\
                    control.precision(boundary_value)
            elif boundary_type == "periodic":
                self.validate_periodic_boundary(
                    user_boundary_name,
                    segments,
                    mesh
                )
        # ------- Validate scalar dict ------- #
        if scalar:
            if "scalar_value" not in user_boundary_dict:
                # TODO: no scalar transport solver implemented
                pass

        # ------- Create boundary elements ------- #
        orientations = []
        locations = []
        for segment_no, segment in enumerate(segments):
            # TODO: modifications needed for 1D case
            (x1, y1), (x2, y2) = segment
            if x2 - x1 == 0:
                orientations.append("vertical")
                if x2 == mesh.grid_global_shape[0] - 1:
                    locations.append("right")
                elif x2 == 0:
                    locations.append("left")
            elif y2 - y1 == 0:
                orientations.append("horizontal")
                if y2 == mesh.grid_global_shape[1] - 1:
                    locations.append("top")
                elif y2 == 0:
                    locations.append("bottom")
            boundary_element = BoundaryElement(
                user_boundary_name,
                segment,
                segment_no,
                orientations[segment_no],
                locations[segment_no],
                domain,
                control,
                fluid_config=fluid_config,
                phase_config=phase_config
            )
            self.boundary_elements.append(boundary_element)

        self.log_boundary_data(
            user_boundary_name,
            segments,
            orientations,
            locations,
            domain,
            fluid_config=fluid_config,
            phase_config=phase_config,
            verbose=verbose
        )

    def validate_periodic_boundary(
        self,
        user_boundary_name,
        segments,
        mesh
    ):
        """
        Validates periodic boundary input from user
        Args:

        Returns:

        """
        if len(segments) != 2:
            raise ValueError(
                "For periodic boundary, each segment should be" +
                " followed by a corresponding periodic pair: " +
                user_boundary_name
            )
        segment_1 = segments[0]
        segment_2 = segments[1]

        lx1 = np.abs(segment_1[1][0] - segment_1[0][0])
        ly1 = np.abs(segment_1[1][1] - segment_1[0][1])
        lx2 = np.abs(segment_2[1][0] - segment_2[0][0])
        ly2 = np.abs(segment_2[1][1] - segment_2[0][1])

        # check orientation
        is_horizontal = ly1 == 0
        if is_horizontal:
            if ly2 != 0:
                raise ValueError(
                    "invalid periodic pair - different orientation: " +
                    user_boundary_name
                )
        is_vertical = lx1 == 0
        if is_vertical:
            if lx2 != 0:
                raise ValueError(
                    "invalid periodic pair - different orientation: " +
                    user_boundary_name
                )

        # check length
        if not (np.isclose(lx1, lx2) and np.isclose(ly1, ly2)):
            raise ValueError(
                "invalid periodic pair - unequal length segments: " +
                user_boundary_name
            )

        # check periodic connection
        (x1_min, y1_min), (x1_max, y1_max) = segment_1
        (x2_min, y2_min), (x2_max, y2_max) = segment_2
        if is_horizontal:
            if (x1_min, x1_max) != (x2_min, x2_max):
                raise ValueError(
                    "horizontal periodic boundaries must span" +
                    " same x-range : " + user_boundary_name
                )
            if sorted([y1_min, y2_min]) != [0, mesh.grid_global_shape[1] - 1]:
                raise ValueError(
                    "horizontal periodic boundaries must connect" +
                    " top-bottom boundaries : " + user_boundary_name
                )
        elif is_vertical:
            if (y1_min, y1_max) != (y2_min, y2_max):
                raise ValueError(
                    "vertical periodic boundaries must span" +
                    " same y-range : " + user_boundary_name
                )
            if sorted([x1_min, x2_min]) != [0, mesh.grid_global_shape[0] - 1]:
                raise ValueError(
                    "vertical periodic boundaries must connect" +
                    " left-right boundaries : " + user_boundary_name
                )

    def log_boundary_data(
        self,
        user_boundary_name,
        segments,
        orientations,
        locations,
        domain,
        fluid_config=None,
        phase_config=None,
        verbose=True
    ):
        """
        Prints user-defined boundary data to std-out after validation
        Args:

        Returns:

        """
        print_log("boundary name: " + user_boundary_name,
                  domain.mpi_rank, verbose)
        if fluid_config is not None:
            print_log("fluid boundary condition:" + fluid_config["type"],
                      domain.mpi_rank, verbose)
            if fluid_config["scalar_value"] is not None:
                print_log("fluid value:" + str(fluid_config["scalar_value"]),
                          domain.mpi_rank, verbose)
            if fluid_config["vector_value"] is not None:
                print_log("fluid value:" + str(fluid_config["vector_value"]),
                          domain.mpi_rank, verbose)
        if phase_config is not None:
            print_log("phase boundary condition:" + phase_config["type"],
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
                "segement_" + str(segment_no + 1) + ":" +
                str(segments[segment_no]).ljust(30) + "| orientation: " +
                orientations[segment_no].ljust(5) + " | location: " +
                str(locations[segment_no]),
                domain.mpi_rank,
                verbose
            )
        print_log("\n", domain.mpi_rank, verbose)
