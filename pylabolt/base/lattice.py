import numpy as np

from pylabolt.utils.IO import print_log


class Lattice:
    def __init__(
        self,
        simulation,   # user-defined simulation module
        control,
        mesh,
        rank,
        verbose=True
    ):
        """
        Container for lattice specific information
        Attributes:
            lattice_type: str
            cs: float
            cs2: float
            cs_2: float
            cs_4: float
            no_of_directions: int
            c_x: (no_of_directions) int array
            c_y: (no_of_directions) int array
            weights: (no_of_directions) float array
            inv_list: (no_of_directions) int array
        """
        if not hasattr(simulation, "lattice_dict"):
            raise ValueError(
                "lattice_dict not found in simulation.py file"
            )
        lattice_dict = simulation.lattice_dict

        if "lattice_type" not in lattice_dict:
            raise ValueError(
                "lattice_type missing in lattice_dict"
            )
        self.lattice_type = lattice_dict["lattice_type"]

        self.cs = control.precision(1/np.sqrt(3))
        self.cs2 = (self.cs * self.cs)
        self.cs_2 = 1.0 / self.cs2
        self.cs_4 = self.cs_2 * self.cs_2
        if self.lattice_type == "D2Q9":
            if mesh.dimensions != 2:
                raise ValueError(
                    "grid dimensions and lattice type are incompatible"
                )
            self.c_x = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1],
                                dtype=np.int32)
            self.c_y = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1],
                                dtype=np.int32)
            self.weights = np.array(
                [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36],
                dtype=control.precision
            )
            self.inv_list = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6],
                                     dtype=np.int32)
            self.no_of_directions = np.int32(9)
        elif self.lattice_type == "D1Q3":
            if mesh.dimensions != 1:
                raise ValueError(
                    "grid dimensions and lattice type are incompatible"
                )
            self.c_x = np.array([0, 1, -1], dtype=np.int32)
            self.c_y = np.array([0, 0, 0], dtype=np.int32)
            self.weights = np.array([2/3, 1/6, 1/6], dtype=control.precision)
            self.invList = np.array([0, 2, 1], dtype=np.int32)
            self.no_of_directions = np.int32(3)
        else:
            raise ValueError(
                "Unsupported lattice type"
            )
        print_log("lattice type set: " + self.lattice_type, rank, verbose)
