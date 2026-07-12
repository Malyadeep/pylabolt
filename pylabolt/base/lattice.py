import numpy as np

from pylabolt.utils.helpers import print_log


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
            cs_2: float
            inv_cs_2: float
            inv_cs_4: float
            no_of_directions: int
            cx: (no_of_directions) int array
            cy: (no_of_directions) int array
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
        self.cs_2 = (self.cs * self.cs)
        self.inv_cs_2 = 1.0 / self.cs_2
        self.inv_cs_4 = self.inv_cs_2 * self.inv_cs_2
        if self.lattice_type == "D2Q9":
            if mesh.dimensions != 2:
                raise ValueError(
                    "grid dimensions and lattice type are incompatible"
                )
            self.cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1],
                               dtype=np.int32)
            self.cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1],
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
            self.cx = np.array([0, 1, -1], dtype=np.int32)
            self.cy = np.array([0, 0, 0], dtype=np.int32)
            self.weights = np.array([2/3, 1/6, 1/6], dtype=control.precision)
            self.inv_list = np.array([0, 2, 1], dtype=np.int32)
            self.no_of_directions = np.int32(3)
        else:
            raise ValueError(
                "Unsupported lattice type"
            )
        print_log("lattice type set: " + self.lattice_type, rank, verbose)

    def set_backend(
        self,
        backend
    ):
        """
        Configure backend attributes for lattice object
        Args:

        Returns:

        """
        if backend.backend_type == "gpu":
            self._device_attrs = [
                "cs",
                "cs_2",
                "inv_cs_2",
                "inv_cs_4",
                "no_of_directions",
                "cx",
                "cy",
                "weights",
                "inv_list"
            ]
            for arg_name in self._device_attrs:
                arg_device = backend.allocate_to_device(
                    getattr(self, arg_name)
                )
                setattr(self, arg_name + "_device", arg_device)
