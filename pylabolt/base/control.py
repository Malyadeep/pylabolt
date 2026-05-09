import numpy as np

from pylabolt.utils.IO import print_log


class Control:
    def __init__(
        self,
        simulation,
        rank,
        verbose=True
    ):
        """
        Container for simulation control parameters
        Attributes:
            start_time: int
            end_time: int
            std_out_interval: int
            save_interval: int
            checkpoint_interval: int
            precision_type: str
            precision: dtype
        """
        if not hasattr(simulation, "control_dict"):
            raise ValueError("control_dict not found in simulation.py file")
        control_dict = simulation.control_dict

        print_log("-" * 80, rank, verbose=verbose)
        print_log("Setting control parameters...", rank, verbose=verbose)
        try:
            self.start_time = control_dict["start_time"]
            self.end_time = control_dict["end_time"]
            self.std_out_interval = control_dict["std_out_interval"]
            self.save_interval = control_dict["save_interval"]
            self.checkpoint_interval = control_dict["checkpoint_interval"]
            self.precision_type = control_dict["precision"]
            if self.precision_type == "single":
                self.precision = np.float32
            elif self.precision_type == "double":
                self.precision = np.float64
            else:
                raise ValueError(
                    "unsupported precision specified." +
                    "available precision (single, double)"
                )
        except KeyError as e:
            raise ValueError(e.args[0] + " missing in control_dict")
        if rank == 0:
            print_log("Setting control parameters done!", rank, verbose=verbose)
            print_log("-" * 80, rank, verbose=verbose)
