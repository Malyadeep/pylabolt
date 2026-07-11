import os
import sys
import numpy as np


def print_log(mssg, mpi_rank, verbose):
    if verbose and mpi_rank == 0:
        print(mssg, flush=True)


def load_simulation(comm, mpi_rank):
    try:
        try:
            working_dir = os.getcwd()
            sys.path.append(working_dir)
            import simulation
            return simulation
        except ImportError:
            raise ImportError(
                "Missing simulation.py file in current working directory"
            )
    except Exception as e:
        print_log("-" * 80, mpi_rank, verbose=True)
        print_log("FATAL ERROR!", mpi_rank, verbose=True)
        print_log(str(e), mpi_rank, verbose=True)
        comm.Abort()


class SimulationStatusLogger:
    def __init__(
        self,
        mpi_rank,
        verbose=True
    ):
        self.mpi_rank = mpi_rank
        self.verbose = verbose

    def log_data(
        self,
        time_step,
        **values
    ):
        log_string = [f"{'time:':<5} {time_step:<10}"]
        for key, value in values.items():
            if np.isscalar(value):
                value_str = f"{value:.5e}"
            elif len(value) == 1:
                value_str = f"{value[0]:.5e}"
            else:
                value_str = "(" + ", ".join(f"{v:.5e}" for v in value) + ")"
            log_string.append(f"{key:<5}: {value_str}")
        print_log(" | ".join(log_string), self.mpi_rank, self.verbose)
