import os
import sys


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
