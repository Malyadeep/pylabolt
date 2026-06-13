import time
from mpi4py import MPI

from pylabolt.utils.helpers import load_simulation
from pylabolt.parallel.backend import Backend
from pylabolt.parallel.MPI_operator import MPIOperator
from pylabolt.base.state import State
from pylabolt.base.obstacle_operator import ObstacleOperator


class Solver:
    def __init__(self, comm, backend, n_threads):
        mpi_rank = comm.Get_rank()
        simulation = load_simulation(comm, mpi_rank)
        self.backend = Backend(
            backend,
            n_threads
        )
        self.state = State(
            simulation,
            comm,
            mpi_rank,
            fluid=True
        )
        self.mpi_operator = MPIOperator(
            comm,
            self.state,
        )
        self.obstacle_operator = ObstacleOperator(
            comm,
            self.state,
            self.backend,
            self.mpi_operator
        )


def main(backend, n_threads):
    MPI.Init()
    comm = MPI.COMM_WORLD
    solver = Solver(comm, backend, n_threads)
    MPI.Finalize()
