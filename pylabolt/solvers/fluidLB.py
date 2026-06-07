import time
from mpi4py import MPI

from pylabolt.utils.helpers import load_simulation
from pylabolt.base.state import State
from pylabolt.base.obstacle_operator import ObstacleOperator


class Solver:
    def __init__(self, comm):
        mpi_rank = comm.Get_rank()
        simulation = load_simulation(comm, mpi_rank)
        self.state = State(
            simulation,
            comm,
            mpi_rank,
            fluid=True
        )
        self.obstacle_operator = ObstacleOperator(self.state)


def main():
    MPI.Init()
    comm = MPI.COMM_WORLD
    solver = Solver(comm)
    MPI.Finalize()
