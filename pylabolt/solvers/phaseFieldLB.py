import time
from mpi4py import MPI

from pylabolt.base.state import State


class Solver:
    def __init__(self, comm):
        rank = comm.Get_rank()
        self.state = State(comm, rank, fluid=True, phase=True)


def main():
    MPI.Init()
    comm = MPI.COMM_WORLD
    solver = Solver(comm)
    MPI.Finalize()
