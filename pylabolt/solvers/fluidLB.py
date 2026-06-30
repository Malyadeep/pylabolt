import time
from mpi4py import MPI

from pylabolt.utils.helpers import load_simulation, print_log
from pylabolt.parallel.backend import Backend
from pylabolt.parallel.MPI_operator import MPIOperator
from pylabolt.base.state import State
from pylabolt.base.obstacle_operator import ObstacleOperator
from pylabolt.base.collision_operator import CollisionOperator
from pylabolt.base.streaming_operator import StreamingOperator
from pylabolt.base.boundary_operator import BoundaryOperator


class FluidLB:
    def __init__(self):
        self.equilibrium_models = {
            "fluid": ["density_based_second_order"]
        }
        self.forcing_models = {
            "fluid": [
                None,
                "guo_linear",
                "guo_second_order"
            ]
        }
        self.collision_models = {
            "fluid": [
                "BGK",
                "MRT"
            ]
        }
        self.streaming_type = {
            "fluid": "scalar_based"
        }
        self.boundary_condition_type = {
            "fluid": "density_based"
        }

    def get_collision_args(self):
        return {
            "fluid": {
                "domain": ["size"],
                "lattice": [
                    "cx", "cy", "weights", "no_of_directions", "inv_cs_2", "inv_cs_4"
                ],
                "fields": [
                    "solid", "ghost_node", "density", "velocity", "pop_fluid",
                    "pop_fluid_new"
                ]
            }
        }

    def get_streaming_args(self):
        return {
            "fluid": {
                "domain": ["size", "shape"],
                "lattice": [
                    "cx", "cy", "weights", "inv_list", "no_of_directions",
                    "inv_cs_2"
                ],
                "fields": [
                    "solid", "ghost_node", "density", "velocity", "pop_fluid",
                    "pop_fluid_new"
                ]
            }
        }


class Solver:
    def __init__(self, comm, backend, n_threads):
        mpi_rank = comm.Get_rank()
        from importlib.metadata import version
        print_log(
            f"\n{'PyLaBolt':<10}: {version("pylabolt")}",
            mpi_rank, verbose=True
        )
        print_log(f"{'Solver':<10}: fluidLB", mpi_rank, verbose=True)
        self.model = FluidLB()
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
        self.collision_operator = CollisionOperator(
            comm,
            simulation,
            self.model,
            self.state,
            self.backend
        )
        self.streaming_operator = StreamingOperator(
            self.model,
            self.state,
            self.backend
        )
        self.boundary_operator = BoundaryOperator(
            self.model,
            self.state,
            self.backend
        )


def main(backend, n_threads):
    MPI.Init()
    comm = MPI.COMM_WORLD
    solver = Solver(comm, backend, n_threads)
    MPI.Finalize()
