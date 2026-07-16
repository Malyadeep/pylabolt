import time
from mpi4py import MPI

from pylabolt.utils.helpers import (
    load_simulation,
    print_log,
    SimulationStatusLogger
)
from pylabolt.utils.residues import ResidueOperator
from pylabolt.utils.io_operator import InputOutputOperator
from pylabolt.parallel.backend import Backend
from pylabolt.parallel.MPI_operator import MPIOperator
from pylabolt.base.state import State
from pylabolt.base.obstacle_operator import ObstacleOperator
from pylabolt.base.collision_operator import CollisionOperator
from pylabolt.base.compute_fields_operator import ComputeFieldsOperator
from pylabolt.base.streaming_operator import StreamingOperator
from pylabolt.base.boundary_operator import BoundaryOperator


class FluidLB:
    def __init__(self):
        self.solver_name = "fluidLB"
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
        self.compute_fields_type = {
            "fluid": "density_based"
        }
        self.residue_fields = ["density", "velocity"]
        self.save_fields = ["density", "velocity", "solid", "solid_id"]

    def get_collision_args(self):
        return {
            "collision": {
                "fluid": {
                    "domain": ["size"],
                    "lattice": [
                        "cx", "cy", "weights", "no_of_directions", "inv_cs_2",
                        "inv_cs_4"
                    ],
                    "fields": [
                        "solid", "ghost_node", "density", "velocity",
                        "pop_fluid", "pop_fluid_new"
                    ]
                }
            },
            "initialization": {
                "fluid": {
                    "domain": ["size"],
                    "lattice": [
                        "cx", "cy", "weights", "no_of_directions", "inv_cs_2",
                        "inv_cs_4"
                    ],
                    "fields": [
                        "solid", "ghost_node", "density", "velocity",
                        "pop_fluid", "pop_fluid_new"
                    ]
                }
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

    def get_compute_fields_args(self):
        return {
            "fluid": {
                "domain": ["size"],
                "lattice": [
                    "cx", "cy", "no_of_directions"
                ],
                "fields": [
                    "solid", "ghost_node", "density", "velocity",
                    "pop_fluid_new"
                ]
            }
        }


class Solver:
    def __init__(self, comm, backend, n_threads):
        mpi_rank = comm.Get_rank()
        from importlib.metadata import version
        print_log(
            f"\n{'PyLaBolt':<10}: {version('pylabolt')}",
            mpi_rank, verbose=True
        )
        print_log(f"{'Solver':<10}: fluidLB", mpi_rank, verbose=True)
        self.model = FluidLB()
        simulation = load_simulation(comm, mpi_rank)
        self.backend = Backend(
            backend,
            n_threads,
            mpi_rank
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
            self.state,
            self.backend,
            self.mpi_operator
        )
        self.collision_operator = CollisionOperator(
            simulation,
            self.model,
            self.state,
            self.mpi_operator
        )
        self.compute_fields_operator = ComputeFieldsOperator(
            self.model,
            self.state,
            self.collision_operator
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
        self.residue_operator = ResidueOperator(
            self.model,
            self.state,
            self.backend,
            self.mpi_operator
        )
        self.io_operator = InputOutputOperator(
            self.model,
            self.state,
            self.backend,
            self.mpi_operator
        )
        self.logger = SimulationStatusLogger(
            self.state.domain.mpi_rank,
            verbose=True
        )

    def set_backend(self, verbose=True):
        print_log("-" * 80, self.state.domain.mpi_rank, verbose)
        print_log("Setting simulation backend...\n",
                  self.state.domain.mpi_rank, verbose)

        self.state.set_backend(self.backend)
        self.mpi_operator.set_backend(self.state, self.backend)
        # self.obstacle_operator.set_backend(self.backend)
        self.collision_operator.set_backend(self.state, self.backend)
        self.compute_fields_operator.set_backend(self.state, self.backend)
        self.streaming_operator.set_backend(self.state, self.backend)
        self.boundary_operator.set_backend(self.state, self.backend)
        self.residue_operator.set_backend(self.state, self.backend)
        self.io_operator.set_backend(self.state, self.backend)

        print_log("\nSetting simulation backend done!",
                  self.state.domain.mpi_rank, verbose)
        print_log("-" * 80, self.state.domain.mpi_rank, verbose)

    def compile(self, verbose=True):
        print_log("-" * 80, self.state.domain.mpi_rank, verbose)
        print_log("JIT Compilation starts...\n",
                  self.state.domain.mpi_rank, verbose)

        self.mpi_operator.compile(self.state, self.backend)
        self.collision_operator.compile(self.state, self.backend)
        self.compute_fields_operator.compile(self.state, self.backend)
        self.streaming_operator.compile(self.state, self.backend)
        self.boundary_operator.compile(self.state, self.backend)
        self.residue_operator.compile(self.state, self.backend)
        self.io_operator.compile(self.state, self.backend)

        print_log("\nJIT Compilation done!",
                  self.state.domain.mpi_rank, verbose)
        print_log("-" * 80, self.state.domain.mpi_rank, verbose)

    def verify_kernel_signatures(self, verbose=True):
        print_log("-" * 80, self.state.domain.mpi_rank, verbose)
        print_log("Kernel signature verification selected, check starts...\n",
                  self.state.domain.mpi_rank, verbose)

        args = (self.state, self.backend)
        self.mpi_operator.verify_kernel_signatures(*args)
        self.collision_operator.verify_kernel_signatures(*args)
        self.streaming_operator.verify_kernel_signatures(*args)
        self.compute_fields_operator.verify_kernel_signatures(*args)
        self.boundary_operator.verify_kernel_signatures(*args)
        self.residue_operator.verify_kernel_signatures(*args)
        self.io_operator.verify_kernel_signatures(*args)

        print_log("\nKernel signature verification done!",
                  self.state.domain.mpi_rank, verbose)
        print_log("-" * 80, self.state.domain.mpi_rank, verbose)

    def run(
        self,
        verbose=True
    ):
        print_log("\n" + "-" * 80, self.state.domain.mpi_rank, verbose)
        print_log(
            "Running simulation...\n", self.state.domain.mpi_rank,
            verbose
        )
        self.io_operator.write_fields(
            self.state,
            self.state.control.start_time
        )
        self.collision_operator.initialize_pop(self.state, self.backend)

        run_time_start = time.perf_counter()

        for time_step in range(
            self.state.control.start_time + 1,
            self.state.control.end_time + 1,
        ):
            self.collision_operator.collide(self.state, fluid=True)
            self.mpi_operator.halo_exchange(
                self.state,
                float_buffers=["pop_fluid"]
            )
            self.streaming_operator.stream(self.state, fluid=True)
            self.boundary_operator.set_boundary(self.state, fluid=True)
            self.compute_fields_operator.compute_fields(self.state, fluid=True)
            if time_step % self.state.control.std_out_interval == 0:
                self.residue_operator.compute_residues(
                    self.state,
                    self.mpi_operator
                )
                self.logger.log_data(
                    time_step,
                    res_density=self.residue_operator.residues["res_density"],
                    res_velocity=self.residue_operator.residues["res_velocity"]
                )
            if time_step % self.state.control.save_interval == 0:
                self.io_operator.write_fields(self.state, time_step)

        run_time = time.perf_counter() - run_time_start
        print_log("\n" + "-" * 80, self.state.domain.mpi_rank, verbose)
        print_log(
            f"{'Simulation complete, run time':<30}: "
            f"{str(run_time) + ' s':<30}",
            self.state.domain.mpi_rank,
            verbose
        )
        print_log("-" * 80, self.state.domain.mpi_rank, verbose)


def main(backend, n_threads, debug_mode=False):
    MPI.Init()
    comm = MPI.COMM_WORLD
    solver = Solver(comm, backend, n_threads)
    solver.set_backend()
    solver.compile()
    solver.run()
    if debug_mode:
        solver.verify_kernel_signatures()
    MPI.Finalize()
