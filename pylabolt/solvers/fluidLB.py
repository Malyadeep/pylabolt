import time

from mpi4py import MPI

from pylabolt.utils.inputOutput import loadState, saveState
import pylabolt.base.createSim as createSim
from pylabolt.base.baseAlgorithm import baseAlgorithm
from pylabolt.parallel.parallelSetup import parallelSetup
from pylabolt.parallel.MPI_decompose import distributeBoundaries_mpi


def main(parallelization, n_threads):
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    simulation = createSim.simulation(parallelization, rank, size, comm)
    if simulation.startTime != 0:
        temp = loadState(simulation.startTime)
        if temp is None:
            saveState(simulation.startTime, simulation)
        else:
            simulation.fields = temp
    elif simulation.startTime == 0:
        saveState(simulation.startTime, simulation)
    base = baseAlgorithm()
    parallel = parallelSetup(parallelization, n_threads, base, simulation)

    if parallel.mode != 'cuda':
        if size > 1:
            distributeBoundaries_mpi(simulation.boundary, simulation.mpiParams,
                                     simulation.mesh, rank, size, comm)
            # if rank == 1:
            #     simulation.boundary.details()
        base.jitWarmUp(simulation, size, rank, comm)
    if parallel.mode == 'cuda':
        start = time.perf_counter()
        base.solver_cuda(simulation, parallel)
        runTime = time.perf_counter() - start
    else:
        start = time.perf_counter()
        base.solver(simulation, size, rank, comm)
        runTime = time.perf_counter() - start
    if rank == 0:
        print('\nSimulation completed : Runtime = ' + str(runTime) + '\n')

    MPI.Finalize()
