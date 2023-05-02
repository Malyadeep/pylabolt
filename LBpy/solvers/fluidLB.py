import time

from mpi4py import MPI

from LBpy.utils.inputOutput import loadState, saveState
import LBpy.base.createSim as createSim
from LBpy.base.baseAlgorithm import baseAlgorithm
from LBpy.parallel.parallelSetup import parallelSetup
from LBpy.parallel.MPI_comm import distributeBoundaries_mpi


def main(parallelization, n_threads):
    if parallelization != 'cuda':
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
            # if rank == 3:
            #     simulation.boundary.details()
        # if rank == 0:
        #     base.jitWarmUp(simulation, size, rank, comm)
    if parallel.mode == 'cuda':
        start = time.perf_counter()
        base.solver_cuda(simulation, parallel)
        runTime = time.perf_counter() - start
    else:
        start = time.perf_counter()
        base.solver(simulation, size, rank, comm)
        runTime = time.perf_counter() - start
    if rank == 0:
        print('Simulation completed : Runtime = ' + str(runTime) + '\n')
    if parallelization != 'cuda':
        MPI.Finalize()
