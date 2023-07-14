import time

from mpi4py import MPI

from pylabolt.utils.inputOutput import loadState, saveState
import pylabolt.solvers.fluidLB.createSim as createSim
from pylabolt.solvers.fluidLB.baseAlgorithm import baseAlgorithm
from pylabolt.solvers.fluidLB.deviceFields import deviceData
from pylabolt.parallel.parallelSetup import parallelSetup
from pylabolt.parallel.MPI_decompose import (distributeBoundaries_mpi,
                                             distributeForceNodes_mpi)


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
        if simulation.saveStateInterval is not None:
            saveState(simulation.startTime, simulation)
    base = baseAlgorithm()
    parallel = parallelSetup(parallelization, n_threads)
    if parallel.mode is None:
        parallel.setupParallel(base, simulation, parallel=False)
    elif parallel.mode == 'openmp':
        parallel.setupParallel(base, simulation, parallel=True)
    elif parallel.mode == 'cuda':
        device = deviceData(simulation.mesh, simulation.lattice,
                            simulation.precision)
        parallel.setupCuda(simulation, device)

    if parallel.mode != 'cuda':
        if size > 1:
            distributeBoundaries_mpi(simulation.boundary, simulation.mpiParams,
                                     simulation.mesh, rank, size,
                                     simulation.precision, comm)
            # if rank == 1:
            #     simulation.boundary.details()
            if (simulation.options.computeForces is True or
                    simulation.options.computeTorque is True):
                distributeForceNodes_mpi(simulation,
                                         rank, size, comm)
            # if rank == 0:
            #     simulation.options.details(simulation.rank, simulation.mesh,
            #                                simulation.fields.solid,
            #                                simulation.fields.u, flag='local')
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
        print('\nSimulation completed : Runtime = ' + str(runTime) + ' s \n',
              flush=True)

    MPI.Finalize()
