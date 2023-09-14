import time

from mpi4py import MPI

from pylabolt.utils.inputOutput import loadState, saveState
import pylabolt.solvers.phaseFieldLB.createSim as createSim
from pylabolt.solvers.phaseFieldLB.baseAlgorithm import baseAlgorithm
from pylabolt.solvers.phaseFieldLB.deviceFields import deviceData
from pylabolt.parallel.parallelSetup import parallelSetup


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
