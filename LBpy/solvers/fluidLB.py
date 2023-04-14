import time

from LBpy.utils.inputOutput import loadState, saveState
import LBpy.base.createSim as createSim
from LBpy.base.baseAlgorithm import baseAlgorithm
from LBpy.parallel.parallelSetup import parallelSetup


def main(parallelization, n_threads):
    simulation = createSim.simulation(parallelization)
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
    base.jitWarmUp(simulation)

    if parallel.mode == 'cuda':
        start = time.perf_counter()
        base.solver_cuda(simulation, parallel.device, parallel)
        runTime = time.perf_counter() - start
    else:
        start = time.perf_counter()
        base.solver(simulation)
        runTime = time.perf_counter() - start
    print('Simulation completed : Runtime = ' + str(runTime) + '\n')
