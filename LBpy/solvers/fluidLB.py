import time

from LBpy.utils.inputOutput import loadState, saveState
import LBpy.base.createSim as createSim
from LBpy.base.baseAlgorithm import solver


def main():
    simulation = createSim.simulation()
    if simulation.startTime != 0:
        temp = loadState(simulation.startTime)
        if temp is None:
            saveState(simulation.startTime, simulation)
        else:
            simulation.fields = temp
    elif simulation.startTime == 0:
        saveState(simulation.startTime, simulation)

    start = time.perf_counter()
    solver(simulation)
    runTime = time.perf_counter() - start
    print('Simulation completed : Runtime = ' + str(runTime) + '\n')
