import inputOutput
import time

import createSim
from baseAlgorithm import solver


def main():
    simulation = createSim.simulation()
    if simulation.startTime != 0:
        temp = inputOutput.loadState(simulation.startTime)
        if temp is None:
            inputOutput.saveState(simulation.startTime, simulation)
        else:
            simulation = temp
    createSim.initializePopulations(simulation.mesh,
                                    simulation.elements,
                                    simulation.equilibriumFunc)

    start = time.perf_counter()
    solver(simulation)
    runTime = time.perf_counter() - start
    print('Simulation completed : Runtime = ' + str(runTime) + '\n')


if __name__ == '__main__':
    main()
