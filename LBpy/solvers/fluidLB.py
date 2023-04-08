# import inputOutput
import time

import LBpy.base.createSim as createSim
from LBpy.base.baseAlgorithm import solver


def main():
    simulation = createSim.simulation()
    # if simulation.startTime != 0:
    #     temp = inputOutput.loadState(simulation.startTime)
    #     if temp is None:
    #         inputOutput.saveState(simulation.startTime, simulation)
    #     else:
    #         simulation = temp
    createSim.initializePopulations(simulation.fields,
                                    simulation.mesh,
                                    simulation.equilibriumFunc,
                                    simulation.equilibriumArgs)

    start = time.perf_counter()
    solver(simulation)
    runTime = time.perf_counter() - start
    print('Simulation completed : Runtime = ' + str(runTime) + '\n')

