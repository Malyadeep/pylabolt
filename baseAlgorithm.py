import numba
import numpy as np

import inputOutput
from schemeLB import stream


@numba.njit
def equilibriumRelaxation(mesh, elements,
                          collisionFunc, preFactor, equilibriumFunc,
                          eqilibriumArgs):
    for ind in range(mesh.Nx * mesh.Ny):
        equilibriumFunc(elements[ind].f_eq, elements[ind].u,
                        elements[ind].rho, *eqilibriumArgs)

        collisionFunc(elements[ind].f, elements[ind].f_new,
                      elements[ind].f_eq, preFactor)


@numba.njit
def computeFields(elements, mesh, lattice, u_old, rho_old):
    u_sq, u_err_sq = 0, 0
    v_sq, v_err_sq = 0, 0
    rho_sq, rho_err_sq = 0, 0
    for ind in range(mesh.Nx * mesh.Ny):
        elements[ind].computeFields(lattice)
        u_err_sq += (elements[ind].u[0] - u_old[ind, 0]) * \
            (elements[ind].u[0] - u_old[ind, 0])
        u_sq += u_old[ind, 0] * u_old[ind, 0]
        v_err_sq += (elements[ind].u[1] - u_old[ind, 1]) * \
            (elements[ind].u[1] - u_old[ind, 1])
        v_sq += u_old[ind, 1] * u_old[ind, 1]
        rho_err_sq += (elements[ind].rho - rho_old[ind]) * \
            (elements[ind].rho - rho_old[ind])
        rho_sq += rho_old[ind] * rho_old[ind]
        u_old[ind, 0] = elements[ind].u[0]
        u_old[ind, 1] = elements[ind].u[1]
        rho_old[ind] = elements[ind].rho
    resU = np.sqrt(u_err_sq/(u_sq + 1e-8)) / (mesh.Nx * mesh.Ny + 1e-8)
    resV = np.sqrt(v_err_sq/(v_sq + 1e-8)) / (mesh.Nx * mesh.Ny + 1e-8)
    resRho = np.sqrt(rho_err_sq/(rho_sq + 1e-8)) / (mesh.Nx * mesh.Ny + 1e-8)
    return resU, resV, resRho


@numba.njit
def swapPopulations(elements, mesh, u_old, rho_old):
    for ind in range(mesh.Nx * mesh.Ny):
        elements[ind].f[:] = elements[ind].f_new[:]


def solver(simulation):
    resU, resV = 1e6, 1e6
    resRho = 1e6
    u_old = np.zeros((simulation.mesh.Nx * simulation.mesh.Ny, 2),
                     dtype=np.float64)
    print(np.max(u_old))
    rho_old = np.zeros((simulation.mesh.Nx * simulation.mesh.Ny),
                       dtype=np.float64)
    for timeStep in range(simulation.startTime, simulation.endTime,
                          simulation.lattice.deltaT):
        resU, resV, resRho = computeFields(simulation.elements,
                                           simulation.mesh, simulation.lattice,
                                           u_old, rho_old)
        # if (resU < simulation.relTolU and resV < simulation.relTolV and
        #         resRho < simulation.relTolRho):
        #     break
        if timeStep % simulation.stdOutputInterval == 0:
            print('timeStep = ' + str(timeStep), flush=True)
            print('resU = ' + str(resU), flush=True)
            print('resV = ' + str(resV), flush=True)
            print('resRho = ' + str(resRho) + '\n\n', flush=True)
        if timeStep % simulation.saveInterval == 0:
            inputOutput.writeFields(timeStep, simulation.elements)
        # if timeStep % simulation.saveStateInterval == 0:
        #     inputOutput.saveState(timeStep, simulation)

        equilibriumRelaxation(simulation.mesh, simulation.elements,
                              simulation.collisionFunc,
                              simulation.collisionScheme.preFactor,
                              simulation.equilibriumFunc,
                              simulation.equilibriumArgs
                              )
        stream(simulation.elements, simulation.lattice,
               simulation.mesh)
        simulation.setBoundaryFunc(simulation.elements, simulation.lattice,
                                   simulation.mesh)

        swapPopulations(simulation.elements,
                        simulation.mesh, u_old, rho_old)
