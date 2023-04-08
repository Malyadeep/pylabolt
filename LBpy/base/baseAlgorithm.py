import numba
import numpy as np

from LBpy.utils.inputOutput import writeFields
from LBpy.base.schemeLB import stream


@numba.njit
def equilibriumRelaxation(Nx, Ny, f_eq, f, f_new, u, rho,
                          collisionFunc, equilibriumFunc, preFactor,
                          eqilibriumArgs):
    for ind in range(Nx * Ny):
        equilibriumFunc(f_eq[ind, :], u[ind, :],
                        rho[ind], *eqilibriumArgs)

        collisionFunc(f[ind, :], f_new[ind, :],
                      f_eq[ind, :], preFactor)


@numba.njit
def computeFields(Nx, Ny, f_new, u, rho, c,
                  noOfDirections, u_old, rho_old):
    u_sq, u_err_sq = 0, 0
    v_sq, v_err_sq = 0, 0
    rho_sq, rho_err_sq = 0, 0
    for ind in range(Nx * Ny):
        rhoSum = 0
        uSum = 0
        vSum = 0
        for k in range(noOfDirections):
            rhoSum += f_new[ind, k]
            uSum += c[k, 0] * f_new[ind, k]
            vSum += c[k, 1] * f_new[ind, k]
        rho[ind] = rhoSum
        u[ind, 0] = uSum/(rho[ind] + 1e-9)
        u[ind, 1] = vSum/(rho[ind] + 1e-9)

        # Residue calculation
        u_err_sq += (u[ind, 0] - u_old[ind, 0]) * \
            (u[ind, 0] - u_old[ind, 0])
        u_sq += u_old[ind, 0] * u_old[ind, 0]
        v_err_sq += (u[ind, 1] - u_old[ind, 1]) * \
            (u[ind, 1] - u_old[ind, 1])
        v_sq += u_old[ind, 1] * u_old[ind, 1]
        rho_err_sq += (rho[ind] - rho_old[ind]) * \
            (rho[ind] - rho_old[ind])
        rho_sq += rho_old[ind] * rho_old[ind]
        u_old[ind, 0] = u[ind, 0]
        u_old[ind, 1] = u[ind, 1]
        rho_old[ind] = rho[ind]
    resU = np.sqrt(u_err_sq/(u_sq + 1e-8))
    resV = np.sqrt(v_err_sq/(v_sq + 1e-8))
    resRho = np.sqrt(rho_err_sq/(rho_sq + 1e-8))
    return resU, resV, resRho


def solver(simulation):
    resU, resV = 1e6, 1e6
    resRho = 1e6
    u_old = np.zeros((simulation.mesh.Nx * simulation.mesh.Ny, 2),
                     dtype=np.float64)
    rho_old = np.zeros((simulation.mesh.Nx * simulation.mesh.Ny),
                       dtype=np.float64)
    for timeStep in range(simulation.startTime, simulation.endTime,
                          simulation.lattice.deltaT):
        resU, resV, resRho = computeFields(*simulation.computeFieldsArgs,
                                           u_old, rho_old)
        if timeStep % simulation.stdOutputInterval == 0:
            print('timeStep = ' + str(timeStep), flush=True)
            print('resU = ' + str(resU), flush=True)
            print('resV = ' + str(resV), flush=True)
            print('resRho = ' + str(resRho) + '\n\n', flush=True)
        if timeStep % simulation.saveInterval == 0:
            writeFields(timeStep, simulation.fields)
        if (resU < simulation.relTolU and resV < simulation.relTolV and
                resRho < simulation.relTolRho):
            print('Convergence Criteria matched!!', flush=True)
            print('timeStep = ' + str(timeStep), flush=True)
            print('resU = ' + str(resU), flush=True)
            print('resV = ' + str(resV), flush=True)
            print('resRho = ' + str(resRho) + '\n\n', flush=True)
            writeFields(timeStep, simulation.fields)
            break

        equilibriumRelaxation(*simulation.collisionArgs)

        stream(*simulation.streamArgs)

        simulation.setBoundaryFunc(simulation.fields, simulation.lattice,
                                   simulation.mesh)
