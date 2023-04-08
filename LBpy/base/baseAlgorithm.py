import numba
import numpy as np

from LBpy.utils.inputOutput import writeFields
from LBpy.base.schemeLB import stream


@numba.njit
def equilibriumRelaxation(mesh, fields, collisionFunc,
                          preFactor, equilibriumFunc,
                          eqilibriumArgs):
    for ind in range(mesh.Nx * mesh.Ny):
        equilibriumFunc(fields.f_eq[ind, :], fields.u[ind, :],
                        fields.rho[ind], *eqilibriumArgs)

        collisionFunc(fields.f[ind, :], fields.f_new[ind, :],
                      fields.f_eq[ind, :], preFactor)


@numba.njit
def computeFields(fields, mesh, lattice, u_old, rho_old):
    u_sq, u_err_sq = 0, 0
    v_sq, v_err_sq = 0, 0
    rho_sq, rho_err_sq = 0, 0
    for ind in range(mesh.Nx * mesh.Ny):
        rhoSum = 0
        uSum = 0
        vSum = 0
        for k in range(lattice.noOfDirections):
            rhoSum += fields.f_new[ind, k]
            uSum += lattice.c[k, 0] * fields.f_new[ind, k]
            vSum += lattice.c[k, 1] * fields.f_new[ind, k]
        fields.rho[ind] = rhoSum
        fields.u[ind, 0] = uSum/(fields.rho[ind] + 1e-9)
        fields.u[ind, 1] = vSum/(fields.rho[ind] + 1e-9)

        # Residue calculation
        u_err_sq += (fields.u[ind, 0] - u_old[ind, 0]) * \
            (fields.u[ind, 0] - u_old[ind, 0])
        u_sq += u_old[ind, 0] * u_old[ind, 0]
        v_err_sq += (fields.u[ind, 1] - u_old[ind, 1]) * \
            (fields.u[ind, 1] - u_old[ind, 1])
        v_sq += u_old[ind, 1] * u_old[ind, 1]
        rho_err_sq += (fields.rho[ind] - rho_old[ind]) * \
            (fields.rho[ind] - rho_old[ind])
        rho_sq += rho_old[ind] * rho_old[ind]
        u_old[ind, 0] = fields.u[ind, 0]
        u_old[ind, 1] = fields.u[ind, 1]
        rho_old[ind] = fields.rho[ind]
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
        resU, resV, resRho = computeFields(simulation.fields,
                                           simulation.mesh, simulation.lattice,
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

        equilibriumRelaxation(simulation.mesh, simulation.fields,
                              simulation.collisionFunc,
                              simulation.collisionScheme.preFactor,
                              simulation.equilibriumFunc,
                              simulation.equilibriumArgs
                              )
        stream(simulation.fields, simulation.lattice,
               simulation.mesh)
        simulation.setBoundaryFunc(simulation.fields, simulation.lattice,
                                   simulation.mesh)
