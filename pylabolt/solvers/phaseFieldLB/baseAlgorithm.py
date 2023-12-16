import numpy as np
from numba import prange, cuda
import numba
import copy

from pylabolt.utils.inputOutput import (writeFields, saveState,
                                        copyFields_cuda, writeFields_mpi)
from pylabolt.solvers.phaseFieldLB.kernels import (equilibriumRelaxation_cuda,
                                                   computeFields_cuda,
                                                   stream_cuda,
                                                   computeResiduals_cuda)
from pylabolt.parallel.MPI_comm import (reduceComm, proc_boundary,
                                        gatherForcesTorque_mpi,
                                        proc_boundaryGradTerms)
from pylabolt.solvers.phaseFieldLB.phaseField import forcePhaseField


def equilibriumRelaxationFluid(Nx, Ny, f_eq, f, f_new, u, p, rho, solid,
                               forceField, source, collisionFuncFluid,
                               equilibriumFuncFluid, preFactorFluid,
                               equilibriumArgsFluid, boundaryNode,
                               procBoundary, forceFunc_force,
                               forceArgs_force, forcingPreFactorFluid, cs_2):
    for ind in prange(Nx * Ny):
        if (solid[ind, 0] != 1 and procBoundary[ind] != 1
                and boundaryNode[ind] != 1):
            force = forceField[ind, :] * cs_2 / rho[ind]
            forceFunc_force(u[ind, :], source[ind], force, *forceArgs_force)
            equilibriumFuncFluid(f_eq[ind, :], u[ind, :],
                                 p[ind], *equilibriumArgsFluid)
            collisionFuncFluid(f[ind, :], f_new[ind, :], f_eq[ind, :],
                               preFactorFluid[ind], forcingPreFactorFluid[ind],
                               source[ind], force=True)


def equilibriumRelaxationPhase(Nx, Ny, h_eq, h, h_new, u, phi, gradPhi,
                               solid, boundaryNode, interfaceWidth, c, w,
                               collisionFuncPhase, equilibriumFuncPhase,
                               preFactorPhase, equilibriumArgsPhase,
                               procBoundary, forcingPreFactorPhase,
                               noOfDirections, precision):
    for ind in prange(Nx * Ny):
        if (solid[ind, 0] != 1 and procBoundary[ind] != 1
                and boundaryNode[ind] != 1):
            force = np.zeros(noOfDirections, dtype=precision)
            forcePhaseField(force, phi[ind], gradPhi[ind], interfaceWidth,
                            c, w, noOfDirections)
            # if ind == int(34 * 93 + 49):
            #     print(force)
            #     print(gradPhi[ind])
            equilibriumFuncPhase(h_eq[ind, :], u[ind, :], phi[ind],
                                 *equilibriumArgsPhase)

            collisionFuncPhase(h[ind, :], h_new[ind, :], h_eq[ind, :],
                               preFactorPhase, forcingPreFactorPhase,
                               force, force=True)


def computeFieldsFluid(Nx, Ny, f_new, u, p, rho, phi, forceField, solid, c,
                       cs, noOfDirections, boundaryNode, procBoundary, size,
                       forceFunc_vel, forceCoeffVel, rho_l, rho_g, phi_g,
                       precision, u_old, rho_old, residues):
    u_sq, u_err_sq = 0., 0.
    v_sq, v_err_sq = 0., 0.
    rho_sq, rho_err_sq = 0., 0.
    for ind in prange(Nx * Ny):
        if (solid[ind, 0] != 1 and procBoundary[ind] != 1
                and boundaryNode[ind] != 1):
            rho[ind] = rho_g + (phi[ind] - phi_g) * (rho_l - rho_g)
            pSum = precision(0)
            uSum = precision(0)
            vSum = precision(0)
            for k in range(noOfDirections):
                pSum += f_new[ind, k]
                uSum += c[k, 0] * f_new[ind, k]
                vSum += c[k, 1] * f_new[ind, k]
            p[ind] = pSum
            u[ind, 0] = uSum
            u[ind, 1] = vSum
            forceFunc_vel(u[ind, :], rho[ind], forceField[ind, :],
                          forceCoeffVel)
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
    residues[0], residues[1] = u_sq, u_err_sq
    residues[2], residues[3] = v_sq, v_err_sq
    residues[4], residues[5] = rho_sq, rho_err_sq


def computeFieldsPhase(Nx, Ny, h_new, phi, solid, procBoundary,
                       boundaryNode, noOfDirections,
                       precision, phi_old, residues):
    phi_sq, phi_err_sq = 0., 0.
    for ind in prange(Nx * Ny):
        if (solid[ind, 0] != 1 and procBoundary[ind] != 1
                and boundaryNode[ind] != 1):
            phiSum = precision(0)
            for k in range(noOfDirections):
                phiSum += h_new[ind, k]
            phi[ind] = phiSum
        # Residue calculation
        phi_err_sq += (phi[ind] - phi_old[ind]) * \
            (phi[ind] - phi_old[ind])
        phi_sq += phi_old[ind] * phi_old[ind]
        phi_old[ind] = phi[ind]
    residues[6], residues[7] = phi_sq, phi_err_sq


def streamFluid(Nx, Ny, f, f_new, solid, rho, u, procBoundary, boundaryNode,
                c, w, noOfDirections, cs_2, invList, nx, ny, nProc_x, nProc_y,
                size):
    for ind in prange(Nx * Ny):
        if (procBoundary[ind] != 1 and boundaryNode[ind] != 1 and
                solid[ind, 0] != 1):
            i, j = int(ind / Ny), int(ind % Ny)
            for k in range(noOfDirections):
                i_old = i - int(c[k, 0])
                j_old = j - int(c[k, 1])
                i_solid = i - int(c[k, 0])
                j_solid = j - int(c[k, 1])
                if size == 1:
                    if i - int(2 * c[k, 0]) < 0 or i - int(2 * c[k, 0]) >= Nx:
                        i_old = (i_old - int(2 * c[k, 0]) + Nx) % Nx
                        i_solid = (i_solid - int(2 * c[k, 0]) + Nx) % Nx
                    else:
                        i_old = (i_old + Nx) % Nx
                        i_solid = (i_solid + Nx) % Nx
                    if j - int(2 * c[k, 1]) < 0 or j - int(2 * c[k, 1]) >= Ny:
                        j_old = (j_old - int(2 * c[k, 1]) + Ny) % Ny
                        j_solid = (j_solid - int(2 * c[k, 1]) + Ny) % Ny
                    else:
                        j_old = (j_old + Ny) % Ny
                        j_solid = (j_solid + Ny) % Ny
                elif size > 1 and boundaryNode[ind] == 2:
                    if (i - int(2 * c[k, 0]) == 0 and nx == 0 or
                            i - int(2 * c[k, 0]) == Nx - 1 and
                            nx == nProc_x - 1):
                        i_solid = i_solid - int(c[k, 0])
                    if (j - int(2 * c[k, 1]) == 0 and ny == 0 or
                            j - int(2 * c[k, 1]) == Ny - 1 and
                            ny == nProc_y - 1):
                        j_solid = j_solid - int(c[k, 1])
                if solid[i_solid * Ny + j_solid, 0] != 1:
                    f_new[ind, k] = f[i_old * Ny + j_old, k]
                elif solid[i_solid * Ny + j_solid, 0] == 1:
                    ind_solid = i_solid * Ny + j_solid
                    preFactor = 2 * w[invList[k]] * \
                        (c[invList[k], 0] * u[ind_solid, 0] +
                         c[invList[k], 1] * u[ind_solid, 1]) * cs_2
                    f_new[ind, k] = f[ind, invList[k]] - preFactor


def streamPhase(Nx, Ny, h, h_new, solid, procBoundary, boundaryNode, c,
                noOfDirections, invList, nx, ny, nProc_x, nProc_y, size):
    for ind in prange(Nx * Ny):
        if (solid[ind, 0] != 1 and procBoundary[ind] != 1
                and boundaryNode[ind] != 1):
            i, j = int(ind / Ny), int(ind % Ny)
            for k in range(noOfDirections):
                i_old = i - int(c[k, 0])
                j_old = j - int(c[k, 1])
                i_solid = i - int(c[k, 0])
                j_solid = j - int(c[k, 1])
                if size == 1:
                    if i - int(2 * c[k, 0]) < 0 or i - int(2 * c[k, 0]) >= Nx:
                        i_old = (i_old - int(2 * c[k, 0]) + Nx) % Nx
                        i_solid = (i_solid - int(2 * c[k, 0]) + Nx) % Nx
                    else:
                        i_old = (i_old + Nx) % Nx
                        i_solid = (i_solid + Nx) % Nx
                    if j - int(2 * c[k, 1]) < 0 or j - int(2 * c[k, 1]) >= Ny:
                        j_old = (j_old - int(2 * c[k, 1]) + Ny) % Ny
                        j_solid = (j_solid - int(2 * c[k, 1]) + Ny) % Ny
                    else:
                        j_old = (j_old + Ny) % Ny
                        j_solid = (j_solid + Ny) % Ny
                elif size > 1 and boundaryNode[ind] == 2:
                    if (i - int(2 * c[k, 0]) == 0 and nx == 0 or
                            i - int(2 * c[k, 0]) == Nx - 1 and
                            nx == nProc_x - 1):
                        i_solid = i_solid - int(c[k, 0])
                    if (j - int(2 * c[k, 1]) == 0 and ny == 0 or
                            j - int(2 * c[k, 1]) == Ny - 1 and
                            ny == nProc_y - 1):
                        j_solid = j_solid - int(c[k, 1])
                if solid[i_solid * Ny + j_solid, 0] != 1:
                    h_new[ind, k] = h[i_old * Ny + j_old, k]
                elif solid[i_solid * Ny + j_solid, 0] == 1:
                    h_new[ind, k] = h[ind, invList[k]]


def computeResiduals(residues, tempResidues, comm, rank, size, precision):
    finalSum = reduceComm(residues, tempResidues, comm, rank, size, precision)
    sum_u = finalSum[0]
    sum_v = finalSum[2]
    sum_rho = finalSum[4]
    sum_phi = finalSum[6]
    sum_u_sq = finalSum[1]
    sum_v_sq = finalSum[3]
    sum_rho_sq = finalSum[5]
    sum_phi_sq = finalSum[7]
    if rank == 0:
        if np.isclose(sum_u, 0, rtol=1e-10):
            sum_u += 1e-10
        if np.isclose(sum_v, 0, rtol=1e-10):
            sum_v += 1e-10
        if np.isclose(sum_rho, 0, rtol=1e-10):
            sum_rho += 1e-10
        if np.isclose(sum_phi, 0, rtol=1e-10):
            sum_phi += 1e-10
        resU = np.sqrt(sum_u_sq/(sum_u))
        resV = np.sqrt(sum_v_sq/(sum_v))
        resRho = np.sqrt(sum_rho_sq/(sum_rho))
        resPhi = np.sqrt(sum_phi_sq/(sum_phi))
        return resU, resV, resRho, resPhi
    else:
        return 0, 0, 0, 0


class baseAlgorithm:
    def __init__(self):
        self.equilibriumRelaxationFluid = equilibriumRelaxationFluid
        self.streamFluid = streamFluid
        self.computeFieldsFluid = computeFieldsFluid
        self.equilibriumRelaxationPhase = equilibriumRelaxationPhase
        self.streamPhase = streamPhase
        self.computeFieldsPhase = computeFieldsPhase

    def setupBase_cpu(self, parallel):
        self.equilibriumRelaxationFluid = \
            numba.njit(self.equilibriumRelaxationFluid, parallel=parallel,
                       cache=False, nogil=True)
        self.streamFluid = numba.njit(self.streamFluid, parallel=parallel,
                                      cache=False, nogil=True)
        self.computeFieldsFluid = numba.njit(self.computeFieldsFluid,
                                             parallel=parallel, cache=False,
                                             nogil=True)
        self.equilibriumRelaxationPhase = \
            numba.njit(self.equilibriumRelaxationPhase, parallel=parallel,
                       cache=False, nogil=True)
        self.streamPhase = numba.njit(self.streamPhase, parallel=parallel,
                                      cache=False, nogil=True)
        self.computeFieldsPhase = numba.njit(self.computeFieldsPhase,
                                             parallel=parallel, cache=False,
                                             nogil=True)

    def jitWarmUp(self, simulation, size, rank, comm):
        if rank == 0:
            print('\nJIT warmup...', flush=True)
        tempSim = copy.deepcopy(simulation)
        tempSim.startTime = 1
        tempSim.endTime = 2
        tempSim.stdOutputInterval = 100
        tempSim.saveInterval = 100
        tempSim.saveStateInterval = None
        tempSim.phaseField.diffusionTime = 2
        self.solver(tempSim, size, rank, comm, warmup=True)
        if rank == 0:
            print('JIT warmup done!!\n\n', flush=True)
        del tempSim

    def solver(self, simulation, size, rank, comm, warmup=False):
        resU, resV = 1e6, 1e6
        resRho = 1e6
        u_old = np.zeros((simulation.mesh.Nx * simulation.mesh.Ny, 2),
                         dtype=simulation.precision)
        rho_old = np.zeros((simulation.mesh.Nx * simulation.mesh.Ny),
                           dtype=simulation.precision)
        phi_old = np.zeros((simulation.mesh.Nx * simulation.mesh.Ny),
                           dtype=simulation.precision)
        residues = np.zeros(8, dtype=simulation.precision)
        tempResidues = np.zeros(8, dtype=simulation.precision)

        # Smooth Initialization for interface
        simulation.collisionArgsPhase[5] = simulation.fields.u_initialize
        if rank == 0 and warmup is False:
            print('Beginning smooth initialization of phase field...')
            print('Calculated time steps required to initialize = ',
                  simulation.phaseField.diffusionTime, '\n')
        for timeStep in range(0, 0):
            self.computeFieldsPhase(*simulation.computeFieldsArgsPhase,
                                    phi_old, residues)
            if size > 1:
                comm.Barrier()
                proc_boundaryGradTerms(*simulation.commPhiArgs, comm,
                                       inner=True)
                comm.Barrier()
            simulation.phaseField.computeGradLapPhi(*simulation.
                                                    computeGradLapPhiArgs,
                                                    initial=True)
            if timeStep % simulation.stdOutputInterval == 0:
                resU, resV, resRho, resPhi = \
                    computeResiduals(residues, tempResidues, comm, rank,
                                     size, simulation.precision)
                if rank == 0 and warmup is False:
                    print('timeStep = ' + str(round(timeStep, 10)).ljust(16) +
                          ' | resPhi = ' + str(round(resPhi, 10)).ljust(16),
                          flush=True)
            self.equilibriumRelaxationPhase(*simulation.collisionArgsPhase)
            if size > 1:
                comm.Barrier()
                proc_boundary(*simulation.procBoundaryArgsPhase, comm,
                              inner=True)
                comm.Barrier()
            self.streamPhase(*simulation.streamArgsPhase)
            simulation.boundary.setBoundary(simulation.fields, simulation
                                            .lattice, simulation.mesh,
                                            equilibriumFunc=simulation.
                                            collisionScheme.
                                            equilibriumFuncPhase,
                                            equilibriumArgs=simulation.
                                            collisionScheme.
                                            equilibriumArgsPhase,
                                            initialPhase=True)

        simulation.collisionArgsPhase[5] = simulation.fields.u
        simulation.initializePopulations(u_initial='u')

        if rank == 0 and warmup is False:
            print('\nSmooth initialization of phase field done\n\n')
            print('Starting simulation...\n')
        if rank == 0:
            massFile = open('mass.dat', 'w')
        for timeStep in range(simulation.startTime, simulation.endTime + 1,
                              simulation.lattice.deltaT):
            # Compute macroscopic fields and forces #
            self.computeFieldsPhase(*simulation.computeFieldsArgsPhase,
                                    phi_old, residues)
            if size > 1:
                comm.Barrier()
                proc_boundaryGradTerms(*simulation.commPhiArgs, comm,
                                       inner=True)
                comm.Barrier()
            if simulation.phaseField.contactAngle is not None:
                simulation.phaseField.setSolidPhaseField(simulation.options,
                                                         simulation.fields,
                                                         simulation.lattice,
                                                         simulation.mesh,
                                                         size)
                if size > 1:
                    comm.Barrier()
                    proc_boundaryGradTerms(*simulation.commPhiArgs, comm,
                                           inner=True)
                    comm.Barrier()

            simulation.phaseField.computeGradLapPhi(*simulation.
                                                    computeGradLapPhiArgs,
                                                    initial=False)

            if size > 1:
                comm.Barrier()
                proc_boundaryGradTerms(*simulation.commNormalPhiArgs, comm,
                                       inner=True)
                comm.Barrier()
            # if simulation.phaseField.contactAngle is not None:
            #     simulation.phaseField.correctNormalPhi(simulation.options,
            #                                            simulation.fields,
            #                                            simulation.precision,
            #                                            simulation.mesh,
            #                                            simulation.lattice,
            #                                            simulation.size,
            #                                            timeStep,
            #                                            simulation.saveInterval)
            simulation.phaseField.forceFluid(*simulation.forceFluidArgs)
            self.computeFieldsFluid(*simulation.computeFieldsArgsFluid,
                                    u_old, rho_old, residues)
            # Write data #
            if timeStep % simulation.stdOutputInterval == 0:
                resU, resV, resRho, resPhi = \
                    computeResiduals(residues, tempResidues, comm, rank,
                                     size, simulation.precision)
                if simulation.phaseField.displayMass is True:
                    mass = simulation.phaseField.\
                        computeMass(simulation.phaseField.mass,
                                    simulation.fields.phi, simulation.fields.
                                    solid, simulation.fields.boundaryNode,
                                    simulation.fields.procBoundary,
                                    simulation.mesh.Nx, simulation.mesh.Ny)
                    # print(rank, mass)
                    if size > 1:
                        mass = reduceComm(simulation.phaseField.mass,
                                          simulation.phaseField.mass,
                                          comm, rank, size, simulation.
                                          precision)
                if simulation.obstacle.displaySolidMass is True:
                    solidMass = simulation.obstacle.\
                        computeSolidMass(simulation.obstacle.solidMass,
                                         simulation.fields.solid, simulation.
                                         fields.boundaryNode, simulation.
                                         fields.procBoundary, simulation.mesh.
                                         Nx, simulation.mesh.Ny)
                    # print(rank, mass)
                    if size > 1:
                        solidMass = reduceComm(simulation.obstacle.solidMass,
                                               simulation.obstacle.solidMass,
                                               comm, rank, size, simulation.
                                               precision)
                if (rank == 0 and warmup is False and
                        simulation.phaseField.displayMass is True and
                        simulation.obstacle.displaySolidMass is True):
                    print('timeStep = ' + str(round(timeStep, 10)).ljust(12) +
                          ' | resU = ' + str(round(resU, 10)).ljust(12) +
                          ' | resV = ' + str(round(resV, 10)).ljust(12) +
                          ' | resRho = ' + str(round(resRho, 10)).ljust(12) +
                          ' | resPhi = ' + str(round(resPhi, 10)).ljust(12) +
                          ' | phiMass = ' + str(round(mass[0], 8)).ljust(10) +
                          ' | solidMass = ' + str(round(solidMass[0], 8)).
                          ljust(10), flush=True)
                    massFile.write(str(timeStep) + '\t' + str(solidMass[0]) +
                                   '\t' + str(mass[0]) + '\n')
                elif (rank == 0 and warmup is False and
                        simulation.phaseField.displayMass is False and
                        simulation.obstacle.displaySolidMass is True):
                    print('timeStep = ' + str(round(timeStep, 10)).ljust(12) +
                          ' | resU = ' + str(round(resU, 10)).ljust(12) +
                          ' | resV = ' + str(round(resV, 10)).ljust(12) +
                          ' | resRho = ' + str(round(resRho, 10)).ljust(12) +
                          ' | resPhi = ' + str(round(resPhi, 10)).ljust(12) +
                          ' | solidMass = ' + str(round(solidMass[0], 8)).
                          ljust(10), flush=True)
                elif (rank == 0 and warmup is False and
                        simulation.phaseField.displayMass is True and
                        simulation.obstacle.displaySolidMass is False):
                    print('timeStep = ' + str(round(timeStep, 10)).ljust(12) +
                          ' | resU = ' + str(round(resU, 10)).ljust(12) +
                          ' | resV = ' + str(round(resV, 10)).ljust(12) +
                          ' | resRho = ' + str(round(resRho, 10)).ljust(12) +
                          ' | resPhi = ' + str(round(resPhi, 10)).ljust(12) +
                          ' | phiMass = ' + str(round(mass[0], 8)).ljust(10),
                          flush=True)
                elif (rank == 0 and warmup is False and
                        simulation.phaseField.displayMass is False and
                        simulation.obstacle.displaySolidMass is False):
                    print('timeStep = ' + str(round(timeStep, 10)).ljust(12) +
                          ' | resU = ' + str(round(resU, 10)).ljust(12) +
                          ' | resV = ' + str(round(resV, 10)).ljust(12) +
                          ' | resRho = ' + str(round(resRho, 10)).ljust(12) +
                          ' | resPhi = ' + str(round(resPhi, 10)).ljust(12),
                          flush=True)
            if timeStep % simulation.saveInterval == 0:
                if size == 1:
                    writeFields(timeStep, simulation.fields,
                                simulation.lattice, simulation.mesh)
                    # np.savez('curvatureChemPot.npz', curvature=simulation.fields.
                    #          curvature)
                    # np.savez('normalPhiChemPot.npz', normalPhi=simulation.fields.
                    #          normalPhi)
                    # np.savez('gradPhiChemPot.npz', gradPhi=simulation.fields.
                    #          gradPhi)
                else:
                    writeFields_mpi(timeStep, simulation.fields,
                                    simulation.lattice, simulation.mesh,
                                    rank, comm)
                if (simulation.options.computeForces is True
                        or simulation.options.computeTorque is True):
                    args = (simulation.fields, simulation.transport,
                            simulation.lattice, simulation.mesh,
                            simulation.precision, size)
                    if size > 1:
                        simulation.options. \
                            forceTorqueCalc(*args, timeStep,
                                            mpiParams=simulation.mpiParams,
                                            ref_index=None,
                                            phaseField=simulation.phaseField)
                        names, forces, torque = \
                            gatherForcesTorque_mpi(simulation.options,
                                                   comm, rank, size,
                                                   simulation.precision)
                    else:
                        simulation.options. \
                            forceTorqueCalc(*args, timeStep, mpiParams=None,
                                            ref_index=None,
                                            phaseField=simulation.phaseField)
                        names = simulation.options.surfaceNamesGlobal
                        forces = np.array(simulation.options.forces,
                                          dtype=simulation.precision)
                        torque = np.array(simulation.options.torque,
                                          dtype=simulation.precision)
                    if rank == 0:
                        simulation.options.writeForces(timeStep, names,
                                                       forces, torque)
            if simulation.saveStateInterval is not None:
                if timeStep % simulation.saveStateInterval == 0:
                    saveState(timeStep, simulation)
            # Write data #

            # Check for convergence #
            if rank == 0 and warmup is False:
                if (resU < simulation.relTolU and
                        resV < simulation.relTolV and
                        resRho < simulation.relTolRho and
                        resPhi < simulation.relTolPhi):
                    print('Convergence Criteria matched!!', flush=True)
                    print('timeStep = ' + str(round(timeStep, 10)).ljust(12) +
                          ' | resU = ' + str(round(resU, 10)).ljust(12) +
                          ' | resV = ' + str(round(resV, 10)).ljust(12) +
                          ' | resRho = ' + str(round(resRho, 10)).ljust(12) +
                          ' | resPhi = ' + str(round(resV, 10)).ljust(12),
                          flush=True)
                    if size == 1:
                        writeFields(timeStep, simulation.fields,
                                    simulation.lattice, simulation.mesh)
                        if (simulation.options.computeForces is True or
                                simulation.options.computeTorque is True):
                            args = (simulation.fields, simulation.transport,
                                    simulation.lattice, simulation.mesh,
                                    simulation.precision, size)
                            simulation.options. \
                                forceTorqueCalc(*args, timeStep,
                                                mpiParams=None,
                                                ref_index=None,
                                                phaseField=simulation.
                                                phaseField)
                            names = simulation.options.surfaceNamesGlobal
                            forces = np.array(simulation.options.forces,
                                              dtype=simulation.precision)
                            torque = np.array(simulation.options.torque,
                                              dtype=simulation.precision)
                            if rank == 0:
                                simulation.options.writeForces(timeStep, names,
                                                               forces, torque)
                        break
                    else:
                        for proc in range(size):
                            comm.send(1, dest=proc, tag=1*proc)
                        writeFields_mpi(timeStep, simulation.fields,
                                        simulation.lattice,
                                        simulation.mesh, rank, comm)
                        if (simulation.options.computeForces is True or
                                simulation.options.computeTorque is True):
                            simulation.options. \
                                forceTorqueCalc(*args, mpiParams=simulation.
                                                mpiParams, ref_index=None,
                                                phaseField=simulation.
                                                phaseField)
                            names, forces, torque = \
                                gatherForcesTorque_mpi(simulation.options,
                                                       comm, rank, size,
                                                       simulation.precision)
                            if rank == 0:
                                simulation.options.writeForces(timeStep, names,
                                                               forces, torque)
                        break
                else:
                    if size > 1:
                        for proc in range(1, size):
                            comm.send(0, dest=proc, tag=1*proc)
            elif size > 1 and rank != 0 and warmup is False:
                flag = comm.recv(source=0, tag=1*rank)
                if flag == 1:
                    writeFields_mpi(timeStep, simulation.fields,
                                    simulation.lattice,
                                    simulation.mesh, rank, comm)
                    if (simulation.options.computeForces is True or
                            simulation.options.computeTorque is True):
                        args = (simulation.fields, simulation.transport,
                                simulation.lattice, simulation.mesh,
                                simulation.precision, size)
                        simulation.options. \
                            forceTorqueCalc(*args, mpiParams=simulation.
                                            mpiParams, ref_index=None,
                                            phaseField=simulation.phaseField)
                        names, forces, torque = \
                            gatherForcesTorque_mpi(simulation.options,
                                                   comm, rank, size,
                                                   simulation.precision)
                    break
            comm.Barrier()
            # Check for convergence done!#

            # Obstacle modification #
            if simulation.obstacle.obsModifiable is True:
                args = (simulation.fields, simulation.transport,
                        simulation.lattice, simulation.mesh,
                        simulation.precision, size, timeStep)
                if size == 1:
                    simulation.options. \
                        forceTorqueCalc(*args, mpiParams=None,
                                        ref_index=simulation.obstacle.
                                        obsOrigin, phaseField=simulation.
                                        phaseField, warmup=warmup)
                    # if timeStep == 2:
                    #     np.savez('u_p.npz', u=simulation.fields.u)
                    #     np.savez('rho_p.npz', rho=simulation.fields.rho)
                    #     np.savez('solid_p.npz', solid=simulation.fields.solid)
                    #     np.savez('solidNb_p.npz', solidNb=simulation.options.surfaceNodes[0])
                    simulation.obstacle.\
                        modifyObstacle(simulation.options, simulation.fields,
                                       simulation.mesh, simulation.lattice,
                                       simulation.boundary,
                                       simulation.transport,
                                       simulation.collisionScheme, size, comm,
                                       timeStep, rank, simulation.precision,
                                       mpiParams=None, phaseField=simulation.
                                       phaseField,
                                       forces=simulation.options.forces,
                                       torque=simulation.options.torque)
                    simulation.options.\
                        computeMovingSolidBoundary(simulation.mesh, simulation.
                                                   lattice, simulation.fields,
                                                   size, simulation.precision)
                elif size > 1:
                    simulation.options. \
                        forceTorqueCalc(*args, mpiParams=simulation.
                                        mpiParams, ref_index=simulation.
                                        obstacle.obsOrigin,
                                        phaseField=simulation.phaseField,
                                        warmup=warmup)
                    # if timeStep == 1498:
                    #     print(simulation.options.forces, rank)
                    comm.Barrier()
                    names, forces, torque = \
                        gatherForcesTorque_mpi(simulation.options,
                                               comm, rank, size,
                                               simulation.precision)
                    # if timeStep == 1498:
                    #     print(forces, rank)
                    #     np.savez('solid_' + str(rank) + '.npz', solid=simulation.fields.solid)
                    #     np.savez('solidNb_' + str(rank) + '.npz', solid=simulation.options.surfaceNodes[0])
                    #     np.savez('boundaryNode_' + str(rank) + '.npz', solid=simulation.fields.boundaryNode)
                    simulation.obstacle.\
                        modifyObstacle(simulation.options, simulation.fields,
                                       simulation.mesh, simulation.lattice,
                                       simulation.boundary,
                                       simulation.transport,
                                       simulation.collisionScheme, size, comm,
                                       timeStep, rank, simulation.precision,
                                       mpiParams=simulation.mpiParams,
                                       phaseField=simulation.phaseField,
                                       forces=forces, torque=torque)
                    simulation.options.\
                        computeMovingSolidBoundary(simulation.mesh, simulation.
                                                   lattice, simulation.fields,
                                                   size, simulation.precision,
                                                   mpiParams=simulation.
                                                   mpiParams)
            # Obstacle modification done #

            # Equilibrium and Collision
            self.equilibriumRelaxationPhase(*simulation.collisionArgsPhase)
            self.equilibriumRelaxationFluid(*simulation.collisionArgsFluid)

            # Share data between processors
            if size > 1:
                comm.Barrier()
                proc_boundary(*simulation.procBoundaryArgsPhase, comm,
                              inner=True)
                proc_boundary(*simulation.procBoundaryArgsFluid, comm,
                              inner=True)
                comm.Barrier()

            # Streaming
            self.streamPhase(*simulation.streamArgsPhase)
            self.streamFluid(*simulation.streamArgsFluid)

            # Boundary condition
            simulation.boundary.setBoundary(simulation.fields, simulation
                                            .lattice, simulation.mesh,
                                            equilibriumFunc=simulation.
                                            collisionScheme.
                                            equilibriumFuncPhase,
                                            equilibriumArgs=simulation.
                                            collisionScheme.
                                            equilibriumArgsPhase)
        if rank == 0:
            massFile.close()

    def solver_cuda(self, simulation, parallel):
        resU, resV = 1e6, 1e6
        resRho = 1e6
        u_old = np.zeros((simulation.mesh.Nx * simulation.mesh.Ny, 2),
                         dtype=simulation.precision)
        rho_old = np.zeros((simulation.mesh.Nx * simulation.mesh.Ny),
                           dtype=simulation.precision)
        residues = np.zeros((simulation.mesh.Nx * simulation.mesh.Ny, 6),
                            dtype=simulation.precision)
        # Copy to device
        residues_device = cuda.to_device(residues)
        noOfResidueArrays_device = \
            cuda.to_device(np.array([6], dtype=np.int64))
        residueArgs = (residues_device, noOfResidueArrays_device[0])
        # u_sq_device = cuda.to_device(u_sq)
        # u_err_sq_device = cuda.to_device(u_err_sq)
        # rho_sq_device = cuda.to_device(rho_sq)
        # rho_err_sq_device = cuda.to_device(rho_err_sq)
        u_old_device = cuda.to_device(u_old)
        rho_old_device = cuda.to_device(rho_old)
        for timeStep in range(simulation.startTime, simulation.endTime + 1,
                              simulation.lattice.deltaT):
            computeFields_cuda[parallel.blocks,
                               parallel.n_threads](*parallel.device.
                                                   computeFieldsArgs,
                                                   u_old_device,
                                                   rho_old_device,
                                                   *residueArgs)
            resU, resV, resRho = computeResiduals_cuda(*residueArgs,
                                                       parallel.blocks,
                                                       parallel.n_threads,
                                                       parallel.blockSize)
            if timeStep % simulation.stdOutputInterval == 0:
                print('timeStep = ' + str(round(timeStep, 10)).ljust(16) +
                      ' | resU = ' + str(round(resU, 10)).ljust(16) +
                      ' | resV = ' + str(round(resV, 10)).ljust(16) +
                      ' | resRho = ' + str(round(resRho, 10)).ljust(16),
                      flush=True)
            if timeStep % simulation.saveInterval == 0:
                copyFields_cuda(parallel.device, simulation.fields,
                                flag='standard')
                writeFields(timeStep, simulation.fields, simulation.mesh)
                if (simulation.options.computeForces is True or
                        simulation.options.computeTorque is True):
                    simulation.options.forceTorqueCalc_cuda(parallel.device,
                                                            simulation.
                                                            precision,
                                                            parallel.n_threads,
                                                            parallel.blocks,
                                                            parallel.blockSize)
                    names = simulation.options.surfaceNamesGlobal
                    forces = np.array(simulation.options.forces,
                                      dtype=simulation.precision)
                    torque = np.array(simulation.options.torque,
                                      dtype=simulation.precision)
                    simulation.options.writeForces(timeStep, names,
                                                   forces, torque)
            if simulation.saveStateInterval is not None:
                if timeStep % simulation.saveStateInterval == 0:
                    copyFields_cuda(parallel.device, simulation.fields,
                                    flag='all')
                    saveState(timeStep, simulation)
            if (resU < simulation.relTolU and resV < simulation.relTolV and
                    resRho < simulation.relTolRho):
                print('Convergence Criteria matched!!', flush=True)
                print('timeStep = ' + str(round(timeStep, 10)).ljust(16) +
                      ' | resU = ' + str(round(resU, 10)).ljust(16) +
                      ' | resV = ' + str(round(resV, 10)).ljust(16) +
                      ' | resRho = ' + str(round(resRho, 10)).ljust(16),
                      flush=True)
                copyFields_cuda(parallel.device, simulation.fields,
                                flag='standard')
                writeFields(timeStep, simulation.fields, simulation.mesh)
                if (simulation.options.computeForces is True or
                        simulation.options.computeTorque is True):
                    simulation.options.forceTorqueCalc_cuda(parallel.device,
                                                            simulation.
                                                            precision,
                                                            parallel.n_threads,
                                                            parallel.blocks,
                                                            parallel.blockSize)
                    names = simulation.options.surfaceNamesGlobal
                    forces = np.array(simulation.options.forces,
                                      dtype=simulation.precision)
                    torque = np.array(simulation.options.torque,
                                      dtype=simulation.precision)
                    simulation.options.writeForces(timeStep, names,
                                                   forces, torque)
                break
            if simulation.obstacle.obsModifiable is True:
                simulation.options.forceTorqueCalc_cuda(parallel.device,
                                                        simulation.
                                                        precision,
                                                        parallel.n_threads,
                                                        parallel.blocks,
                                                        parallel.blockSize,
                                                        ref_index=simulation.
                                                        obstacle.
                                                        obsOrigin_device)
                simulation.obstacle.modifyObstacle_cuda(simulation.options.
                                                        torque, parallel.
                                                        device, timeStep,
                                                        parallel.blocks,
                                                        parallel.n_threads)

            equilibriumRelaxation_cuda[parallel.blocks,
                                       parallel.n_threads](*parallel.device.
                                                           collisionArgs)
            stream_cuda[parallel.blocks,
                        parallel.n_threads](*parallel.device.streamArgs)

            simulation.boundary.setBoundary_cuda(parallel.n_threads,
                                                 parallel.blocks,
                                                 parallel.device)
