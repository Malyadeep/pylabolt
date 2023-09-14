import numpy as np
from numba import prange, cuda
import numba
import copy

from pylabolt.utils.inputOutput import (writeFields, saveState,
                                        copyFields_cuda, writeFields_mpi)
from pylabolt.solvers.fluidLB.kernels import (equilibriumRelaxation_cuda,
                                              computeFields_cuda, stream_cuda,
                                              computeResiduals_cuda)
from pylabolt.parallel.MPI_comm import (residueComm, proc_boundary,
                                        gatherForcesTorque_mpi)


def equilibriumRelaxation(Nx, Ny, f_eq, f, f_new, u, rho, solid, boundaryNode,
                          source, gravity, collisionFunc, equilibriumFunc,
                          preFactor, equilibriumArgs, procBoundary,
                          forceFunc_force, forceArgs_force, forcingPreFactor,
                          noOfDirections, precision):
    for ind in prange(Nx * Ny):
        if (solid[ind, 0] != 1 and procBoundary[ind] != 1
                and boundaryNode[ind] != 1):
            force = False
            if forceFunc_force is not None:
                force = True
                gravityForce = rho[ind] * gravity
                forceFunc_force(u[ind], source[ind], gravityForce,
                                *forceArgs_force)
            equilibriumFunc(f_eq[ind, :], u[ind, :],
                            rho[ind], *equilibriumArgs)

            collisionFunc(f[ind], f_new[ind], f_eq[ind], preFactor,
                          forcingPreFactor, source[ind], force=force)


def computeFields(Nx, Ny, f_new, u, rho, solid, boundaryNode,
                  gravity, c, noOfDirections, procBoundary, size,
                  forceFunc_vel, forceCoeffVel, f_eq,
                  precision, u_old, rho_old, residues):
    u_sq, u_err_sq = 0., 0.
    v_sq, v_err_sq = 0., 0.
    rho_sq, rho_err_sq = 0., 0.
    for ind in prange(Nx * Ny):
        if (solid[ind, 0] != 1 and procBoundary[ind] != 1
                and boundaryNode[ind] != 1):
            rhoSum = precision(0)
            uSum = precision(0)
            vSum = precision(0)
            for k in range(noOfDirections):
                rhoSum += f_new[ind, k]
                uSum += c[k, 0] * f_new[ind, k]
                vSum += c[k, 1] * f_new[ind, k]
            rho[ind] = rhoSum
            u[ind, 0] = uSum/rho[ind]
            u[ind, 1] = vSum/rho[ind]
            if forceFunc_vel is not None:
                gravityForce = rho[ind] * gravity
                forceFunc_vel(u[ind], rho[ind], gravityForce, forceCoeffVel)
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


def stream(Nx, Ny, f, f_new, solid, rho, u, boundaryNode, procBoundary, c, w,
           noOfDirections, cs_2, invList, size):
    for ind in prange(Nx * Ny):
        if procBoundary[ind] != 1 and boundaryNode[ind] != 1:
            i, j = int(ind / Ny), int(ind % Ny)
            for k in range(noOfDirections):
                i_old = i - int(c[k, 0])
                j_old = j - int(c[k, 1])
                if size == 1:
                    if i - int(2 * c[k, 0]) < 0 or i - int(2 * c[k, 0]) >= Nx:
                        i_old = (i_old - int(2 * c[k, 0]) + Nx) % Nx
                    else:
                        i_old = (i_old + Nx) % Nx
                    if j - int(2 * c[k, 1]) < 0 or j - int(2 * c[k, 1]) >= Ny:
                        j_old = (j_old - int(2 * c[k, 1]) + Ny) % Ny
                    else:
                        j_old = (j_old + Ny) % Ny
                if solid[i_old * Ny + j_old, 0] != 1:
                    f_new[ind, k] = f[i_old * Ny + j_old, k]
                elif solid[i_old * Ny + j_old, 0] == 1:
                    ind_old = i_old * Ny + j_old
                    preFactor = 2 * w[invList[k]] * rho[ind] * \
                        (c[invList[k], 0] * u[ind_old, 0] +
                         c[invList[k], 1] * u[ind_old, 1]) * cs_2
                    f_new[ind, k] = f[ind, invList[k]] - preFactor


def computeResiduals(residues, tempResidues, comm, rank, size, precision):
    finalSum = residueComm(residues, tempResidues, comm, rank, size, precision)
    sum_u = finalSum[0]
    sum_v = finalSum[2]
    sum_rho = finalSum[4]
    sum_u_sq = finalSum[1]
    sum_v_sq = finalSum[3]
    sum_rho_sq = finalSum[5]
    if rank == 0:
        if np.isclose(sum_u, 0, rtol=1e-10):
            sum_u += 1e-10
        if np.isclose(sum_v, 0, rtol=1e-10):
            sum_v += 1e-10
        if np.isclose(sum_rho, 0, rtol=1e-10):
            sum_rho += 1e-10
        resU = np.sqrt(sum_u_sq/(sum_u))
        resV = np.sqrt(sum_v_sq/(sum_v))
        resRho = np.sqrt(sum_rho_sq/(sum_rho))
        return resU, resV, resRho
    else:
        return 0, 0, 0


class baseAlgorithm:
    def __init__(self):
        self.equilibriumRelaxation = equilibriumRelaxation
        self.stream = stream
        self.computeFields = computeFields

    def setupBase_cpu(self, parallel):
        self.equilibriumRelaxation = \
            numba.njit(self.equilibriumRelaxation, parallel=parallel,
                       cache=False, nogil=True)
        self.stream = numba.njit(self.stream, parallel=parallel,
                                 cache=False, nogil=True)
        self.computeFields = numba.njit(self.computeFields,
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
        self.solver(tempSim, size, rank, comm)
        if rank == 0:
            print('JIT warmup done!!\n\n', flush=True)
        del tempSim

    def solver(self, simulation, size, rank, comm):
        # np.set_printoptions(precision=5, suppress=True)
        # print(simulation.fields.boundaryNode.reshape(simulation.mesh.Nx,
        #       simulation.mesh.Ny))
        # print(simulation.fields.u[:, 0].reshape(simulation.mesh.Nx,
        #       simulation.mesh.Ny))
        resU, resV = 1e6, 1e6
        resRho = 1e6
        u_old = np.zeros((simulation.mesh.Nx * simulation.mesh.Ny, 2),
                         dtype=simulation.precision)
        rho_old = np.zeros((simulation.mesh.Nx * simulation.mesh.Ny),
                           dtype=simulation.precision)
        residues = np.zeros(6, dtype=simulation.precision)
        tempResidues = np.zeros(6, dtype=simulation.precision)

        for timeStep in range(simulation.startTime, simulation.endTime + 1,
                              simulation.lattice.deltaT):
            # Compute macroscopic fields and residues #
            self.computeFields(*simulation.computeFieldsArgs,
                               u_old, rho_old, residues)
            resU, resV, resRho = computeResiduals(residues, tempResidues,
                                                  comm, rank, size,
                                                  simulation.precision)
            # Compute macroscopic fields and residues done #

            # Write data #
            if timeStep % simulation.stdOutputInterval == 0 and rank == 0:
                print('timeStep = ' + str(round(timeStep, 10)).ljust(16) +
                      ' | resU = ' + str(round(resU, 10)).ljust(16) +
                      ' | resV = ' + str(round(resV, 10)).ljust(16) +
                      ' | resRho = ' + str(round(resRho, 10)).ljust(16),
                      flush=True)
            if timeStep % simulation.saveInterval == 0:
                if size == 1:
                    writeFields(timeStep, simulation.fields,
                                simulation.lattice, simulation.mesh)
                else:
                    writeFields_mpi(timeStep, simulation.fields,
                                    simulation.lattice,
                                    simulation.mesh, rank, comm)
                if (simulation.options.computeForces is True
                        or simulation.options.computeTorque is True):
                    args = (simulation.fields, simulation.lattice,
                            simulation.mesh, simulation.precision,
                            size)
                    if size > 1:
                        simulation.options. \
                            forceTorqueCalc(*args, mpiParams=simulation.
                                            mpiParams, ref_index=None)
                        names, forces, torque = \
                            gatherForcesTorque_mpi(simulation.options,
                                                   comm, rank, size,
                                                   simulation.precision)
                    else:
                        simulation.options. \
                            forceTorqueCalc(*args, mpiParams=None,
                                            ref_index=None)
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
            if rank == 0:
                if (resU < simulation.relTolU and
                        resV < simulation.relTolV and
                        resRho < simulation.relTolRho):
                    print('Convergence Criteria matched!!', flush=True)
                    print('timeStep = ' + str(round(timeStep, 10)).ljust(16) +
                          ' | resU = ' + str(round(resU, 10)).ljust(16) +
                          ' | resV = ' + str(round(resV, 10)).ljust(16) +
                          ' | resRho = ' + str(round(resRho, 10)).ljust(16),
                          flush=True)
                    if size == 1:
                        writeFields(timeStep, simulation.fields,
                                    simulation.lattice, simulation.mesh)
                        if (simulation.options.computeForces is True or
                                simulation.options.computeTorque is True):
                            args = (simulation.fields, simulation.lattice,
                                    simulation.mesh, simulation.precision,
                                    size)
                            simulation.options. \
                                forceTorqueCalc(*args, mpiParams=None,
                                                ref_index=None)
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
                                                mpiParams, ref_index=None)
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
            elif size > 1 and rank != 0:
                flag = comm.recv(source=0, tag=1*rank)
                if flag == 1:
                    writeFields_mpi(timeStep, simulation.fields,
                                    simulation.lattice,
                                    simulation.mesh, rank, comm)
                    if (simulation.options.computeForces is True or
                            simulation.options.computeTorque is True):
                        args = (simulation.fields, simulation.lattice,
                                simulation.mesh, simulation.precision,
                                size)
                        simulation.options. \
                            forceTorqueCalc(*args, mpiParams=simulation.
                                            mpiParams, ref_index=None)
                        names, forces, torque = \
                            gatherForcesTorque_mpi(simulation.options,
                                                   comm, rank, size,
                                                   simulation.precision)
                    break
            comm.Barrier()
            # Check for convergence done!#

            # Obstacle modification #
            if simulation.obstacle.obsModifiable is True:
                args = (simulation.fields, simulation.lattice,
                        simulation.mesh, simulation.precision,
                        size)
                if size == 1:
                    simulation.options. \
                        forceTorqueCalc(*args, mpiParams=None,
                                        ref_index=simulation.obstacle.
                                        obsOrigin)
                    torque = np.array(simulation.options.torque,
                                      dtype=simulation.precision)
                    forces = np.array(simulation.options.forces,
                                      dtype=simulation.precision)
                    simulation.obstacle.\
                        modifyObstacle(torque, simulation.fields,
                                       simulation.mesh, size, comm,
                                       timeStep, rank, simulation.precision,
                                       mpiParams=None)
                elif size > 1:
                    simulation.options. \
                        forceTorqueCalc(*args, mpiParams=simulation.
                                        mpiParams, ref_index=simulation.
                                        obstacle.obsOrigin)
                    names, forces, torque = \
                        gatherForcesTorque_mpi(simulation.options,
                                               comm, rank, size,
                                               simulation.precision)
                    simulation.obstacle.\
                        modifyObstacle(torque, simulation.fields,
                                       simulation.mesh, size, comm,
                                       timeStep, rank, simulation.precision,
                                       mpiParams=simulation.mpiParams)
            # Obstacle modification done #

            # Equilibrium and Collision
            self.equilibriumRelaxation(*simulation.collisionArgs)

            if size > 1:
                comm.Barrier()
                proc_boundary(*simulation.proc_boundaryArgs, comm, inner=True)
                comm.Barrier()

            # Streaming
            self.stream(*simulation.streamArgs)

            # Boundary condition
            simulation.setBoundaryFunc(simulation.fields, simulation.lattice,
                                       simulation.mesh)

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
