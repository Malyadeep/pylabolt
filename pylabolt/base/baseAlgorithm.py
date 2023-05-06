import numpy as np
from numba import prange, cuda
import copy

from pylabolt.utils.inputOutput import (writeFields, saveState,
                                        copyFields_cuda, writeFields_mpi)
from pylabolt.base.cuda.kernels import (equilibriumRelaxation_cuda,
                                        computeFields_cuda, stream_cuda,
                                        computeResiduals_cuda)
from pylabolt.parallel.MPI_comm import computeResiduals, proc_boundary


def equilibriumRelaxation(Nx, Ny, f_eq, f, f_new, u, rho, solid,
                          collisionFunc, equilibriumFunc, preFactor,
                          eqilibriumArgs, procBoundary):
    for ind in prange(Nx * Ny):
        if solid[ind] != 1 and procBoundary[ind] != 1:
            equilibriumFunc(f_eq[ind, :], u[ind, :],
                            rho[ind], *eqilibriumArgs)

            collisionFunc(f[ind, :], f_new[ind, :],
                          f_eq[ind, :], preFactor)


def computeFields(Nx, Ny, f_new, u, rho, solid, c,
                  noOfDirections, procBoundary, size, u_old, rho_old):
    u_sq, u_err_sq = 0., 0.
    v_sq, v_err_sq = 0., 0.
    rho_sq, rho_err_sq = 0., 0.
    for ind in prange(Nx * Ny):
        if solid[ind] != 1 and procBoundary[ind] != 1:
            rhoSum = 0.
            uSum = 0.
            vSum = 0.
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
    return u_err_sq, u_sq, v_err_sq, v_sq, rho_err_sq, rho_sq


def stream(Nx, Ny, f, f_new, c, noOfDirections, invList, solid, size):
    for ind in prange(Nx * Ny):
        i, j = int(ind / Ny), int(ind % Ny)
        for k in range(noOfDirections):
            i_old = i - int(c[k, 0])
            j_old = j - int(c[k, 1])
            if size == 1:
                i_old = (i_old + Nx) % Nx
                j_old = (j_old + Ny) % Ny
            if solid[i_old * Ny + j_old] != 1:
                f_new[ind, k] = f[i_old * Ny + j_old, k]
            elif solid[i_old * Ny + j_old] == 1:
                f_new[ind, k] = f[ind, invList[k]]


class baseAlgorithm:
    def __init__(self):
        self.equilibriumRelaxation = equilibriumRelaxation
        self.stream = stream
        self.computeFields = computeFields

    def jitWarmUp(self, simulation, size, rank, comm):
        if rank == 0:
            print('\nJIT warmup...')
        tempSim = copy.deepcopy(simulation)
        tempSim.startTime = 1
        tempSim.endTime = 2
        tempSim.stdOutputInterval = 100
        tempSim.saveInterval = 100
        tempSim.saveStateInterval = None
        self.solver(tempSim, size, rank, comm)
        if rank == 0:
            print('JIT warmup done!!\n\n')
        del tempSim

    def solver(self, simulation, size, rank, comm):
        resU, resV = 1e6, 1e6
        resRho = 1e6
        u_old = np.zeros((simulation.mesh.Nx * simulation.mesh.Ny, 2),
                         dtype=simulation.precision)
        rho_old = np.zeros((simulation.mesh.Nx * simulation.mesh.Ny),
                           dtype=simulation.precision)

        for timeStep in range(simulation.startTime, simulation.endTime + 1,
                              simulation.lattice.deltaT):
            u_err_sq, u_sq, v_err_sq, v_sq, rho_err_sq, rho_sq = \
                self.computeFields(*simulation.
                                   computeFieldsArgs,
                                   u_old, rho_old)
            resU, resV, resRho = computeResiduals(u_err_sq, u_sq, v_err_sq,
                                                  v_sq, rho_err_sq, rho_sq,
                                                  comm, rank, size)
            if timeStep % simulation.stdOutputInterval == 0 and rank == 0:
                print('timeStep = ' + str(round(timeStep, 10)).ljust(16) +
                      ' | resU = ' + str(round(resU, 10)).ljust(16) +
                      ' | resV = ' + str(round(resV, 10)).ljust(16) +
                      ' | resRho = ' + str(round(resRho, 10)).ljust(16),
                      flush=True)
            if timeStep % simulation.saveInterval == 0:
                if size == 1:
                    writeFields(timeStep, simulation.fields, simulation.mesh)
                else:
                    writeFields_mpi(timeStep, simulation.fields,
                                    simulation.mesh, rank, comm)
            if simulation.saveStateInterval is not None:
                if timeStep % simulation.saveStateInterval == 0:
                    saveState(timeStep, simulation)
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
                                    simulation.mesh)
                        break
                    else:
                        for proc in range(size):
                            comm.send(1, dest=proc, tag=1*proc)
                        writeFields_mpi(timeStep, simulation.fields,
                                        simulation.mesh, rank, comm)
                        break
                else:
                    if size > 1:
                        for proc in range(size):
                            comm.send(0, dest=proc, tag=1*proc)
            elif size > 1 and rank != 0:
                flag = comm.recv(source=0, tag=1*rank)
                if flag == 1:
                    writeFields_mpi(timeStep, simulation.fields,
                                    simulation.mesh, rank, comm)
                    break

            self.equilibriumRelaxation(*simulation.collisionArgs)

            if size > 1:
                comm.Barrier()
                proc_boundary(*simulation.proc_boundaryArgs, comm)
                comm.Barrier()

            self.stream(*simulation.streamArgs)

            simulation.setBoundaryFunc(simulation.fields, simulation.lattice,
                                       simulation.mesh)

    def solver_cuda(self, simulation, parallel):
        resU, resV = 1e6, 1e6
        resRho = 1e6
        u_old = np.zeros((simulation.mesh.Nx * simulation.mesh.Ny, 2),
                         dtype=np.float64)
        rho_old = np.zeros((simulation.mesh.Nx * simulation.mesh.Ny),
                           dtype=np.float64)
        u_sq = np.zeros_like(u_old)
        u_err_sq = np.zeros_like(u_old)
        rho_sq = np.zeros_like(rho_old)
        rho_err_sq = np.zeros_like(rho_old)

        # Copy to device
        u_sq_device = cuda.to_device(u_sq)
        u_err_sq_device = cuda.to_device(u_err_sq)
        rho_sq_device = cuda.to_device(rho_sq)
        rho_err_sq_device = cuda.to_device(rho_err_sq)
        u_old_device = cuda.to_device(u_old)
        rho_old_device = cuda.to_device(rho_old)
        residueArgs = (
            u_sq_device, u_err_sq_device, rho_sq_device, rho_err_sq_device,
        )
        for timeStep in range(simulation.startTime, simulation.endTime + 1,
                              simulation.lattice.deltaT):
            computeFields_cuda[parallel.blocks,
                               parallel.n_threads](*parallel.device.
                                                   computeFieldsArgs,
                                                   u_old_device,
                                                   rho_old_device,
                                                   *residueArgs)
            resU, resV, resRho = computeResiduals_cuda(*residueArgs)
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
                break
            equilibriumRelaxation_cuda[parallel.blocks,
                                       parallel.n_threads](*parallel.device.
                                                           collisionArgs)

            stream_cuda[parallel.blocks,
                        parallel.n_threads](*parallel.device.streamArgs)

            simulation.boundary.setBoundary_cuda(parallel.n_threads,
                                                 parallel.blocks,
                                                 parallel.device)
