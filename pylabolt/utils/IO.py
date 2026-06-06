import os
import sys
import pickle
import h5py
import numpy as np
import numba
from numba import prange


def print_log(mssg, mpi_rank, verbose):
    if verbose and mpi_rank == 0:
        print(mssg, flush=True)


def load_simulation(comm, mpi_rank):
    try:
        try:
            working_dir = os.getcwd()
            sys.path.append(working_dir)
            import simulation
            return simulation
        except ImportError:
            raise ImportError(
                "Missing simulation.py file in current working directory"
            )
    except Exception as e:
        print_log("-" * 80, mpi_rank, verbose=True)
        print_log("FATAL ERROR!", mpi_rank, verbose=True)
        print_log(str(e), mpi_rank, verbose=True)
        comm.Abort()


def copyCoreDomainDataScalar(
    field_out, field, Ny, Ny_0, domainSize, field_aux,
    scalar_aux, field_aux_present=False
):
    if field_aux_present is False:
        for ind in prange(domainSize):
            i = ind // Ny_0
            j = ind - i * Ny_0
            ind_global = (i + 1) * Ny + j + 1
            field_out[ind] = field[ind_global]
    else:
        for ind in prange(domainSize):
            i = ind // Ny_0
            j = ind - i * Ny_0
            ind_global = (i + 1) * Ny + j + 1
            field_out[ind] = field[ind_global] *\
                field_aux[ind_global] * scalar_aux


def copyCoreDomainDataVector(field_out, field, Ny, Ny_0, domainSize):
    for ind in prange(domainSize):
        i = ind // Ny_0
        j = ind - i * Ny_0
        ind_global = (i + 1) * Ny + j + 1
        field_out[ind, 0] = field[ind_global, 0]
        field_out[ind, 1] = field[ind_global, 1]


class IOSetup:
    def __init__(self, simulation):
        self.Nx_0 = simulation.mesh.Nx - 2
        self.Ny_0 = simulation.mesh.Ny - 2
        self.domainSize = (simulation.mesh.Nx - 2) *\
            (simulation.mesh.Ny - 2)
        self.solid_out = np.zeros((self.domainSize, 2),
                                  dtype=np.int32)
        for fieldName in simulation.fields.fieldList:
            if fieldName == "u":
                self.u_out = np.zeros((self.domainSize, 2),
                                      dtype=simulation.precision)
            if fieldName == "rho":
                self.rho_out = np.zeros(self.domainSize,
                                        dtype=simulation.precision)
            if fieldName == "p":
                self.p_out = np.zeros(self.domainSize,
                                      dtype=simulation.precision)
            if fieldName == "phi":
                self.phi_out = np.zeros(self.domainSize,
                                        dtype=simulation.precision)
                if simulation.phaseField.outputSolidContactAngle:
                    self.thetaSolid_out = np.zeros(self.domainSize,
                                                   dtype=simulation.precision)
            if fieldName == "T":
                self.T_out = np.zeros(self.domainSize,
                                      dtype=simulation.precision)
        self.fieldsOutputFile = None
        self.obsOutputFile = None
        self.obstaclesPresent = False
        self.boundaryForcePresent = False

    def createFieldOutputFile(self, simulation):
        if os.path.isfile("output.h5"):
            print("Error! Previous output file exists!")
            os._exit(0)
        self.fieldsOutputFile = h5py.File(
            "output.h5", "a", libver="latest"
        )
        parameter_group_name = r"parameters"
        parameter_group = self.fieldsOutputFile.\
            create_group(parameter_group_name)
        parameter_group.attrs["Nx"] = simulation.mesh.Nx - 2
        parameter_group.attrs["Ny"] = simulation.mesh.Ny - 2
        parameter_group.attrs["endTime"] = simulation.endTime
        parameter_group.attrs["saveInterval"] = simulation.saveInterval
        self.fieldsOutputFile.flush()
        self.fieldsOutputFile.close()

    def createObstacleOutputFile(self, simulation):
        if os.path.isfile("obstacleData.h5"):
            print("Error! Previous obstacle data output file exists!")
            os._exit(0)
        if (simulation.options.computeForces is True or
                simulation.options.computeTorque is True):
            self.obsOutputFile = h5py.File(
                "obstacleData.h5", "a", libver="latest"
            )
            parameter_group_name = r"parameters"
            parameter_group = self.obsOutputFile.\
                create_group(parameter_group_name)
            parameter_group.attrs["endTime"] = simulation.endTime
            parameter_group.attrs["saveInterval"] =\
                simulation.obstacle.writeInterval
            if (simulation.obstacle.noOfObstacles > 0):
                self.obstaclesPresent = True
                particle_group_name = r"particles"
                self.obsOutputFile.\
                    create_group(particle_group_name)
            if ((simulation.options.noOfSurfaces -
                    simulation.obstacle.noOfObstacles) > 0):
                self.boundaryForcePresent = True
                boundary_group_name = r"boundaries"
                self.obsOutputFile.\
                    create_group(boundary_group_name)
            self.createParticleDataSet(simulation)
            self.createBoundaryDataSet(simulation)
            self.obsOutputFile.flush()
            self.obsOutputFile.close()

    def createDataSet(self, group, name, noOfSnaps, noOfComponents):
        return group.create_dataset(
            name,
            shape=(noOfSnaps, noOfComponents),
            dtype=np.float64
        )

    def createParticleDataSet(self, simulation):
        noOfSnaps = simulation.endTime //\
            simulation.obstacle.writeInterval + 1
        groups = ["position", "force", "forceCap", "forceHyd",
                  "velocity", "torque", "torqueCap", "torqueHyd",
                  "angularVelocity", "inclinationAngle"]
        for particleNo, particleName in\
                enumerate(simulation.options.surfaceNamesGlobal):
            if simulation.options.obstacleFlag[particleNo] == 1:
                particle_group = self.obsOutputFile.\
                    create_group(r"particles/" + particleName)
                for groupNo, groupName in enumerate(groups):
                    if groupNo <= 4:
                        self.createDataSet(
                            particle_group, groupName, noOfSnaps, 2
                        )
                    else:
                        self.createDataSet(
                            particle_group, groupName, noOfSnaps, 1
                        )

    def createBoundaryDataSet(self, simulation):
        noOfSnaps = simulation.endTime //\
            simulation.obstacle.writeInterval + 1
        groups = ["force", "forceCap", "forceHyd",
                  "torque", "torqueCap", "torqueHyd"]
        for boundaryNo, boundaryName in\
                enumerate(simulation.options.surfaceNamesGlobal):
            if simulation.options.obstacleFlag[boundaryNo] == 0:
                boundary_group = self.obsOutputFile.\
                    create_group(r"boundaries/" + boundaryName)
                for groupNo, groupName in enumerate(groups):
                    if groupNo <= 2:
                        self.createDataSet(
                            boundary_group, groupName, noOfSnaps, 2
                        )
                    else:
                        self.createDataSet(
                            boundary_group, groupName, noOfSnaps, 1
                        )

    def setupParallel_cpu(self, parallel):
        self.extractCoreDomainDataScalar =  \
            numba.njit(copyCoreDomainDataScalar, parallel=parallel,
                       cache=False, nogil=True)
        self.extractCoreDomainDataVector =  \
            numba.njit(copyCoreDomainDataVector, parallel=parallel,
                       cache=False, nogil=True)

    def writeData(self, timeStep, simulation):
        if timeStep % simulation.saveInterval == 0:
            lockFile = open("output.h5.lock", "w")
            self.fieldsOutputFile = h5py.File(
                "output.h5", "a", libver="latest"
            )
            groupName = r"fields/t_" + str(timeStep)
            group = self.fieldsOutputFile.create_group(groupName)
            self.extractCoreDomainDataVector(
                self.solid_out, simulation.fields.solid, simulation.mesh.Ny,
                self.Ny_0, self.domainSize
            )
            group.create_dataset("obstacle", data=self.solid_out)
            for fieldName in simulation.fields.fieldList:
                if fieldName == "u":
                    self.extractCoreDomainDataVector(
                        self.u_out, simulation.fields.u, simulation.mesh.Ny,
                        self.Ny_0, self.domainSize
                    )
                    group.create_dataset("u", data=self.u_out)
                if fieldName == "rho":
                    self.extractCoreDomainDataScalar(
                        self.rho_out, simulation.fields.rho,
                        simulation.mesh.Ny, self.Ny_0, self.domainSize,
                        simulation.fields.rho, 1
                    )
                    group.create_dataset("rho", data=self.rho_out)
                if fieldName == "p":
                    self.extractCoreDomainDataScalar(
                        self.p_out, simulation.fields.p, simulation.mesh.Ny,
                        self.Ny_0, self.domainSize, simulation.fields.rho,
                        simulation.lattice.cs * simulation.lattice.cs,
                        field_aux_present=True
                    )
                    group.create_dataset("p", data=self.p_out)
                if fieldName == "phi":
                    self.extractCoreDomainDataScalar(
                        self.phi_out, simulation.fields.phi,
                        simulation.mesh.Ny, self.Ny_0, self.domainSize,
                        simulation.fields.phi, 1
                    )
                    group.create_dataset("phi", data=self.phi_out)
                    if simulation.phaseField.outputSolidContactAngle:
                        self.extractCoreDomainDataScalar(
                            self.thetaSolid_out,
                            simulation.fields.contactAngleLocalSolid,
                            simulation.mesh.Ny, self.Ny_0, self.domainSize,
                            simulation.fields.contactAngleLocalSolid, 1
                        )
                        group.create_dataset("contactAngleLocalSolid",
                                             data=self.thetaSolid_out)
                if fieldName == "T":
                    self.extractCoreDomainDataScalar(
                        self.T_out, simulation.fields.T, simulation.mesh.Ny,
                        self.Ny_0, self.domainSize, simulation.fields.T, 1
                    )
                    group.create_dataset("T", data=self.T_out)
            self.fieldsOutputFile["parameters"].\
                attrs["lastValidTimeStep"] = timeStep
            self.fieldsOutputFile.flush()
            self.fieldsOutputFile.close()
            lockFile.close()
            os.remove("output.h5.lock")
        if (self.obsOutputFile is not None and
                timeStep % simulation.obstacle.writeInterval == 0):
            lockFile = open("obstacleData.h5.lock", "w")
            self.obsOutputFile = h5py.File(
                "obstacleData.h5", "a", libver="latest"
            )
            save_idx = timeStep // simulation.obstacle.writeInterval
            for number, name in\
                    enumerate(simulation.options.surfaceNamesGlobal):
                if (simulation.options.obstacleFlag[number] == 1
                        and self.obstaclesPresent is True):
                    particle_group =\
                        self.obsOutputFile[r"particles/" + name]
                    particle_group["position"][save_idx] =\
                        simulation.obstacle.obsOrigin[number] - 1
                    particle_group["force"][save_idx] =\
                        simulation.options.forces[number]
                    particle_group["forceCap"][save_idx] =\
                        simulation.options.capForces[number]
                    particle_group["forceHyd"][save_idx] =\
                        simulation.options.hydForces[number]
                    particle_group["velocity"][save_idx] =\
                        simulation.obstacle.obsU[number]
                    particle_group["torque"][save_idx] =\
                        simulation.options.torque[number]
                    particle_group["torqueCap"][save_idx] =\
                        simulation.options.capTorque[number]
                    particle_group["torqueHyd"][save_idx] =\
                        simulation.options.hydTorque[number]
                    particle_group["angularVelocity"][save_idx] =\
                        simulation.obstacle.obsOmega[number]
                    particle_group["inclinationAngle"][save_idx] =\
                        simulation.obstacle.inclinationAngle[number]
                elif (simulation.options.obstacleFlag[number] == 0
                        and self.boundaryForcePresent is True):
                    boundary_group =\
                        self.obsOutputFile[r"boundaries/" + name]
                    boundary_group["force"][save_idx] =\
                        simulation.options.forces[number]
                    boundary_group["forceCap"][save_idx] =\
                        simulation.options.capForces[number]
                    boundary_group["forceHyd"][save_idx] =\
                        simulation.options.hydForces[number]
                    boundary_group["torque"][save_idx] =\
                        simulation.options.torque[number]
                    boundary_group["torqueCap"][save_idx] =\
                        simulation.options.capTorque[number]
                    boundary_group["torqueHyd"][save_idx] =\
                        simulation.options.hydTorque[number]
            self.obsOutputFile["parameters"].\
                attrs["lastValidTimeStep"] = timeStep
            self.obsOutputFile.flush()
            self.obsOutputFile.close()
            lockFile.close()
            os.remove("obstacleData.h5.lock")


def saveState(timeStep, simulation):
    if not os.path.isdir('states'):
        os.makedirs('states')
    if not os.path.isdir('states/' + str(timeStep)):
        os.makedirs('states/' + str(timeStep))
    fieldsFile = open('states/' + str(timeStep) + '/fields.pkl', 'wb')
    pickle.dump(simulation.fields, fieldsFile,
                protocol=pickle.HIGHEST_PROTOCOL)


def loadState(timeStep):
    if not os.path.isdir('states'):
        print('ERROR! no previous states present!')
    else:
        try:
            fileName = 'states/' + str(timeStep) + '/fields.pkl'
            fields = pickle.load(fileName)
            return fields
        except Exception:
            print('No saved states at time ' + str(timeStep) + ' present')
            print('creating new initial state')
            return None


def copyFields_cuda(device, fields, flag):
    if flag == 'standard':
        device.u.copy_to_host(fields.u)
        device.rho.copy_to_host(fields.rho)
    elif flag == 'all':
        device.f.copy_to_host(fields.f)
        device.f_new.copy_to_host(fields.f_new)
        device.f_eq.copy_to_host(fields.f_eq)
    pass


def writeFields_mpi(timeStep, fields, lattice, mesh, rank, comm):
    u = False
    rho = False
    phi = False
    T = False
    p = False
    if not os.path.isdir('procs'):
        os.makedirs('procs')
    if not os.path.isdir('procs/proc_' + str(rank)):
        os.makedirs('procs/proc_' + str(rank))
    if not os.path.isdir('procs/proc_' + str(rank) + '/' + str(timeStep)):
        os.makedirs('procs/proc_' + str(rank) + '/' + str(timeStep))
    pointsFile = open('procs/proc_' + str(rank) + '/' +
                      str(timeStep) + '/points.dat', 'w')
    solidFile = open('procs/proc_' + str(rank) + '/' +
                     str(timeStep) + '/solid.dat', 'w')
    boundaryNodeFile = open('procs/proc_' + str(rank) + '/' +
                            str(timeStep) + '/boundaryNode.dat', 'w')
    for field in fields.fieldList:
        if field == 'u':
            u = True
            uFile = open('procs/proc_' + str(rank) + '/' +
                         str(timeStep) + '/u.dat', 'w')
        if field == 'rho':
            rho = True
            rhoFile = open('procs/proc_' + str(rank) + '/' +
                           str(timeStep) + '/rho.dat', 'w')
        if field == 'p':
            p = True
            pFile = open('procs/proc_' + str(rank) + '/' +
                         str(timeStep) + '/p.dat', 'w')
        if field == 'phi':
            phi = True
            phiFile = open('procs/proc_' + str(rank) + '/' +
                           str(timeStep) + '/phi.dat', 'w')
        if field == 'T':
            T = True
            TFile = open('procs/proc_' + str(rank) + '/' +
                         str(timeStep) + '/T.dat', 'w')
    for ind in range(mesh.Nx * mesh.Ny):
        pointsFile.write(str(np.round(ind, 10)).ljust(12) + '\n')
        if u is True:
            uFile.write(str(np.round(fields.u[ind, 0], 10)).ljust(12) + '\t' +
                        str(np.round(fields.u[ind, 1], 10)).ljust(12) + '\n')
        if rho is True:
            rhoFile.write(str(np.round(fields.rho[ind], 10)).ljust(12) + '\n')
        if p is True:
            pFile.write(str(np.round(fields.p[ind] * fields.rho[ind]
                        * lattice.cs * lattice.cs, 10)).ljust(12) + '\n')
        if phi is True:
            phiFile.write(str(np.round(fields.phi[ind], 10)).ljust(12) + '\n')
        if T is True:
            TFile.write(str(np.round(fields.T[ind], 10)).ljust(12) + '\n')
        solidFile.write(str(np.round(fields.solid[ind, 0], 10)).ljust(12)
                        + '\n')
        boundaryNodeFile.write(str(np.round(fields.boundaryNode[ind], 10)).
                               ljust(12)
                               + '\n')
    pointsFile.close()
    solidFile.close()
    boundaryNodeFile.close()
    if u is True:
        uFile.close()
    if rho is True:
        rhoFile.close()
    if p is True:
        pFile.close()
    if phi is True:
        phiFile.close()
    if T is True:
        TFile.close()
