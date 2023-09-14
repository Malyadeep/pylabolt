import os
import numpy as np

from pylabolt.base.models import (collisionModels, equilibriumModels,
                                  forcingModels)


class collisionScheme:
    def __init__(self, lattice, mesh, collisionDict, parallelization,
                 transport, rank, precision, comm):
        try:
            self.deltaX = lattice.deltaX
            self.deltaT = lattice.deltaT
            self.cs_2 = 1/(lattice.cs*lattice.cs)
            self.cs_4 = self.cs_2/(lattice.cs*lattice.cs)
            self.c = lattice.c
            self.w = lattice.w
            fluid = collisionDict['fluid']
            self.setCollisionSchemeFluid(fluid, precision, transport,
                                         rank, lattice, mesh, parallelization,
                                         comm)
        except KeyError as e:
            if rank == 0:
                print("ERROR! Keyword: " + str(e) +
                      " missing in 'collisionDict'")
            os._exit(1)

    def setCollisionSchemeFluid(self, fluid, precision, transport, rank,
                                lattice, mesh, parallelization, comm):
        try:
            if fluid['model'] == 'BGK':
                if parallelization == 'cuda':
                    self.collisionType = 1      # Stands for BGK
                    self.collisionFunc = collisionModels.BGK
                else:
                    self.collisionFunc = collisionModels.BGK
                self.tau = self.cs_2 * transport.nu + 0.5
                diagVec = \
                    np.full(lattice.noOfDirections,
                            fill_value=(self.deltaT/self.tau),
                            dtype=precision)
                self.preFactor = np.diag(diagVec, k=0)
            elif fluid['model'] == 'MRT':
                if parallelization == 'cuda':
                    self.collisionType = 2      # Stands for MRT
                    self.collisionFunc = collisionModels.MRT
                else:
                    self.collisionFunc = collisionModels.MRT
                try:
                    from pylabolt.base.models.collisionParams import MRT_def
                    self.nu_bulk = fluid['nu_B']
                    self.tau_nu = self.cs_2 * transport.nu + 0.5
                    self.tau_bulk = self.cs_2 * (transport.nu/3 +
                                                 self.nu_bulk) + 0.5
                    self.S_nu = self.deltaT/self.tau_nu
                    self.S_bulk = self.deltaT/self.tau_bulk
                    self.S_q = fluid['S_q']
                    self.S_epsilon = fluid['S_epsilon']
                    self.MRT = MRT_def(precision, self.S_nu, self.S_bulk,
                                       self.S_q, self.S_epsilon)
                except KeyError as e:
                    if rank == 0:
                        print("ERROR! Keyword: " + str(e) +
                              " missing in 'fluid' in 'collisionDict'")
                    os._exit(1)
                self.preFactor = self.MRT.preFactorMat
            else:
                if rank == 0:
                    print("ERROR! Unsupported collision model : " +
                          fluid['model'])
                os._exit(1)
            self.collisionModelFluid = fluid['model']
            if fluid['equilibrium'] == 'linear':
                try:
                    self.rho_0 = precision(fluid['rho_ref'])
                except KeyError:
                    if rank == 0:
                        print("ERROR! Missing keyword 'rho_ref' " +
                              "in 'fluid' in 'collisionDict'!")
                    os._exit(1)
                if parallelization == 'cuda':
                    self.equilibriumType = 1     # Stands for first order
                    self.equilibriumFunc = equilibriumModels.linear
                    self.equilibriumArgs = (self.rho_0, self.cs_2, self.c,
                                            self.w)
                else:
                    self.equilibriumFunc = equilibriumModels.linear
                    self.equilibriumArgs = (self.rho_0, self.cs_2, self.c,
                                            self.w)
            elif fluid['equilibrium'] == 'secondOrder':
                if parallelization == 'cuda':
                    self.equilibriumType = 2      # Stands for second order
                    self.equilibriumFunc = equilibriumModels.secondOrder
                    self.equilibriumArgs = (self.cs_2, self.cs_4,
                                            self.c, self.w)
                else:
                    self.equilibriumFunc = equilibriumModels.secondOrder
                    self.equilibriumArgs = (self.cs_2, self.cs_4,
                                            self.c, self.w)
            elif fluid['equilibrium'] == 'incompressible':
                try:
                    self.rho_0 = precision(fluid['rho_ref'])
                except KeyError:
                    if rank == 0:
                        print("ERROR! Missing keyword 'rho_ref' " +
                              "in 'fluid' in 'collisionDict'!")
                    comm.Barrier()
                    os._exit(1)
                if parallelization == 'cuda':
                    self.equilibriumType = 3      # Stands for incompressible
                    self.equilibriumFunc = equilibriumModels.incompressible
                    self.equilibriumArgs = (self.rho_0, self.cs_2, self.cs_4,
                                            self.c, self.w)
                else:
                    self.equilibriumFunc = equilibriumModels.incompressible
                    self.equilibriumArgs = (self.rho_0, self.cs_2, self.cs_4,
                                            self.c, self.w)
            elif fluid['equilibrium'] == 'oseen':
                try:
                    self.rho_0 = precision(fluid['rho_ref'])
                    self.U_0 = fluid['U_ref']
                    if isinstance(self.U_0, list):
                        self.U_0 = np.array(self.U_0, dtype=precision)
                    else:
                        if rank == 0:
                            print("ERROR!")
                            print("For 'oseen' model, 'U_ref' must be a list "
                                  + "of velocity components: [x1, x2]")
                        os._exit(1)
                except KeyError as e:
                    if rank == 0:
                        print("ERROR! Missing keyword" +
                              " in collisionDict!")
                        print(str(e))
                    os._exit(1)
                if parallelization == 'cuda':
                    self.equilibriumType = 4    # Stands for oseen equilibrium
                    self.equilibriumFunc = equilibriumModels.oseen
                    self.equilibriumArgs = (self.rho_0, self.U_0, self.cs_2,
                                            self.cs_4, self.c, self.w)
                else:
                    self.equilibriumFunc = equilibriumModels.oseen
                    self.equilibriumArgs = (self.rho_0, self.U_0, self.cs_2,
                                            self.cs_4, self.c, self.w)
            else:
                if rank == 0:
                    print("ERROR! Unsupported equilibrium model : " +
                          fluid['equilibrium'])
                os._exit(1)
            self.equilibriumModel = fluid['equilibrium']
        except KeyError as e:
            if rank == 0:
                print("ERROR! Keyword: " + str(e) +
                      " missing in 'fluid' in 'collisionDict'")
            os._exit(1)


class forcingScheme:
    def __init__(self, precision, lattice):
        self.forceFunc_force = None
        self.forceArgs_force = None
        self.forceFunc_vel = None
        self.forceCoeffVel = None
        self.forcingPreFactor = np.zeros((lattice.noOfDirections,
                                         lattice.noOfDirections),
                                         dtype=precision)
        self.forcingFlag = 0

    def setForcingScheme(self, lattice, collisionScheme, rank, precision):
        try:
            from simulation import forcingDict
            keyList = list(forcingDict.keys())
            if len(keyList) == 0:
                if rank == 0:
                    print("No forcing scheme selected!")
                self.forcingFlag = 0
                self.gravity = np.zeros(2, dtype=precision)
                return
            self.cs_2 = 1/(lattice.cs*lattice.cs)
            self.cs_4 = self.cs_2/(lattice.cs*lattice.cs)
            self.c = lattice.c
            self.w = lattice.w
            self.noOfDirections = lattice.noOfDirections
            self.preFactor = collisionScheme.preFactor

            self.forcingModel = forcingDict['model']
            self.gravity = forcingDict['g']

            if isinstance(self.gravity, list) is True:
                self.gravity = np.array(self.gravity, dtype=precision)
            else:
                if rank == 0:
                    print("ERROR! gravity must be a list of"
                          + " components: [x1, x2]", flush=True)
                os._exit(1)
            if self.forcingModel == 'Guo':
                self.forcingType = 1      # Stands for Guo
                self.forcingFlag = 1
                self.forceFunc_vel = forcingModels.Guo_vel
                self.forceFunc_force = forcingModels.Guo_force
                self.forceCoeffVel = 0.5
                if collisionScheme.collisionModelFluid == 'BGK':
                    diagVec = np.diag(self.preFactor, k=0)
                    self.forcingPreFactor = np.diag(1 - 0.5 * diagVec, k=0)
                elif collisionScheme.collisionModelFluid == 'MRT':
                    self.forcingPreFactor = collisionScheme.MRT.setForcingGuo()
                self.forceArgs_force = (self.c, self.w,
                                        self.noOfDirections, self.cs_2,
                                        self.cs_4)
            elif self.forcingModel == 'GuoLinear':
                self.forcingType = 1      # Stands for Guo
                self.forcingFlag = 1
                self.forceFunc_vel = forcingModels.Guo_vel
                self.forceFunc_force = forcingModels.Guo_force_linear
                self.forceCoeffVel = 0.5
                if collisionScheme.collisionModelFluid == 'BGK':
                    diagVec = np.diag(self.preFactor, k=0)
                    self.forcingPreFactor = np.diag(1 - 0.5 * diagVec, k=0)
                elif collisionScheme.collisionModelFluid == 'MRT':
                    self.forcingPreFactor = collisionScheme.MRT.setForcingGuo()
                self.forceArgs_force = (self.c, self.w,
                                        self.noOfDirections)
            else:
                if rank == 0:
                    print("ERROR! Unsupported forcing model : " +
                          self.forcingModel, flush=True)
                os._exit(1)
        except KeyError as e:
            if rank == 0:
                print("ERROR! Keyword: " + str(e) +
                      " missing in 'forcingDict'")
            os._exit(1)
        except ImportError:
            if rank == 0:
                print("No forcing scheme selected!")
                self.forcingFlag = 0
            self.gravity = np.zeros(2, dtype=precision)
            return
