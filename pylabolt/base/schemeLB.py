import os
import numpy as np

from pylabolt.base.models import (collisionModels, equilibriumModels,
                                  forcingModels)


class collisionScheme:
    def __init__(self, lattice, collisionDict, parallelization,
                 rank, precision):
        try:
            self.deltaX = lattice.deltaX
            self.deltaT = lattice.deltaT
            self.cs_2 = 1/(lattice.cs*lattice.cs)
            self.cs_4 = self.cs_2/(lattice.cs*lattice.cs)
            self.c = lattice.c
            self.w = lattice.w
            if collisionDict['model'] == 'BGK':
                if parallelization == 'cuda':
                    self.collisionType = 1      # Stands for BGK
                    self.collisionFunc = collisionModels.BGK
                else:
                    self.collisionFunc = collisionModels.BGK

            else:
                if rank == 0:
                    print("ERROR! Unsupported collision model : " +
                          collisionDict['model'])
                os._exit(1)
            self.collisionModel = collisionDict['model']
            if collisionDict['equilibrium'] == 'stokesLinear':
                try:
                    self.rho_0 = precision(collisionDict['rho_ref'])
                except KeyError:
                    if rank == 0:
                        print("ERROR! Missing keyword 'rho_ref' " +
                              "in collisionDict!")
                    os._exit(1)
                if parallelization == 'cuda':
                    self.equilibriumType = 1     # Stands for first order
                    self.equilibriumFunc = equilibriumModels.stokesLinear
                    self.equilibriumArgs = (self.rho_0, self.cs_2, self.c,
                                            self.w)
                else:
                    self.equilibriumFunc = equilibriumModels.stokesLinear
                    self.equilibriumArgs = (self.rho_0, self.cs_2, self.c,
                                            self.w)
            elif collisionDict['equilibrium'] == 'secondOrder':
                if parallelization == 'cuda':
                    self.equilibriumType = 2      # Stands for second order
                    self.equilibriumFunc = equilibriumModels.secondOrder
                    self.equilibriumArgs = (self.cs_2, self.cs_4,
                                            self.c, self.w)
                else:
                    self.equilibriumFunc = equilibriumModels.secondOrder
                    self.equilibriumArgs = (self.cs_2, self.cs_4,
                                            self.c, self.w)
            elif collisionDict['equilibrium'] == 'incompressible':
                try:
                    self.rho_0 = precision(collisionDict['rho_ref'])
                except KeyError as e:
                    if rank == 0:
                        print("ERROR! Missing keyword" +
                              " in collisionDict!")
                        print(str(e))
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
            elif collisionDict['equilibrium'] == 'oseen':
                try:
                    self.rho_0 = precision(collisionDict['rho_ref'])
                    self.U_0 = collisionDict['U_ref']
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
                          collisionDict['equilibrium'])
                os._exit(1)
            self.equilibriumModel = collisionDict['equilibrium']
        except KeyError as e:
            if rank == 0:
                print("ERROR! Keyword: " + str(e) +
                      " missing in 'collisionDict'")
            os._exit(1)
        try:
            self.tau = collisionDict['tau']
        except KeyError as e:
            if rank == 0:
                print("ERROR! Keyword: " + str(e) +
                      " missing in 'collisionDict'")
            os._exit(1)
        self.preFactor = self.deltaT/self.tau


class forcingScheme:
    def __init__(self, precision, lattice):
        self.forceFunc_force = None
        self.forceArgs_force = None
        self.forceFunc_vel = None
        self.forceArgs_vel = None

    def setForcingScheme(self, lattice, collisionScheme, rank, precision):
        try:
            from simulation import forcingDict
            keyList = list(forcingDict.keys())
            if len(keyList) == 0:
                if rank == 0:
                    print("No forcing scheme selected!")
                return
            self.cs_2 = 1/(lattice.cs*lattice.cs)
            self.cs_4 = self.cs_2/(lattice.cs*lattice.cs)
            self.c = lattice.c
            self.w = lattice.w
            self.noOfDirections = lattice.noOfDirections
            self.tau = collisionScheme.tau

            self.forcingModel = forcingDict['model']
            self.forcingValue = forcingDict['value']

            if isinstance(self.forcingValue, list) is True:
                self.F = np.array(self.forcingValue, dtype=precision)
            else:
                if rank == 0:
                    print("ERROR! force value must be a list of"
                          + " components: [x1, x2]", flush=True)
                os._exit(1)
            if self.forcingModel == 'Guo':
                self.forcingType = 1      # Stands for Guo
                self.forceFunc_vel = forcingModels.Guo_vel
                self.forceFunc_force = forcingModels.Guo_force
                self.A = 0.5
                self.forceArgs_vel = (self.F, self.A)
                self.forceArgs_force = (self.F, self.c, self.w,
                                        self.noOfDirections, self.cs_2,
                                        self.cs_4, self.tau)
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
            return
