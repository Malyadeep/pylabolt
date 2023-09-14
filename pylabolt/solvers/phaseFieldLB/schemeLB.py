
import os
import numpy as np

from pylabolt.base.models import (collisionModels, equilibriumModels,
                                  forcingModels, velocityEquilibriumModels)
from pylabolt.base.models.collisionParams import (MRT_def,
                                                  constructBGKOperator,
                                                  constructMRTOperator)
from pylabolt.solvers.phaseFieldLB.phaseField import (stressTensorBGK,
                                                      stressTensorMRT)


class collisionScheme:
    def __init__(self, lattice, mesh, collisionDict, parallelization,
                 transport, phaseField, rank, precision):
        try:
            self.deltaX = lattice.deltaX
            self.deltaT = lattice.deltaT
            self.cs_2 = 1/(lattice.cs*lattice.cs)
            self.cs_4 = self.cs_2/(lattice.cs*lattice.cs)
            self.c = lattice.c
            self.w = lattice.w
            phase = collisionDict['phase']
            self.setCollisionSchemePhase(phase, precision, phaseField,
                                         rank, lattice, parallelization)
            fluid = collisionDict['fluid']
            self.setCollisionSchemeFluid(fluid, precision, transport,
                                         rank, lattice, mesh, parallelization)
        except KeyError as e:
            if rank == 0:
                print("ERROR! Keyword: " + str(e) +
                      " missing in 'collisionDict'")
            os._exit(1)

    def setCollisionSchemePhase(self, phase, precision, phaseField, rank,
                                lattice, parallelization):
        try:
            if phase['model'] == 'BGK':
                if parallelization == 'cuda':
                    self.collisionTypePhase = 1      # Stands for BGK
                    self.collisionFuncPhase = collisionModels.BGK
                else:
                    self.collisionFuncPhase = collisionModels.BGK
                self.tauPhase = self.cs_2 * phaseField.M + 0.5
                diagValue = \
                    np.full(lattice.noOfDirections,
                            fill_value=(self.deltaT/self.tauPhase),
                            dtype=precision)
                self.preFactorPhase = np.diag(diagValue, k=0)
            else:
                if rank == 0:
                    print("ERROR! Unsupported collision model for 'phase': " +
                          phase['model'])
                os._exit(1)
            self.collisionModelPhase = phase['model']
            if phase['equilibrium'] == 'linear':
                try:
                    self.rho_0 = precision(phase['rho_ref'])
                except KeyError:
                    if rank == 0:
                        print("ERROR! Missing keyword 'rho_ref' " +
                              "in collisionDict!")
                    os._exit(1)
                if parallelization == 'cuda':
                    self.equilibriumTypePhase = 1     # Stands for first order
                    self.equilibriumFuncPhase = equilibriumModels.stokesLinear
                    self.equilibriumArgsPhase = (self.rho_0, self.cs_2, self.c,
                                                 self.w)
                else:
                    self.equilibriumFuncPhase = equilibriumModels.stokesLinear
                    self.equilibriumArgsPhase = (self.rho_0, self.cs_2, self.c,
                                                 self.w)
            elif phase['equilibrium'] == 'secondOrder':
                if parallelization == 'cuda':
                    self.equilibriumTypePhase = 2     # Stands for second order
                    self.equilibriumFuncPhase = equilibriumModels.secondOrder
                    self.equilibriumArgsPhase = (self.cs_2, self.cs_4,
                                                 self.c, self.w)
                else:
                    self.equilibriumFuncPhase = equilibriumModels.secondOrder
                    self.equilibriumArgsPhase = (self.cs_2, self.cs_4,
                                                 self.c, self.w)
            else:
                if rank == 0:
                    print("ERROR! Unsupported equilibrium model for 'phase': "
                          + phase['equilibrium'])
                os._exit(1)
            self.equilibriumModelPhase = phase['equilibrium']
        except KeyError as e:
            if rank == 0:
                print("ERROR! Keyword: " + str(e) +
                      " missing in 'phase'")
            os._exit(1)

    def setCollisionSchemeFluid(self, fluid, precision, transport, rank,
                                lattice, mesh, parallelization):
        try:
            if fluid['model'] == 'BGK':
                if parallelization == 'cuda':
                    self.collisionTypeFluid = 1      # Stands for BGK
                    self.collisionFuncFluid = collisionModels.BGK
                else:
                    self.collisionFuncFluid = collisionModels.BGK
                    self.constructOperatorFunc = constructBGKOperator
                    self.constructStressTensorFunc = stressTensorBGK
                self.collisionOperatorArgs = (lattice.cs_2,
                                              lattice.noOfDirections)
            elif fluid['model'] == 'MRT':
                if lattice.noOfDirections != 9:
                    if rank == 0:
                        print('ERROR! MRT collision model is currently only' +
                              'available for D2Q9 lattices!')
                    os._exit(1)
                if parallelization == 'cuda':
                    self.collisionTypeFluid = 2      # Stands for MRT
                    self.collisionFuncFluid = collisionModels.MRT
                else:
                    self.collisionFuncFluid = collisionModels.MRT
                    self.constructOperatorFunc = constructMRTOperator
                    self.constructStressTensorFunc = stressTensorMRT
                try:
                    self.nu_bulk = fluid['nu_B']
                    self.S_nu = 1.0
                    self.S_bulk = 1.0
                    self.S_q = fluid['S_q']
                    self.S_epsilon = fluid['S_epsilon']
                    self.MRT = MRT_def(precision, self.S_nu, self.S_bulk,
                                       self.S_q, self.S_epsilon)
                except KeyError as e:
                    if rank == 0:
                        print("ERROR! Keyword: " + str(e) +
                              " missing in 'collisionDict'")
                    os._exit(1)
                self.collisionOperatorArgs = (self.nu_bulk, self.MRT.M,
                                              self.MRT.M_1, self.S_q,
                                              self.S_epsilon,
                                              lattice.cs_2,
                                              lattice.noOfDirections,
                                              precision)
            else:
                if rank == 0:
                    print("ERROR! Unsupported collision model for 'fluid': " +
                          fluid['model'])
                os._exit(1)
            self.collisionModelFluid = fluid['model']
            self.preFactorFluid = \
                np.zeros((mesh.Nx * mesh.Ny, lattice.noOfDirections,
                         lattice.noOfDirections), dtype=precision)
            if fluid['equilibrium'] == 'velocityBased':
                if parallelization == 'cuda':
                    self.equilibriumTypeFluid = 2     # Stands for second order
                    self.equilibriumFuncFluid = velocityEquilibriumModels.\
                        secondOrder
                    self.equilibriumArgsFluid = (self.cs_2, self.cs_4,
                                                 self.c, self.w)
                else:
                    self.equilibriumFuncFluid = velocityEquilibriumModels.\
                        secondOrder
                    self.equilibriumArgsFluid = (self.cs_2, self.cs_4,
                                                 self.c, self.w)
            else:
                if rank == 0:
                    print("ERROR! Unsupported equilibrium model for 'fluid': "
                          + fluid['equilibrium'])
                os._exit(1)
            self.equilibriumModelFluid = fluid['equilibrium']
        except KeyError as e:
            if rank == 0:
                print("ERROR! Keyword: " + str(e) +
                      " missing in 'fluid'")
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
                return
            self.cs_2 = 1/(lattice.cs*lattice.cs)
            self.cs_4 = self.cs_2/(lattice.cs*lattice.cs)
            self.c = lattice.c
            self.w = lattice.w
            self.noOfDirections = lattice.noOfDirections
            self.forcingModel = forcingDict['model']
            self.forcingPreFactorPhase = \
                np.diag(1 - 0.5 * np.diag(collisionScheme.
                        preFactorPhase, k=0), k=0)
            try:
                self.gravity = forcingDict['g']
                if not isinstance(self.gravity, list) is True:
                    if rank == 0:
                        print("ERROR! force value must be a list of"
                              + " components: [x1, x2]", flush=True)
                    os._exit(1)
                self.gravity = np.array(self.gravity, dtype=precision)
            except KeyError:
                self.gravity = np.zeros(2, dtype=precision)
                if rank == 0:
                    print('gravity is absent!')

            if self.forcingModel == 'Guo':
                self.forcingType = 1      # Guo's scheme
                self.forcingFlag = 1
                self.forceFunc_vel = forcingModels.Guo_vel
                self.forceFunc_force = forcingModels.Guo_force
                self.forceCoeffVel = 0.5
                self.forceArgs_vel = (self.forceCoeffVel)
                self.forceArgs_force = (self.c, self.w,
                                        self.noOfDirections, self.cs_2,
                                        self.cs_4)
            elif self.forcingModel == 'GuoLinear':
                self.forcingType = 2      # Guo's scheme linear in velocity
                self.forcingFlag = 1
                self.forceFunc_vel = forcingModels.Guo_vel
                self.forceFunc_force = forcingModels.Guo_force_linear
                self.forceCoeffVel = 0.5
                self.forceArgs_vel = (self.forceCoeffVel)
                self.forceArgs_force = (self.c, self.w,
                                        self.noOfDirections)
            else:
                if rank == 0:
                    print("ERROR! Unsupported forcing model : " +
                          self.forcingModel, flush=True)
                os._exit(1)
            self.forcingPreFactorFluid = np.zeros_like(collisionScheme.
                                                       preFactorFluid)
        except KeyError as e:
            if rank == 0:
                print("ERROR! Keyword: " + str(e) +
                      " missing in 'forcingDict'")
            os._exit(1)
        except ImportError:
            if rank == 0:
                print("ERROR! No forcing scheme selected!")
                print("For phase-field based simulations forcing " +
                      "scheme must be set!")
            os._exit(1)
