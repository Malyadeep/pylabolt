import os
from pylabolt.base.models import collisionModels, equilibriumModels


class collisionScheme:
    def __init__(self, lattice, collisionDict, parallelization):
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
                print("ERROR! Unsupported collision model : " +
                      collisionDict['model'])
            self.collisionModel = collisionDict['model']
            if collisionDict['equilibrium'] == 'firstOrder':
                if parallelization == 'cuda':
                    self.equilibriumType = 1     # Stands for first order
                    self.equilibriumFunc = equilibriumModels.firstOrder
                    self.equilibriumArgs = (self.cs_2, self.c, self.w)
                else:
                    self.equilibriumFunc = equilibriumModels.firstOrder
                    self.equilibriumArgs = (self.cs_2, self.c, self.w)
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
            else:
                print("ERROR! Unsupported equilibrium model : " +
                      collisionDict['equilibrium'])
                os._exit(1)
            self.equilibriumModel = collisionDict['equilibrium']
        except KeyError as e:
            print("ERROR! Keyword: " + str(e) + " missing in 'latticeDict'")
            os._exit(1)
        try:
            self.tau = collisionDict['tau']
        except KeyError as e:
            print("ERROR! Keyword: " + str(e) + " missing in 'collisionDict'")
            os._exit(1)
        self.preFactor = self.deltaT/self.tau
