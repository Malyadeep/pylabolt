import os


class transportDef:
    def __init__(self):
        # Required for fluid flow
        self.nu = 0.

        # Required for heat transfer
        self.alpha = 0.

        # Required for multiphase
        self.rho_l = 0.
        self.rho_g = 0.
        self.mu_l = 0.
        self.mu_g = 0.
        self.sigma = 0.
        self.phi_l = 1.
        self.phi_g = 0.

    def readFluidTransportDict(self, transportDict, precision, rank):
        try:
            self.nu = precision(transportDict['nu'])
        except KeyError as e:
            if rank == 0:
                print("ERROR! Keyword: " + str(e) +
                      " missing in 'transportDict'")
            os._exit(1)

    def readPhaseTransportDict(self, transportDict, precision, rank):
        try:
            self.rho_l = precision(transportDict['rho_l'])
            self.rho_g = precision(transportDict['rho_g'])
            self.mu_l = precision(transportDict['mu_l'])
            self.mu_g = precision(transportDict['mu_g'])
            self.sigma = precision(transportDict['sigma'])
        except KeyError as e:
            if rank == 0:
                print("ERROR! Keyword: " + str(e) +
                      " missing in 'transportDict'")
            os._exit(1)
