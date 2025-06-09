import os
import numpy as np
import numba
from numba import prange
# from line_profiler import profile
# from scalene import profile


class phaseFieldDef:
    def __init__(self, phaseDict, lattice, transport, rank, precision):
        try:
            self.M = precision(phaseDict['M'])
            self.interfaceWidth = precision(phaseDict['interfaceWidth'])
            self.viscInterpolation = phaseDict['nuInterpolation']
            if self.viscInterpolation == 'linear':
                self.computeViscFunc = computeViscLinear
                self.viscInterpolationNo = 1
            else:
                self.computeViscFunc = computeViscHarmonic
                self.viscInterpolationNo = 2
            self.contactAngle = phaseDict['contactAngle']
            self.surfaceTensionModel = phaseDict['surfaceTensionModel']
            if self.contactAngle is not None:
                if (not isinstance(self.contactAngle, int) and
                        not isinstance(self.contactAngle, float)):
                    if rank == 0:
                        print("ERROR! contact angle must be a " +
                              "float/int representing angle in degrees!",
                              flush=True)
                    os._exit(1)
                self.contactAngle = self.contactAngle / 180 * np.pi
                self.wettingModel = phaseDict["wettingModel"]
                self.contactAngleHysteresis = \
                    phaseDict["contactAngleHysteresis"]
                if self.contactAngleHysteresis is True:
                    self.contactAngleAdvancing = \
                        phaseDict["contactAngleAdvancing"]
                    self.contactAngleReceding = \
                        phaseDict["contactAngleReceding"]
                    if (not isinstance(self.contactAngleAdvancing, int) and
                            not isinstance(self.contactAngleAdvancing, float)
                            or not isinstance(self.contactAngleReceding, int)
                            and not isinstance(self.contactAngleReceding,
                                               float)):
                        if rank == 0:
                            print("ERROR! contact angle must be a " +
                                  "float/int representing angle in degrees!",
                                  flush=True)
                        os._exit(1)
                    self.contactAngleAdvancing = self.contactAngleAdvancing *\
                        np.pi / 180
                    self.contactAngleReceding = self.contactAngleReceding *\
                        np.pi / 180
                else:
                    self.contactAngleAdvancing = 0
                    self.contactAngleReceding = 0
                if self.wettingModel == "fakhari":
                    self.wettingFunction = fakhariWetting
                elif self.wettingModel == "regularized":
                    self.wettingFunction = regularized
                elif self.wettingModel == "liang":
                    self.wettingFunction = liangWetting
                else:
                    if rank == 0:
                        print("ERROR! Unsupported Wetting model!",
                              flush=True)
                    os._exit(1)
            self.displayMass = False
            try:
                self.displayMass = phaseDict['displayMass']
            except KeyError:
                pass
            if self.displayMass is True:
                self.mass = np.zeros(1, dtype=precision)
            self.alpha = 1 - 10 * self.M / 3
            phiWeight_1_4 = (1 - self.alpha) / 5
            phiWeight_5_8 = (1 - self.alpha) / 20
            self.phiWeight = \
                np.array([self.alpha, phiWeight_1_4, phiWeight_1_4,
                          phiWeight_1_4, phiWeight_1_4, phiWeight_5_8,
                          phiWeight_5_8, phiWeight_5_8, phiWeight_5_8],
                         dtype=np.float64)
            self.segregation = True
            self.diffusionTime = int((self.interfaceWidth * self.
                                     interfaceWidth) / self.M)
            self.kappa = 1.5 * transport.sigma * self.interfaceWidth
            self.beta = 12 * transport.sigma / self.interfaceWidth
            if self.surfaceTensionModel == 'chemicalPotential':
                self.surfaceTensionFunc = chemicalPotentialModel
                self.surfaceTensionArgs = (self.beta, self.kappa,
                                           transport.phi_l, transport.phi_g)
                self.surfaceTensionModelNo = 1
            elif self.surfaceTensionModel == 'continuum':
                self.surfaceTensionFunc = continuumModel
                self.surfaceTensionArgs = (transport.sigma, lattice.c,
                                           lattice.w, lattice.cs_2,
                                           lattice.noOfDirections,
                                           self.interfaceWidth)
                self.surfaceTensionModelNo = 2
            else:
                if rank == 0:
                    print("ERROR! Unsupported surface tension force model!",
                          flush=True)
                os._exit(1)
        except KeyError as e:
            if rank == 0:
                print("ERROR! Keyword: " + str(e) +
                      " missing in 'phaseDict'")
            os._exit(1)

    def setupParallel_cpu(self, parallel):
        self.computeGradLapPhi =  \
            numba.njit(computeGradLapPhi, parallel=parallel,
                       cache=False, nogil=True)
        self.forceFluid =  \
            numba.njit(forceFluid, parallel=parallel,
                       cache=False, nogil=True)
        # self.forceFluid = forceFluid
        self.computeMass =  \
            numba.njit(computeMass, parallel=parallel,
                       cache=False, nogil=True)
        if self.contactAngle is not None:
            self.wettingFunction =  \
                numba.njit(self.wettingFunction, parallel=parallel,
                           cache=False, nogil=True)
            self.copyPhi = \
                numba.njit(copyPhi, parallel=parallel,
                           cache=False, nogil=True)
        self.initExtraMass =  \
            numba.njit(initExtraMass, parallel=parallel,
                       cache=False, nogil=True)
        self.setExtraMass =  \
            numba.njit(setExtraMass, parallel=parallel,
                       cache=False, nogil=True)

    def setSolidPhaseField(self, options, fields, boundary, lattice, mesh,
                           size, timeStep, saveInterval, initial=False):
        if not os.path.isdir("output"):
            os.makedirs("output")
        self.copyPhi(mesh.Nx, mesh.Ny, fields.phi, fields.phi_temp)
        for itr in range(options.noOfSurfaces):
            # self.setSolidPhi(options.solidNbNodes[itr], fields.phi,
            #                  fields.solid, fields.boundaryNode,
            #                  fields.procBoundary, lattice.w, lattice.c,
            #                  lattice.noOfDirections, mesh.Nx, mesh.Ny,
            #                  size)
            # self.liangWetting(options.solidNbNodes[itr], fields.phi,
            #                   self.contactAngle, mesh.Nx, mesh.Ny)
            # if options.obstacleFlag[itr] == 0:
            #     contactAngle = np.pi / 2
            #     hysteresis = False
            # else:
            #     contactAngle = self.contactAngle
            #     hysteresis = self.contactAngleHysteresis
            # if itr == 2:
            #     print(options.surfaceNormals[itr])
            if options.boundaryType[itr] != "fixedValue":
                self.wettingFunction(options.solidNbNodes[itr], options.
                                     surfaceNormals[itr], fields.phi,
                                     fields.phi_temp, fields.normalPhi,
                                     fields.boundaryNode, self.contactAngle,
                                     self.interfaceWidth, mesh.Nx, mesh.Ny,
                                     size, hysteresis=self.
                                     contactAngleHysteresis,
                                     contactAngleReceding=self.
                                     contactAngleReceding,
                                     contactAngleAdvancing=self.
                                     contactAngleAdvancing,
                                     initial=initial)
            # self.wettingFunction(options.solidNbNodes[itr], options.
            #                      surfaceNormals[itr], fields.phi,
            #                      fields.normalPhi, fields.boundaryNode,
            #                      self.contactAngle, self.interfaceWidth,
            #                      mesh.Nx, mesh.Ny, size, hysteresis=self.
            #                      contactAngleHysteresis,
            #                      contactAngleReceding=self.
            #                      contactAngleReceding,
            #                      contactAngleAdvancing=self.
            #                      contactAngleAdvancing,
            #                      initial=initial)
            # if (timeStep == 9999 or timeStep == 10000 or timeStep == 19999 or
            #         timeStep == 20000 or timeStep == 0 and initial is False):
            # if timeStep % saveInterval == 0 and initial is False:
            #     np.savez("output/localTheta_surface_" + str(itr) + "_"
            #              + str(timeStep) + ".npz",
            #              theta=contactAngleLocalStore,
            #              cosTheta=cosThetaStore,
            #              solidNodes=options.solidNbNodes[itr],
            #              phi=fields.phi, solid=fields.solid)

    def initializeExtraMass(self, fields, mesh, lattice, size, rank,
                            mpiParams=None):
        if mpiParams is None:
            nx, ny = 0, 0
            nProc_x, nProc_y = 0, 0
        else:
            nx, ny = mpiParams.nx, mpiParams.ny
            nProc_x, nProc_y = mpiParams.nProc_x, mpiParams.nProc_y
        args = (fields.deltaM, fields.massAdded, fields.solid,
                fields.boundaryNode, fields.procBoundary,
                lattice.noOfDirections, lattice.c, mesh.Nx, mesh.Ny,
                nx, ny, nProc_x, nProc_y, size)
        self.initExtraMass(*args)

    def adjustExtraMass(self, fields, options, mesh, lattice, size):
        for itr in range(options.noOfSurfaces):
            if options.obstacleFlag[itr] == 1:
                args = (fields.deltaM, fields.phi, fields.solid,
                        options.solidNbNodes[itr], options.surfaceNodes[itr],
                        fields.procBoundary, fields.boundaryNode, mesh.Nx,
                        mesh.Ny, lattice.noOfDirections, lattice.c, size)
                self.setExtraMass(*args)
        # return deltaMAdded

    def correctNormalPhi(self, options, fields, precision, mesh, lattice,
                         size, timeStep, saveInterval):
        for itr in range(options.noOfSurfaces):
            pass
            # self.correctNormal(fields.normalPhi, fields.gradPhi,
            #                    options.surfaceNodes[itr],
            #                    options.surfaceNormals[itr], self.contactAngle)
            # checkDot =\
            #     self.newCorrectNormal(options.surfaceNodes[itr],
            #                           options.surfaceNormals[itr],
            #                           fields.gradPhi, fields.normalPhi,
            #                           fields.procBoundary, self.contactAngle,
            #                           precision)
            # self.leclaireWetting(options.surfaceNodes[itr],
            #                      options.surfaceNormals[itr],
            #                      fields.normalPhi, fields.gradPhi,
            #                      self.contactAngle, precision)
            # self.simpleWetting(options.surfaceNodes[itr],
            #                    options.surfaceNormals[itr],
            #                    fields.normalPhi, fields.gradPhi,
            #                    fields.phi, self.contactAngle, precision,
            #                    fields.solid, fields.boundaryNode,
            #                    mesh.Nx, mesh.Ny, size, lattice.c, lattice.w,
            #                    lattice.cs_2)
            # if itr == 0 and timeStep % saveInterval == 0:
            #     np.savez('output/' + str(timeStep) + '/checkDot.npz',
            #              checkDot=checkDot)


@numba.njit
def forcePhaseField(force, phi, gradPhi, interfaceWidth,
                    c, w, noOfDirections):
    magGradPhi = np.sqrt(gradPhi[0] * gradPhi[0] +
                         gradPhi[1] * gradPhi[1])
    constant = (1 - 4 * (phi - 0.5) * (phi - 0.5)) / interfaceWidth
    for k in range(noOfDirections):
        force[k] = constant * w[k] * (c[k, 0] * gradPhi[0] +
                                      c[k, 1] * gradPhi[1]) /\
                                        (magGradPhi + 1e-17)


@numba.njit
def computeViscLinear(mu_l, mu_g, phi_g, phi, rho, rho_l, rho_g):
    mu = mu_g + (phi - phi_g) * (mu_l - mu_g)
    return (mu/rho)


@numba.njit
def computeViscHarmonic(mu_l, mu_g, phi_g, phi, rho, rho_l, rho_g):
    nu = 1/((1 - phi)/(mu_g/rho_g) + phi/(mu_l/rho_l))
    return nu


def computeGradLapPhi(Nx, Ny, phi, gradPhi, normalPhi, lapPhi, solid,
                      procBoundary, boundaryNode, cs_2, c, w, noOfDirections,
                      size, initial=False):
    for ind in prange(Nx * Ny):
        if (procBoundary[ind] != 1 and solid[ind, 0] != 1 and
                boundaryNode[ind] != 1):
            i, j = int(ind / Ny), int(ind % Ny)
            gradPhiSum_x, gradPhiSum_y = 0., 0.
            lapPhiSum = 0.
            denominator = 0.
            noOfBoundaryNodes = 0
            for k in range(noOfDirections):
                i_nb = i + int(c[k, 0])
                j_nb = j + int(c[k, 1])
                if size == 1 and boundaryNode[ind] == 2:
                    if i + int(2 * c[k, 0]) < 0 or i + int(2 * c[k, 0]) >= Nx:
                        i_nb = (i_nb + int(2 * c[k, 0]) + Nx) % Nx
                    else:
                        i_nb = (i_nb + Nx) % Nx
                    if j + int(2 * c[k, 1]) < 0 or j + int(2 * c[k, 1]) >= Ny:
                        j_nb = (j_nb + int(2 * c[k, 1]) + Ny) % Ny
                    else:
                        j_nb = (j_nb + Ny) % Ny
                elif size > 1 and boundaryNode[ind] == 2:
                    if (i + int(2 * c[k, 0]) == 0 or
                            i + int(2 * c[k, 0]) == Nx - 1):
                        i_nb = i_nb + int(c[k, 0])
                    if (j + int(2 * c[k, 1]) == 0 or
                            j + int(2 * c[k, 1]) == Ny - 1):
                        j_nb = j_nb + int(c[k, 1])
                ind_nb = int(i_nb * Ny + j_nb)
                # gradPhiSum_x += c[k, 0] * w[k] * phi[ind_nb]
                # gradPhiSum_y += c[k, 1] * w[k] * phi[ind_nb]
                # lapPhiSum += w[k] * (phi[ind_nb] - phi[ind])
                if boundaryNode[ind_nb] != 1 or solid[ind_nb, 0] == 1:
                    gradPhiSum_x += c[k, 0] * w[k] * phi[ind_nb]
                    gradPhiSum_y += c[k, 1] * w[k] * phi[ind_nb]
                    lapPhiSum += w[k] * (phi[ind_nb] - phi[ind])
                    denominator += w[k]
                else:
                    noOfBoundaryNodes += 1
            # gradPhi[ind, 0] = cs_2 * gradPhiSum_x
            # gradPhi[ind, 1] = cs_2 * gradPhiSum_y
            # lapPhi[ind] = 2.0 * cs_2 * lapPhiSum
            if noOfBoundaryNodes == 0:
                gradPhi[ind, 0] = cs_2 * gradPhiSum_x
                gradPhi[ind, 1] = cs_2 * gradPhiSum_y
                lapPhi[ind] = 2.0 * cs_2 * lapPhiSum
            else:
                gradPhi[ind, 0] = cs_2 * gradPhiSum_x / (denominator + 1e-17)
                gradPhi[ind, 1] = cs_2 * gradPhiSum_y / (denominator + 1e-17)
                lapPhi[ind] = 2.0 * cs_2 * lapPhiSum / (denominator + 1e-17)
            magGradPhi = np.sqrt(gradPhi[ind, 0] * gradPhi[ind, 0] +
                                 gradPhi[ind, 1] * gradPhi[ind, 1])
            # print(magGradPhi)
            normalPhi[ind, 0] = gradPhi[ind, 0] / (magGradPhi + 1e-17)
            normalPhi[ind, 1] = gradPhi[ind, 1] / (magGradPhi + 1e-17)


def initExtraMass(deltaM, massAdded, solid, boundaryNode, procBoundary,
                  noOfDirections, c, Nx, Ny, nx, ny, nProc_x,
                  nProc_y, size):
    for ind in prange(Nx * Ny):
        deltaM[ind] = 0
        solidNbFlag = 0
        if boundaryNode[ind] != 1 and procBoundary[ind] == 0:
            i, j = int(ind / Ny), int(ind % Ny)
            for k in range(1, noOfDirections):
                i_nb = i + int(c[k, 0])
                j_nb = j + int(c[k, 1])
                if size == 1 and boundaryNode[ind] == 2:
                    if (i + int(2 * c[k, 0]) < 0 or i +
                            int(2 * c[k, 0]) >= Nx):
                        i_nb = (i_nb + int(2 * c[k, 0]) + Nx) % Nx
                    else:
                        i_nb = (i_nb + Nx) % Nx
                    if (j + int(2 * c[k, 1]) < 0 or j +
                            int(2 * c[k, 1]) >= Ny):
                        j_nb = (j_nb + int(2 * c[k, 1]) + Ny) % Ny
                    else:
                        j_nb = (j_nb + Ny) % Ny
                elif size > 1 and boundaryNode[ind] == 2:
                    if (i + int(2 * c[k, 0]) == 0 and nx == 0 or
                            i + int(2 * c[k, 0]) == Nx - 1 and
                            nx == nProc_x - 1):
                        i_nb = i_nb + int(c[k, 0])
                    if (j + int(2 * c[k, 1]) == 0 and ny == 0 or
                            j + int(2 * c[k, 1]) == Ny - 1 and
                            ny == nProc_y - 1):
                        j_nb = j_nb + int(c[k, 1])
                ind_nb = int(i_nb * Ny + j_nb)
                if solid[ind_nb, 0] == 1:
                    solidNbFlag = 1
                    break
            if solidNbFlag == 0:
                massAdded[ind] = 0
            # print(i, j)


def setExtraMass(deltaM, phi, solid, solidNbNodes, surfaceNodes, procBoundary,
                 boundaryNode, Nx, Ny, noOfDirections, c, size):
    for itr in range(solidNbNodes.shape[0]):
        ind = solidNbNodes[itr]
        noOfFluidNeighbours = 0
        if procBoundary[ind] == 0 and boundaryNode[ind] != 1:
            i, j = int(ind / Ny), int(ind % Ny)
            for k in range(noOfDirections):
                i_nb = i + int(c[k, 0])
                j_nb = j + int(c[k, 1])
                if size == 1 and boundaryNode[ind] == 2:
                    if i + int(2 * c[k, 0]) < 0 or i + int(2 * c[k, 0]) >= Nx:
                        i_nb = (i_nb + int(2 * c[k, 0]) + Nx) % Nx
                    else:
                        i_nb = (i_nb + Nx) % Nx
                    if j + int(2 * c[k, 1]) < 0 or j + int(2 * c[k, 1]) >= Ny:
                        j_nb = (j_nb + int(2 * c[k, 1]) + Ny) % Ny
                    else:
                        j_nb = (j_nb + Ny) % Ny
                elif size > 1 and boundaryNode[ind] == 2:
                    if (i + int(2 * c[k, 0]) == 0 or
                            i + int(2 * c[k, 0]) == Nx - 1):
                        i_nb = i_nb + int(c[k, 0])
                    if (j + int(2 * c[k, 1]) == 0 or
                            j + int(2 * c[k, 1]) == Ny - 1):
                        j_nb = j_nb + int(c[k, 1])
                ind_nb = int(i_nb * Ny + j_nb)
                if solid[ind_nb, 0] == 0:
                    noOfFluidNeighbours += 1
            # if i == 51 and j == 57:
            #     print(noOfFluidNeighbours)
            if noOfFluidNeighbours > 0:
                deltaM[ind] = deltaM[ind] / noOfFluidNeighbours
    # deltaMAdded = 0
    for itr in prange(surfaceNodes.shape[0]):
        ind = surfaceNodes[itr]
        if procBoundary[ind] == 0 and boundaryNode[ind] != 1:
            i, j = int(ind / Ny), int(ind % Ny)
            for k in range(noOfDirections):
                i_nb = i + int(c[k, 0])
                j_nb = j + int(c[k, 1])
                if size == 1 and boundaryNode[ind] == 2:
                    if i + int(2 * c[k, 0]) < 0 or i + int(2 * c[k, 0]) >= Nx:
                        i_nb = (i_nb + int(2 * c[k, 0]) + Nx) % Nx
                    else:
                        i_nb = (i_nb + Nx) % Nx
                    if j + int(2 * c[k, 1]) < 0 or j + int(2 * c[k, 1]) >= Ny:
                        j_nb = (j_nb + int(2 * c[k, 1]) + Ny) % Ny
                    else:
                        j_nb = (j_nb + Ny) % Ny
                elif size > 1 and boundaryNode[ind] == 2:
                    if (i + int(2 * c[k, 0]) == 0 or
                            i + int(2 * c[k, 0]) == Nx - 1):
                        i_nb = i_nb + int(c[k, 0])
                    if (j + int(2 * c[k, 1]) == 0 or
                            j + int(2 * c[k, 1]) == Ny - 1):
                        j_nb = j_nb + int(c[k, 1])
                ind_nb = int(i_nb * Ny + j_nb)
                if solid[ind_nb, 0] == 1:
                    # if i == 51 and j == 58:
                    #     print(i_nb, j_nb, deltaM[ind_nb])
                    phi[ind] += deltaM[ind_nb]
                    # deltaMAdded += deltaM[ind_nb]
    # return deltaMAdded


def copyPhi(Nx, Ny, phi, phi_old):
    for ind in prange(Nx * Ny):
        phi_old[ind] = phi[ind]


def fakhariWetting(solidNodes, surfaceNormals, phi, phi_temp, normalPhi,
                   boundaryNode, contactAngle, interfaceWidth, Nx, Ny, size,
                   hysteresis=False, contactAngleReceding=0,
                   contactAngleAdvancing=0, initial=False):
    # contactAngleLocalStore = np.zeros(solidNodes.shape[0])
    # cosThetaStore = np.zeros(solidNodes.shape[0])
    for itr in prange(solidNodes.shape[0]):
        epsilon = - 2 * np.cos(contactAngle) / interfaceWidth
        epsilon_1 = 1./(epsilon + 1e-17)
        ind = solidNodes[itr]
        i, j = int(ind / Ny), int(ind % Ny)
        i_nb = i + surfaceNormals[itr, 0]
        j_nb = j + surfaceNormals[itr, 1]
        i_next, j_next = int(np.ceil(i_nb)), int(np.ceil(j_nb))
        i_prev, j_prev = int(np.floor(i_nb)), int(np.floor(j_nb))
        if i_next == i_prev and j_next == j_prev:
            deltaX, deltaY = i_next - i, j_next - j
        else:
            deltaX, deltaY = i_next - i_prev, j_next - j_prev
        i_next_phi, j_next_phi = i_next, j_next
        i_prev_phi, j_prev_phi = i_prev, j_prev
        if size == 1 and boundaryNode[ind] == 2:
            if i_next == Nx - 1:
                i_next_phi = (i_next + 2 + Nx) % Nx
            if j_next == Ny - 1:
                j_next_phi = (j_next + 2 + Ny) % Ny
            if i_prev == 0:
                i_prev_phi = (i_prev - 2 + Nx) % Nx
            if j_prev == 0:
                j_prev_phi = (j_prev - 2 + Ny) % Ny
        elif size > 1 and boundaryNode[ind] == 2:
            if i_next == Nx - 2:
                i_next_phi = i_next + 1
            if j_next == Ny - 2:
                j_next_phi = j_next + 1
            if i_prev == 1:
                i_prev_phi = i_prev - 1
            if j_prev == 1:
                j_prev_phi = j_prev - 1
        updateContactAngle = True
        solidNormalDotNormalPhi = 0
        if hysteresis is True and initial is False:
            if deltaX == 0:
                if surfaceNormals[itr, 1] < 0:
                    normalPhi_fluid_x = \
                        normalPhi[int(i_prev_phi * Ny + j_prev_phi), 0]
                    normalPhi_fluid_y = \
                        normalPhi[int(i_prev_phi * Ny + j_prev_phi), 1]
                elif surfaceNormals[itr, 1] > 0:
                    normalPhi_fluid_x = \
                        normalPhi[int(i_prev_phi * Ny + j_next_phi), 0]
                    normalPhi_fluid_y = \
                        normalPhi[int(i_prev_phi * Ny + j_next_phi), 1]
            elif deltaY == 0:
                if surfaceNormals[itr, 0] < 0:
                    normalPhi_fluid_x = \
                        normalPhi[int(i_prev_phi * Ny + j_prev_phi), 0]
                    normalPhi_fluid_y = \
                        normalPhi[int(i_prev_phi * Ny + j_prev_phi), 1]
                elif surfaceNormals[itr, 0] > 0:
                    normalPhi_fluid_x = \
                        normalPhi[int(i_next_phi * Ny + j_prev_phi), 0]
                    normalPhi_fluid_y = \
                        normalPhi[int(i_next_phi * Ny + j_prev_phi), 1]
            elif deltaX != 0 and deltaY != 0:
                normalPhi_fluid_x =\
                    ((i_next - i_nb) * (j_next - j_nb) *
                     normalPhi[i_prev_phi * Ny + j_prev_phi, 0] +
                     (i_next - i_nb) * (j_nb - j_prev) *
                     normalPhi[i_prev_phi * Ny + j_next_phi, 0] +
                     (i_nb - i_prev) * (j_next - j_nb) *
                     normalPhi[i_next_phi * Ny + j_prev_phi, 0] +
                     (i_nb - i_prev) * (j_nb - j_prev) *
                     normalPhi[i_next_phi * Ny + j_next_phi, 0])
                normalPhi_fluid_y =\
                    ((i_next - i_nb) * (j_next - j_nb) *
                     normalPhi[i_prev_phi * Ny + j_prev_phi, 1] +
                     (i_next - i_nb) * (j_nb - j_prev) *
                     normalPhi[i_prev_phi * Ny + j_next_phi, 1] +
                     (i_nb - i_prev) * (j_next - j_nb) *
                     normalPhi[i_next_phi * Ny + j_prev_phi, 1] +
                     (i_nb - i_prev) * (j_nb - j_prev) *
                     normalPhi[i_next_phi * Ny + j_next_phi, 1])
                normalPhi_fluid_x = normalPhi_fluid_x / (deltaX * deltaY)
                normalPhi_fluid_y = normalPhi_fluid_y / (deltaX * deltaY)
            solidNormalDotNormalPhi = \
                surfaceNormals[itr, 0] * normalPhi_fluid_x +\
                surfaceNormals[itr, 1] * normalPhi_fluid_y
            # print(i, j, np.arccos(-solidNormalDotNormalPhi) * 180 / np.pi)
            # cosThetaStore[itr] = -solidNormalDotNormalPhi
            contactAngleLocal = np.arccos(-solidNormalDotNormalPhi)
            # contactAngleLocalStore[itr] = contactAngleLocal
            if contactAngleLocal < contactAngleReceding:
                epsilon = - 2 * np.cos(contactAngleReceding) / interfaceWidth
                epsilon_1 = 1./(epsilon + 1e-17)
            elif contactAngleLocal > contactAngleAdvancing:
                epsilon = - 2 * np.cos(contactAngleAdvancing) / interfaceWidth
                epsilon_1 = 1./(epsilon + 1e-17)
            else:
                epsilon = - 2 * np.cos(contactAngleLocal) / interfaceWidth
                epsilon_1 = 1./(epsilon + 1e-17)
                updateContactAngle = False
        if updateContactAngle is True:
            # print(hysteresis)
            if deltaX == 0:
                if surfaceNormals[itr, 1] < 0:
                    phi_fluid = phi_temp[int(i_prev_phi * Ny + j_prev_phi)]
                elif surfaceNormals[itr, 1] > 0:
                    phi_fluid = phi_temp[int(i_prev_phi * Ny + j_next_phi)]
            elif deltaY == 0:
                if surfaceNormals[itr, 0] < 0:
                    phi_fluid = phi_temp[int(i_prev_phi * Ny + j_prev_phi)]
                elif surfaceNormals[itr, 0] > 0:
                    phi_fluid = phi_temp[int(i_next_phi * Ny + j_prev_phi)]
            elif deltaX != 0 and deltaY != 0:
                phi_fluid = ((i_next - i_nb) * (j_next - j_nb) *
                             phi_temp[i_prev_phi * Ny + j_prev_phi] +
                             (i_next - i_nb) * (j_nb - j_prev) *
                             phi_temp[i_prev_phi * Ny + j_next_phi] +
                             (i_nb - i_prev) * (j_next - j_nb) *
                             phi_temp[i_next_phi * Ny + j_prev_phi] +
                             (i_nb - i_prev) * (j_nb - j_prev) *
                             phi_temp[i_next_phi * Ny + j_next_phi]) /\
                    (deltaX * deltaY)
            phi[ind] =\
                np.abs(epsilon_1 * (1 + epsilon - np.sqrt((1 + epsilon)
                       * (1 + epsilon) - 4 * epsilon * phi_fluid)) - phi_fluid)
    # return contactAngleLocalStore, cosThetaStore


def regularized(solidNodes, surfaceNormals, phi, phi_temp, normalPhi,
                boundaryNode, contactAngle, interfaceWidth, Nx, Ny, size,
                hysteresis=False, contactAngleReceding=0,
                contactAngleAdvancing=0, initial=False):
    # contactAngleLocalStore = np.zeros(solidNodes.shape[0])
    # cosThetaStore = np.zeros(solidNodes.shape[0])
    for itr in prange(solidNodes.shape[0]):
        epsilon = np.exp(- 4 * np.cos(contactAngle) / interfaceWidth)
        ind = solidNodes[itr]
        i, j = int(ind / Ny), int(ind % Ny)
        i_nb = i + surfaceNormals[itr, 0]
        j_nb = j + surfaceNormals[itr, 1]
        i_next, j_next = int(np.ceil(i_nb)), int(np.ceil(j_nb))
        i_prev, j_prev = int(np.floor(i_nb)), int(np.floor(j_nb))
        if i_next == i_prev and j_next == j_prev:
            deltaX, deltaY = i_next - i, j_next - j
        else:
            deltaX, deltaY = i_next - i_prev, j_next - j_prev
        i_next_phi, j_next_phi = i_next, j_next
        i_prev_phi, j_prev_phi = i_prev, j_prev
        if size == 1 and boundaryNode[ind] == 2:
            if i_next == Nx - 1:
                i_next_phi = (i_next + 2 + Nx) % Nx
            if j_next == Ny - 1:
                j_next_phi = (j_next + 2 + Ny) % Ny
            if i_prev == 0:
                i_prev_phi = (i_prev - 2 + Nx) % Nx
            if j_prev == 0:
                j_prev_phi = (j_prev - 2 + Ny) % Ny
        elif size > 1 and boundaryNode[ind] == 2:
            if i_next == Nx - 2:
                i_next_phi = i_next + 1
            if j_next == Ny - 2:
                j_next_phi = j_next + 1
            if i_prev == 1:
                i_prev_phi = i_prev - 1
            if j_prev == 1:
                j_prev_phi = j_prev - 1
        if deltaX == 0:
            if surfaceNormals[itr, 1] < 0:
                phi_fluid = phi_temp[int(i_prev_phi * Ny + j_prev_phi)]
            elif surfaceNormals[itr, 1] > 0:
                phi_fluid = phi_temp[int(i_prev_phi * Ny + j_next_phi)]
        elif deltaY == 0:
            if surfaceNormals[itr, 0] < 0:
                phi_fluid = phi_temp[int(i_prev_phi * Ny + j_prev_phi)]
            elif surfaceNormals[itr, 0] > 0:
                phi_fluid = phi_temp[int(i_next_phi * Ny + j_prev_phi)]
        elif deltaX != 0 and deltaY != 0:
            phi_fluid = ((i_next - i_nb) * (j_next - j_nb) *
                         phi_temp[i_prev_phi * Ny + j_prev_phi] +
                         (i_next - i_nb) * (j_nb - j_prev) *
                         phi_temp[i_prev_phi * Ny + j_next_phi] +
                         (i_nb - i_prev) * (j_next - j_nb) *
                         phi_temp[i_next_phi * Ny + j_prev_phi] +
                         (i_nb - i_prev) * (j_nb - j_prev) *
                         phi_temp[i_next_phi * Ny + j_next_phi]) /\
                (deltaX * deltaY)
        phi[ind] = np.abs(phi_fluid / (epsilon * (1.0 - phi_fluid)
                          + phi_fluid))
    # return contactAngleLocalStore, cosThetaStore


def liangWetting(solidNodes, surfaceNormals, phi, normalPhi, boundaryNode,
                 contactAngle, interfaceWidth, Nx, Ny, size,
                 hysteresis=False, contactAngleReceding=0,
                 contactAngleAdvancing=0, initial=False):
    for itr in range(solidNodes.shape[0]):
        ind = solidNodes[itr]
        i, j = int(ind / Ny), int(ind % Ny)
        if j == 20:
            ind_xTop_1 = int(i * Ny + j + 1)
            ind_xNext_1 = int((i + 1) * Ny + j + 1)
            ind_xPrev_1 = int((i - 1) * Ny + j + 1)
            ind_xNext_2 = int((i + 1) * Ny + j + 2)
            ind_xPrev_2 = int((i - 1) * Ny + j + 2)
            tangentDotGradPhi = \
                0.5 * (1.5 * (phi[ind_xNext_1] - phi[ind_xPrev_1])
                       - 0.5 * (phi[ind_xNext_2] - phi[ind_xPrev_2]))
            phi[ind] = phi[ind_xTop_1] + np.tan((np.pi/2 - contactAngle)) *\
                np.abs(tangentDotGradPhi)
        if j == 29:
            ind_xTop_1 = int(i * Ny + j - 1)
            ind_xNext_1 = int((i + 1) * Ny + j - 1)
            ind_xPrev_1 = int((i - 1) * Ny + j - 1)
            ind_xNext_2 = int((i + 1) * Ny + j - 2)
            ind_xPrev_2 = int((i - 1) * Ny + j - 2)
            tangentDotGradPhi = \
                0.5 * (1.5 * (phi[ind_xNext_1] - phi[ind_xPrev_1])
                       - 0.5 * (phi[ind_xNext_2] - phi[ind_xPrev_2]))
            phi[ind] = phi[ind_xTop_1] + np.tan((np.pi/2 - contactAngle)) *\
                np.abs(tangentDotGradPhi)


def computeMass(mass, phi, solid, boundaryNode, procBoundary, Nx, Ny):
    temp = 0.
    for ind in prange(Nx * Ny):
        if (solid[ind, 0] == 0 and boundaryNode[ind] != 1 and
                procBoundary[ind] == 0):
            temp += phi[ind]
    mass[0] = temp
    return mass


@numba.njit
def chemicalPotentialModel(phi, gradPhi, normalPhi, curvature, lapPhi, beta,
                           kappa, phi_g, phi_l):
    chemPotential = 4 * beta * (phi - phi_g) * (phi - phi_l)\
                * (phi - 0.5) - kappa * lapPhi
    surfaceTensionForce_x = chemPotential * gradPhi[0]
    surfaceTensionForce_y = chemPotential * gradPhi[1]
    return surfaceTensionForce_x, surfaceTensionForce_y


@numba.njit
def continuumModel(phi, gradPhi, normalPhi, curvature, lapPhi, sigma, c, w,
                   cs_2, noOfDirections, interfaceWidth):
    delta = 24. * phi * phi * (1. - phi) * (1. - phi) / interfaceWidth
    surfaceTensionForce_x = - sigma * curvature * normalPhi[0] * delta
    surfaceTensionForce_y = - sigma * curvature * normalPhi[1] * delta
    return surfaceTensionForce_x, surfaceTensionForce_y


# @profile
# def forceFluid(f_new, f_eq, forceField, rho, p, phi, solid, lapPhi, gradPhi,
#                normalPhi, stressTensor, procBoundary, boundaryNode, gravity,
#                preFactorFluid, forcingPreFactor, constructOperatorFunc,
#                collisionOperatorArgs, constructStressTensorFunc,
#                computeViscFunc, surfaceTensionFunc, surfaceTensionArgs, mu_l,
#                mu_g, rho_l, rho_g, phi_l, phi_g, cs, noOfDirections, w, c,
#                cs_2, Nx, Ny, sigma, size, curvature):
# @profile
def forceFluid(f_new, f_eq, forceField, rho, p, phi, solid, lapPhi, gradPhi,
               normalPhi, stressTensor, surfaceTensionForce, viscousCorrection,
               pressureCorrection, procBoundary, boundaryNode, gravity,
               preFactorFluid, forcingPreFactorFluid, M, M_1, S_q, S_epsilon,
               nu_bulk, collisionTypeFluidNo, surfaceTensionModelNo,
               viscInterpolationNo, mu_l, mu_g, rho_l, rho_g, phi_l, phi_g,
               sigma, interfaceWidth, cs, noOfDirections, w, c, cs_2, Nx, Ny,
               size, precision):
    for ind in prange(Nx * Ny):
        if (solid[ind, 0] != 1 and procBoundary[ind] != 1
                and boundaryNode[ind] != 1):
            if viscInterpolationNo == 1:          # Linear
                nu = (mu_g + (phi[ind] - phi_g) * (mu_l - mu_g))/rho[ind]
            elif viscInterpolationNo == 2:        # Harmonic
                nu = 1./((1. - phi[ind])/(mu_g/rho_g) + phi[ind]/(mu_l/rho_l))
            # nu = computeViscFunc(mu_l, mu_g, phi_g, phi[ind], rho[ind],
            #                      rho_l, rho_g)
            # constructOperatorFunc(preFactorFluid[ind], forcingPreFactor[ind],
            #                       nu, *collisionOperatorArgs)
            if collisionTypeFluidNo == 1:    # BGK
                preFactorValue = 1. / (nu * cs_2 + 0.5)
                for k in range(noOfDirections):
                    preFactorFluid[ind, k, k] = preFactorValue
                    forcingPreFactorFluid[ind, k, k] = 1 - 0.5 * preFactorValue
                for i in range(2):
                    for j in range(2):
                        stressSum = precision(0)
                        for k in range(noOfDirections):
                            stressSum += c[k, i] * c[k, j] *\
                                (f_new[ind, k] - f_eq[ind, k])
                        stressTensor[ind, i, j] = \
                            - nu * preFactorFluid[ind, 0, 0] * cs_2 * stressSum
            elif collisionTypeFluidNo == 2:    # MRT
                S_nu = 1. / (nu * cs_2 + 0.5)
                S_bulk = 1. / (cs_2 * (nu / 3 + nu_bulk) + 0.5)
                # S = np.array([0., S_bulk, S_epsilon, 0., S_q, 0., S_q,
                #              S_nu, S_nu], dtype=precision)
                S = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             S_nu, S_nu], dtype=precision)
                # S = np.array([S_nu, S_nu, S_nu, S_nu, S_nu, S_nu, S_nu,
                #               S_nu, S_nu], dtype=precision)
                # S_ = np.diag(S, k=0)
                # S_f = np.diag(1 - 0.5 * S, k=0)
                # preFactorFluid[ind] = np.dot(np.dot(M_1, S_), M)
                # forcingPreFactorFluid[ind] =\
                #     np.dot(np.dot(M_1, S_f), M)
                for k in range(9):
                    for m in range(9):
                        colSum = precision(0)
                        forceSum = precision(0)
                        for n in range(9):
                            colSum += M_1[k, n] * S[n] * M[n, m]
                            forceSum += M_1[k, n] * (1 - 0.5 * S[n]) * M[n, m]
                        preFactorFluid[ind, k, m] = colSum
                        forcingPreFactorFluid[ind, k, m] = forceSum
                for i in range(2):
                    for j in range(2):
                        stressSum = precision(0)
                        for k in range(noOfDirections):
                            temp = precision(0)
                            for m in range(noOfDirections):
                                temp += preFactorFluid[ind, k, m] *\
                                    (f_new[ind, m] - f_eq[ind, m])
                            stressSum += c[k, i] * c[k, j] * temp
                        stressTensor[ind, i, j] = - nu * cs_2 * stressSum
            # constructStressTensorFunc(f_new[ind], f_eq[ind], stressTensor[ind],
            #                           preFactorFluid[ind], noOfDirections, c,
            #                           cs_2, nu)
            if surfaceTensionModelNo == 1:      # Chemical Potential
                chemPotential = \
                    sigma * ((48 / interfaceWidth) * (phi[ind] - phi_g) *
                             (phi[ind] - phi_l) * (phi[ind] - 0.5) - 1.5 *
                             interfaceWidth * lapPhi[ind])
                surfaceTensionForce_x = chemPotential * gradPhi[ind, 0]
                surfaceTensionForce_y = chemPotential * gradPhi[ind, 1]
            elif surfaceTensionModelNo == 2:      # Continuum
                curvature = 0
                denominator = 0
                i, j = int(ind / Ny), int(ind % Ny)
                for k in range(noOfDirections):
                    i_nb = i + int(c[k, 0])
                    j_nb = j + int(c[k, 1])
                    if size == 1 and boundaryNode[ind] == 2:
                        if i + int(2 * c[k, 0]) < 0 or i + int(2 * c[k, 0]) >= Nx:
                            i_nb = (i_nb + int(2 * c[k, 0]) + Nx) % Nx
                        else:
                            i_nb = (i_nb + Nx) % Nx
                        if j + int(2 * c[k, 1]) < 0 or j + int(2 * c[k, 1]) >= Ny:
                            j_nb = (j_nb + int(2 * c[k, 1]) + Ny) % Ny
                        else:
                            j_nb = (j_nb + Ny) % Ny
                    elif size > 1 and boundaryNode[ind] == 2:
                        if (i + int(2 * c[k, 0]) == 0 or
                                i + int(2 * c[k, 0]) == Nx - 1):
                            i_nb = i_nb + int(c[k, 0])
                        if (j + int(2 * c[k, 1]) == 0 or
                                j + int(2 * c[k, 1]) == Ny - 1):
                            j_nb = j_nb + int(c[k, 1])
                    ind_nb = int(i_nb * Ny + j_nb)
                    if solid[ind_nb, 0] == 0:
                        curvature +=\
                             w[k] * (c[k, 0] * normalPhi[ind_nb, 0] +
                                     c[k, 1] * normalPhi[ind_nb, 1])
                        denominator += w[k]
                curvature = cs_2 * curvature / (denominator + 1e-17)
                delta = 24. * phi[ind] * phi[ind] * (1. - phi[ind]) *\
                    (1. - phi[ind]) / interfaceWidth
                surfaceTensionForce_x = - sigma * curvature *\
                    normalPhi[ind, 0] * delta
                surfaceTensionForce_y = - sigma * curvature *\
                    normalPhi[ind, 1] * delta
            # surfaceTensionForce_x, surfaceTensionForce_y
            #     surfaceTensionFunc(phi[ind], gradPhi[ind], normalPhi[ind],
            #                         curvature[ind], lapPhi[ind],
            #                         *surfaceTensionArgs)
            pressureCorrection_x = - p[ind] * cs * cs * (rho_l - rho_g) *\
                gradPhi[ind, 0]
            pressureCorrection_y = - p[ind] * cs * cs * (rho_l - rho_g) *\
                gradPhi[ind, 1]
            viscousCorrection_x = (stressTensor[ind, 0, 0] * gradPhi[ind, 0] +
                                   stressTensor[ind, 0, 1] * gradPhi[ind, 1])\
                * (rho_l - rho_g)
            viscousCorrection_y = (stressTensor[ind, 1, 0] * gradPhi[ind, 0] +
                                   stressTensor[ind, 1, 1] * gradPhi[ind, 1])\
                * (rho_l - rho_g)
            forceField[ind, 0] = surfaceTensionForce_x + pressureCorrection_x \
                + rho[ind] * gravity[0] + viscousCorrection_x
            forceField[ind, 1] = surfaceTensionForce_y + pressureCorrection_y \
                + rho[ind] * gravity[1] + viscousCorrection_y
            surfaceTensionForce[ind, 0] = surfaceTensionForce_x
            surfaceTensionForce[ind, 1] = surfaceTensionForce_y
            viscousCorrection[ind, 0] = viscousCorrection_x
            viscousCorrection[ind, 1] = viscousCorrection_y
            pressureCorrection[ind, 0] = pressureCorrection_x
            pressureCorrection[ind, 1] = pressureCorrection_y


@numba.njit
def stressTensorBGK(f_new, f_eq, stressTensor, preFactor, noOfDirections,
                    c, cs_2, nu):
    for i in range(2):
        for j in range(2):
            stressSum = 0
            for k in range(noOfDirections):
                stressSum += c[k, i] * c[k, j] * (f_new[k] - f_eq[k])
            stressTensor[i, j] = - nu * preFactor[0, 0] * cs_2 * stressSum


@numba.njit
def stressTensorMRT(f_new, f_eq, stressTensor, preFactor, noOfDirections,
                    c, cs_2, nu):
    for i in range(2):
        for j in range(2):
            stressSum = 0
            for k in range(noOfDirections):
                temp = 0
                for m in range(noOfDirections):
                    temp += preFactor[k, m] * (f_new[m] - f_eq[m])
                stressSum += c[k, i] * c[k, j] * temp
            stressTensor[i, j] = - nu * cs_2 * stressSum
