import os
import numpy as np
import numba
from numba import prange
# from line_profiler import profile
# from scalene import profile
# from memory_profiler import profile


class phaseFieldDef:
    def __init__(self, phaseDict, lattice, transport, rank, precision):
        self.plate = False
        self.cylinder = False
        self.contactAngleLeft = np.zeros(2, dtype=precision)
        self.contactAngleRight = np.zeros(2, dtype=precision)
        self.interfaceWidthEffLeft = np.zeros(2, dtype=precision)
        self.interfaceWidthEffRight = np.zeros(2, dtype=precision)
        self.leftReceding = np.zeros(2, dtype=np.bool_)
        self.leftAdvancing = np.zeros(2, dtype=np.bool_)
        self.rightReceding = np.zeros(2, dtype=np.bool_)
        self.rightAdvancing = np.zeros(2, dtype=np.bool_)
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
            self.contactAngleHysteresis = False
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
                    self.solidPhaseFieldMobility = 0.16
                    self.plate = phaseDict["plate"]
                    self.cylinder = phaseDict["cylinder"]
                    self.sortedIndex = [np.zeros(100, dtype=np.int64)]
                    self.thetaSolid = [np.zeros(100, dtype=np.float64)]
                    self.phiThetaSolid = [np.zeros(100, dtype=np.float64)]
                    self.contactAngleThetaSolid = \
                        [np.zeros(100, dtype=np.float64)]
                    self.noOfSolidBoundary = [0]
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
            try:
                self.massCorrection = phaseDict['massCorrection']
            except KeyError:
                self.massCorrection = False
            self.massCorrectionInterval = 1
            if self.massCorrection is True:
                try:
                    self.massCorrectionInterval =\
                        phaseDict['massCorrectionInterval']
                except KeyError:
                    pass
            self.mass = np.zeros(1, dtype=precision)
            self.diffusionTime = int((self.interfaceWidth * self.
                                     interfaceWidth) / self.M)
            self.kappa = 1.5 * transport.sigma * self.interfaceWidth
            self.beta = 12 * transport.sigma / self.interfaceWidth
            self.segregation = False
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
        self.computeMass =  \
            numba.njit(computeMass, parallel=parallel,
                       cache=False, nogil=True)
        if self.contactAngle is not None:
            # self.wettingFunction =  \
            #     numba.njit(self.wettingFunction, parallel=parallel,
            #                cache=False, nogil=True)
            self.copyPhi = \
                numba.njit(copyPhi, parallel=parallel,
                           cache=False, nogil=True)
        self.initExtraMass =  \
            numba.njit(initExtraMass, parallel=parallel,
                       cache=False, nogil=True)
        self.setExtraMass =  \
            numba.njit(setExtraMass, parallel=parallel,
                       cache=False, nogil=True)
        self.computeSolidPhiBoundary =  \
            numba.njit(obtainSolidPhiData, parallel=parallel,
                       cache=False, nogil=True)
        self.gradPhiSolidBoundary =  \
            numba.njit(gradPhiSolidBoundary, parallel=parallel,
                       cache=False, nogil=True)
        self.gradPhiFluidBoundary =  \
            numba.njit(computeGradPhiFluidBoundary, parallel=parallel,
                       cache=False, nogil=True)
        self.updateGhostHysteresis = \
            numba.njit(updateGhostHysteresis, parallel=parallel,
                       cache=False, nogil=True)
        self.updateGhostHysteresisCircle = \
            numba.njit(updateGhostHysteresisCircle, parallel=parallel,
                       cache=False, nogil=True)
        self.updatePhiThetaSolid = \
            numba.njit(updatePhiThetaSolid, parallel=parallel,
                       cache=False, nogil=True)
        self.setContactAngleFunc = \
            numba.njit(setContactAngle, parallel=parallel,
                       cache=False, nogil=True)
        self.checkPinningPlate = \
            numba.njit(checkPinningPlate, parallel=parallel,
                       cache=False, nogil=True)
        self.checkPinningCircle = \
            numba.njit(checkPinningCircle, parallel=parallel,
                       cache=False, nogil=True)
        self.solidPhaseFieldPlate = \
            numba.njit(solidPhaseFieldHysteresisPlate, parallel=parallel,
                       cache=False, nogil=True)
        self.solidPhaseFieldCircle = \
            numba.njit(solidPhaseFieldHysteresisCircle, parallel=parallel,
                       cache=False, nogil=True)
        self.valuesNewGhostNodes = \
            numba.njit(setValuesNewGhostNodes, parallel=parallel,
                       cache=False, nogil=True)
        # self.updatePhiThetaSolid = updatePhiThetaSolid
        # self.updateGhostHysteresisCircle = updateGhostHysteresisCircle

    def ghostNodeHysteresis(self, options, fields, obstacle, mesh,
                            pointLeft, pointRight, center, pointLeft_0,
                            pointRight_0, center_0, timeStep,
                            cylinder=False, plate=True, smoothening=False):
        for itr in range(options.noOfSurfaces):
            if plate is True and options.obstacleFlag[itr] == 1 and itr == 0:
                self.updateGhostHysteresis(fields.solidBoundary,
                                           fields.phi, fields.phiAdvect,
                                           obstacle.obsU,
                                           mesh.Nx, mesh.Ny, pointLeft,
                                           pointRight, center, pointLeft_0,
                                           pointRight_0, center_0,
                                           self.interfaceWidth,
                                           self.interfaceWidthEffLeft,
                                           self.interfaceWidthEffRight,
                                           timeStep, fields.deltaFuncSolid,
                                           fields.contactAngleLocalSolid,
                                           self.contactAngleReceding,
                                           self.contactAngleAdvancing,
                                           self.contactAngleLeft,
                                           self.contactAngleRight,
                                           smoothening=smoothening)
            if cylinder is True and options.obstacleFlag[itr] == 1:
                self.updateGhostHysteresisCircle(fields.solidBoundary,
                                                 fields.phi, fields.
                                                 deltaFuncSolid,
                                                 fields.contactAngleLocalSolid,
                                                 fields.phiAdvect,
                                                 obstacle.obsOmega[itr],
                                                 self.contactAngleAdvancing,
                                                 self.contactAngleReceding,
                                                 mesh.Nx, mesh.Ny,
                                                 self.thetaSolid,
                                                 self.phiThetaSolid[0],
                                                 pointLeft, pointRight, center,
                                                 pointLeft_0, pointRight_0,
                                                 center_0, self.interfaceWidth,
                                                 obstacle.obsOrigin[itr],
                                                 obstacle.radius[itr],
                                                 self.interfaceWidthEffLeft,
                                                 self.interfaceWidthEffRight,
                                                 self.sortedIndex[0],
                                                 self.noOfSolidBoundary[0],
                                                 timeStep,
                                                 smoothening=smoothening)

    def checkPinning(self, fields, options, obstacle, mesh, center,
                     pointLeft, pointRight, plate=True, cylinder=False):
        # self.leftReceding = np.zeros(2, dtype=np.bool_)
        # self.leftAdvancing = np.zeros(2, dtype=np.bool_)
        # self.rightReceding = np.zeros(2, dtype=np.bool_)
        # self.rightAdvancing = np.zeros(2, dtype=np.bool_)
        for itr in range(options.noOfSurfaces):
            if plate is True and itr == 0:
                self.checkPinningPlate(fields.solidBoundary,
                                       fields.deltaFuncSolid,
                                       fields.contactAngleLocalSolid,
                                       self.leftReceding, self.rightReceding,
                                       self.leftAdvancing, self.rightAdvancing,
                                       self.contactAngleReceding,
                                       self.contactAngleAdvancing,
                                       center, mesh.Nx, mesh.Ny)
            elif cylinder is True and itr == 0:
                self.checkPinningCircle(fields.solidBoundary,
                                        fields.deltaFuncSolid,
                                        fields.contactAngleLocalSolid,
                                        self.leftReceding,
                                        self.rightReceding,
                                        self.leftAdvancing,
                                        self.rightAdvancing,
                                        self.contactAngleReceding,
                                        self.contactAngleAdvancing,
                                        self.interfaceWidth,
                                        self.interfaceWidthEffLeft,
                                        self.interfaceWidthEffRight,
                                        center, pointLeft, pointRight,
                                        obstacle.obsOrigin[itr],
                                        mesh.Nx, mesh.Ny)

    def setSolidPhaseFieldHysteresis(self, fields, options, obstacle, mesh,
                                     center, pointLeft, pointRight, size,
                                     plate=True, cylinder=False):
        for itr in range(options.noOfSurfaces):
            if plate is True and itr == 0:
                self.solidPhaseFieldPlate(fields.phi, fields.phiAdvect,
                                          fields.phiFSolidBoundary,
                                          fields.solidBoundary,
                                          self.interfaceWidth, center,
                                          self.leftReceding,
                                          self.rightReceding,
                                          self.leftAdvancing,
                                          self.rightAdvancing,
                                          self.contactAngleReceding,
                                          self.contactAngleAdvancing,
                                          self.wettingFunction, mesh.Nx,
                                          mesh.Ny)
            elif cylinder is True and itr == 0:
                self.solidPhaseFieldCircle(fields.phi, fields.phiAdvect,
                                           fields.phiFSolidBoundary,
                                           fields.solidBoundary,
                                           self.interfaceWidth, center,
                                           pointLeft, pointRight,
                                           self.leftReceding,
                                           self.rightReceding,
                                           self.leftAdvancing,
                                           self.rightAdvancing,
                                           self.contactAngleReceding,
                                           self.contactAngleAdvancing,
                                           self.wettingFunction,
                                           obstacle.obsOrigin[itr],
                                           mesh.Nx, mesh.Ny)
            elif options.obstacleFlag[itr] == 0:
                contactAngle = np.pi/2
                self.setContactAngleFunc(options.solidNbNodes[itr],
                                         fields.phi, fields.
                                         normalPhiDotNormalSolid,
                                         fields.phiFSolidBoundary,
                                         contactAngle,
                                         self.interfaceWidth, mesh.Nx, mesh.Ny,
                                         self.wettingFunction,
                                         size, hysteresis=False,
                                         contactAngleReceding=self.
                                         contactAngleReceding,
                                         contactAngleAdvancing=self.
                                         contactAngleAdvancing,
                                         initial=False)

    def recomputeSortedIndex(self, mesh, obstacle, fields):
        thetaSolid, sortedIndex, noOfSolidBoundary = \
            sortedIndexMovingBoundary(mesh.Nx, mesh.Ny, fields.solidBoundary,
                                      obstacle.obsOrigin[0])
        if self.thetaSolid[0].shape[0] != self.noOfSolidBoundary[0]:
            # print("I resized", self.phiThetaSolid[0].shape[0],
            #       self.noOfSolidBoundary[0])
            self.thetaSolid[0] = \
                np.resize(self.thetaSolid[0], self.noOfSolidBoundary[0])
            self.sortedIndex[0] = \
                np.resize(self.sortedIndex[0], self.noOfSolidBoundary[0])
        self.thetaSolid[0] = np.copy(thetaSolid)
        self.sortedIndex[0] = np.copy(sortedIndex)
        self.noOfSolidBoundary[0] = noOfSolidBoundary
        # print(self.thetaSolid[0].shape, self.sortedIndex[0].shape,
        #       self.noOfSolidBoundary[0])

    def computeAvgContactAngle(self, obstacle, fields, mesh, pointLeft,
                               pointRight, plate=False, cylinder=True):
        if plate is True:
            avgContactAngle(mesh.Nx, mesh.Ny, fields.solidBoundary,
                            fields.phi, fields.deltaFuncSolid,
                            fields.delPhiSolid, fields.contactAngleLocalSolid,
                            pointLeft, pointRight, self.contactAngleLeft,
                            self.contactAngleRight)
        elif cylinder is True:
            pass
            # interfacePositionUpdateCircle(self.thetaSolid[0], self.
            #                               phiThetaSolid[0],
            #                               pointLeft, pointRight,
            #                               self.noOfSolidBoundary[0],
            #                               obstacle.radius[0],
            #                               self.interfaceWidth)

    def updateInterfaceLocation(self, obstacle, pointLeft, pointRight, center,
                                mesh, fields, plate=False, cylinder=True):
        if plate is True:
            interfacePositionUpdate(mesh.Nx, mesh.Ny, fields.solidBoundary,
                                    fields.phi, self.interfaceWidth,
                                    pointLeft, pointRight, center,
                                    self.contactAngleLeft,
                                    self.contactAngleRight,
                                    self.contactAngleReceding,
                                    self.contactAngleAdvancing,
                                    fields.contactAngleLocalSolid,
                                    fields.deltaFuncSolid,
                                    self.interfaceWidthEffLeft,
                                    self.interfaceWidthEffRight)
        elif cylinder is True:
            interfacePositionUpdateCircle(self.thetaSolid[0], self.
                                          phiThetaSolid[0],
                                          pointLeft, pointRight,
                                          self.noOfSolidBoundary[0],
                                          obstacle.radius[0],
                                          self.interfaceWidth,
                                          self.interfaceWidthEffLeft,
                                          self.interfaceWidthEffRight,
                                          self.contactAngleLeft,
                                          self.contactAngleRight,
                                          self.contactAngleThetaSolid[0])
            self.interfaceWidthEffLeft[0] = self.interfaceWidth /\
                np.sin(self.contactAngleLeft[0])
            self.interfaceWidthEffRight[0] = self.interfaceWidth /\
                np.sin(self.contactAngleRight[0])
            self.interfaceWidthEffLeft[1] = self.interfaceWidth /\
                np.sin(self.contactAngleLeft[1])
            self.interfaceWidthEffRight[1] = self.interfaceWidth /\
                np.sin(self.contactAngleRight[1])

    # @profile
    def updatePhiTheta(self, fields, mesh):
        # phiThetaSolid = np.zeros(self.noOfSolidBoundary[0], dtype=np.float64)
        if self.phiThetaSolid[0].shape[0] != self.noOfSolidBoundary[0]:
            # print("I resized", self.phiThetaSolid[0].shape[0],
            #       self.noOfSolidBoundary[0])
            self.phiThetaSolid[0] = \
                np.resize(self.phiThetaSolid[0], self.noOfSolidBoundary[0])
            self.contactAngleThetaSolid[0] = \
                np.resize(self.contactAngleThetaSolid[0],
                          self.noOfSolidBoundary[0])
        # print(self.phiThetaSolid.shape)
        self.updatePhiThetaSolid(fields.solidBoundary, fields.phi,
                                 fields.contactAngleLocalSolid,
                                 self.phiThetaSolid[0], self.sortedIndex[0],
                                 self.contactAngleThetaSolid[0],
                                 mesh.Nx, mesh.Ny)
        # print("updatePhiTheta done")
        # print(len(self.phiThetaSolid), self.phiThetaSolid[0].shape)
        # print(self.phiThetaSolid.shape)
        # self.phiThetaSolid[0] = phiThetaSolid.copy()
        # print("phiTheta copy done")

    def computePhiSolid(self, options, fields, mesh, size, timeStep,
                        saveInterval, initial=False):
        if not os.path.isdir("output"):
            os.makedirs("output")
        # self.copyPhi(mesh.Nx, mesh.Ny, fields.phi, fields.phi_temp)
        fields.deltaFuncSolid = np.zeros((mesh.Nx * mesh.Ny),
                                         dtype=np.float64)
        fields.delPhiSolid = np.zeros((mesh.Nx * mesh.Ny),
                                      dtype=np.float64)
        fields.contactAngleLocalSolid = np.zeros((mesh.Nx * mesh.Ny),
                                                 dtype=np.float64)
        for itr in range(options.noOfSurfaces):
            self.computeSolidPhiBoundary(options.solidNbNodes[itr],
                                         fields.phiFSolidBoundary,
                                         fields.normalPhiDotNormalSolid,
                                         fields.boundaryNode,
                                         fields.procBoundary,
                                         options.surfaceNormals[itr],
                                         fields.phi, fields.phi_temp,
                                         fields.normalPhi,
                                         fields.normalPhi_temp,
                                         fields.gradPhi, fields.gradPhi_temp,
                                         mesh.Nx, mesh.Ny, size,
                                         fields.contactAngleLocalSolid,
                                         fields.delPhiSolid,
                                         fields.deltaFuncSolid, fields.
                                         solidBoundary)
        # if timeStep % saveInterval == 0 and initial is False:
        # timeStepCond = timeStep >= 9995 and timeStep <= 10005
        # if timeStep % saveInterval == 0 and initial is False:
        #     np.savez("output/solidPhiData_t_" + str(timeStep) + ".npz",
        #              solidBoundary=fields.solidBoundary,
        #              fluidBoundary=fields.fluidBoundary,
        #              phi=fields.phi, solid=fields.solid,
        #              normalPhi=fields.normalPhi,
        #              gradPhi=fields.gradPhi,
        #              gradPhiFSolid=gradPhiFSolid,
        #              normalPhiDotNormalSolid=fields.
        #              normalPhiDotNormalSolid,
        #              phiFSolid=fields.phiFSolidBoundary,
        #              solidNbNodes=options.solidNbNodes[0],
        #              surfaceNormals=options.surfaceNormals[0],
        #              h_new=fields.h_new)

    def setSolidPhaseField(self, options, fields, boundary, lattice, mesh,
                           size, timeStep, saveInterval, initial=False):
        # if not os.path.isdir("output"):
        #     os.makedirs("output")
        # self.copyPhi(mesh.Nx, mesh.Ny, fields.phi, fields.phi_temp)
        for itr in range(options.noOfSurfaces):
            # self.setSolidPhi(options.solidNbNodes[itr], fields.phi,
            #                  fields.solid, fields.boundaryNode,
            #                  fields.procBoundary, lattice.w, lattice.c,
            #                  lattice.noOfDirections, mesh.Nx, mesh.Ny,
            #                  size)
            # self.liangWetting(options.solidNbNodes[itr], fields.phi,
            #                   self.contactAngle, mesh.Nx, mesh.Ny)
            if options.obstacleFlag[itr] == 0:
                contactAngle = np.pi / 2
                hysteresis = False
            else:
                contactAngle = self.contactAngle
                hysteresis = self.contactAngleHysteresis
            # if itr == 2:
            #     print(options.surfaceNormals[itr])
            if options.boundaryType[itr] != "fixedValue":
                self.setContactAngleFunc(options.solidNbNodes[itr],
                                         fields.phi, fields.
                                         normalPhiDotNormalSolid,
                                         fields.phiFSolidBoundary,
                                         contactAngle,
                                         self.interfaceWidth, mesh.Nx, mesh.Ny,
                                         self.wettingFunction,
                                         size, hysteresis=hysteresis,
                                         contactAngleReceding=self.
                                         contactAngleReceding,
                                         contactAngleAdvancing=self.
                                         contactAngleAdvancing,
                                         initial=initial)
                # contactAngleLocalStore, cosThetaStore =\
                #     self.wettingFunction(options.solidNbNodes[itr], options.
                #                          surfaceNormals[itr], fields.phi,
                #                          fields.phi_temp, fields.normalPhi,
                #                          fields.boundaryNode, self.contactAngle,
                #                          self.interfaceWidth, mesh.Nx, mesh.Ny,
                #                          size, hysteresis=self.
                #                          contactAngleHysteresis,
                #                          contactAngleReceding=self.
                #                          contactAngleReceding,
                #                          contactAngleAdvancing=self.
                #                          contactAngleAdvancing,
                #                          initial=initial)
            # if (timeStep == 9999 or timeStep == 10000 or timeStep == 19999 or
            #         timeStep == 20000 or timeStep == 0 and initial is False):
            # if timeStep >= 19995 and timeStep <= 20005 and initial is False:
            # if timeStep % saveInterval == 0 and initial is False:
            #     np.savez("output/localTheta_surface_" + str(itr) + "_"
            #              + str(timeStep) + ".npz",
            #              theta=contactAngleLocalStore,
            #              cosTheta=cosThetaStore,
            #              solidNodes=options.solidNbNodes[itr],
            #              phi=fields.phi, solid=fields.solid,
            #              normalPhi=fields.normalPhi,
            #              surfaceNodes=options.surfaceNodes[itr],
            #              normalSolid=options.surfaceNormals[itr],
            #              normalFluid=options.surfaceNormalsFluid[itr])

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
    # constant = np.abs(1 - 4 * (phi - 0.5) * (phi - 0.5)) / interfaceWidth
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


def gradPhiSolidBoundary(Nx, Ny, phi, gradPhi, normalPhi, solidBoundary,
                         procBoundary, boundaryNode, cs_2, c, w,
                         noOfDirections, size):
    for ind in prange(Nx * Ny):
        if (solidBoundary[ind] == 1 and procBoundary[ind] != 1 and
                boundaryNode[ind] != 1):
            i, j = int(ind / Ny), int(ind % Ny)
            gradPhiSum_x, gradPhiSum_y = 0., 0.
            denominator = 0.
            for k in range(1, noOfDirections):
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
                if solidBoundary[ind_nb] == 1:
                    gradPhiSum_x += c[k, 0] * w[k] * phi[ind_nb]
                    gradPhiSum_y += c[k, 1] * w[k] * phi[ind_nb]
                    denominator += w[k]
            gradPhi[ind, 0] = cs_2 * gradPhiSum_x / (denominator + 1e-17)
            gradPhi[ind, 1] = cs_2 * gradPhiSum_y / (denominator + 1e-17)
            magGradPhi = np.sqrt(gradPhi[ind, 0] * gradPhi[ind, 0] +
                                 gradPhi[ind, 1] * gradPhi[ind, 1])
            normalPhi[ind, 0] = gradPhi[ind, 0] / (magGradPhi + 1e-17)
            normalPhi[ind, 1] = gradPhi[ind, 1] / (magGradPhi + 1e-17)


def computeGradPhiFluidBoundary(Nx, Ny, phi, gradPhi, normalPhi, solid,
                                fluidBoundary, procBoundary, boundaryNode,
                                cs_2, c, w, noOfDirections, size,
                                initial=False):
    for ind in prange(Nx * Ny):
        if (fluidBoundary[ind] == 1 and procBoundary[ind] != 1 and
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
                if boundaryNode[ind_nb] != 1 or solid[ind_nb, 0] == 1:
                    gradPhiSum_x += c[k, 0] * w[k] * phi[ind_nb]
                    gradPhiSum_y += c[k, 1] * w[k] * phi[ind_nb]
                    lapPhiSum += w[k] * (phi[ind_nb] - phi[ind])
                    denominator += w[k]
                else:
                    noOfBoundaryNodes += 1
            if noOfBoundaryNodes == 0:
                gradPhi[ind, 0] = cs_2 * gradPhiSum_x
                gradPhi[ind, 1] = cs_2 * gradPhiSum_y
            else:
                gradPhi[ind, 0] = cs_2 * gradPhiSum_x / (denominator + 1e-17)
                gradPhi[ind, 1] = cs_2 * gradPhiSum_y / (denominator + 1e-17)
            magGradPhi = np.sqrt(gradPhi[ind, 0] * gradPhi[ind, 0] +
                                 gradPhi[ind, 1] * gradPhi[ind, 1])
            # print(magGradPhi)
            normalPhi[ind, 0] = gradPhi[ind, 0] / (magGradPhi + 1e-17)
            normalPhi[ind, 1] = gradPhi[ind, 1] / (magGradPhi + 1e-17)


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
                if boundaryNode[ind_nb] != 1 or solid[ind_nb, 0] == 1:
                    # if boundaryNode[ind_nb] != 1 and solid[ind_nb, 0] == 0:
                    gradPhiSum_x += c[k, 0] * w[k] * phi[ind_nb]
                    gradPhiSum_y += c[k, 1] * w[k] * phi[ind_nb]
                    lapPhiSum += w[k] * (phi[ind_nb] - phi[ind])
                    denominator += w[k]
                else:
                    noOfBoundaryNodes += 1
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


def copyPhi(Nx, Ny, phi, phi_old, normalPhi, gradPhi,
            normalPhi_old, gradPhi_old, phiFSolidBoundary,
            phiAdvect):
    for ind in prange(Nx * Ny):
        phi_old[ind] = phi[ind]
        normalPhi_old[ind, 0] = normalPhi[ind, 0]
        normalPhi_old[ind, 1] = normalPhi[ind, 1]
        gradPhi_old[ind, 0] = gradPhi[ind, 0]
        gradPhi_old[ind, 1] = gradPhi[ind, 1]
        phiFSolidBoundary[ind] = 0
        phiAdvect[ind] = 0


def setValuesNewGhostNodes(normalPhi, phi, phi_old, solidBoundary,
                           solidBoundary_old, solid, solid_old,
                           nodesIdentified, boundaryNode, c, w, noOfDirections,
                           nx, ny, nProc_x, nProc_y, size, Nx, Ny):
    for ind in range(Nx * Ny):
        nodesIdentified[ind] = 0
        if (solidBoundary[ind] == 1 and solidBoundary_old[ind] == 0
                and solid[ind, 0] == 1 and solid_old[ind, 0] == 1):
            nodesIdentified[ind] = 1
            phiSum = 0
            normalPhiSum_x, normalPhiSum_y = 0, 0
            denominator = 0
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
                if solidBoundary_old[ind_nb] == 1:
                    phiSum += w[k] * phi_old[ind_nb]
                    normalPhiSum_x += w[k] * normalPhi[ind_nb, 0]
                    normalPhiSum_y += w[k] * normalPhi[ind_nb, 1]
                    denominator += w[k]
            phi[ind] = phiSum / (denominator + 1e-17)
            normalPhi[ind, 0] = normalPhiSum_x / (denominator + 1e-17)
            normalPhi[ind, 1] = normalPhiSum_y / (denominator + 1e-17)


@numba.njit
def locateInterface(phi, x, start=0):
    for i in range(1, x.shape[0]):
        if start == 1:
            if phi[i] < 0.5:
                slope = (x[i] - x[i - 1])/(phi[i] - phi[i - 1])
                xInt = x[i] - slope * (phi[i] - 0.5)
                break
        elif start == 0:
            if phi[i] > 0.5:
                slope = (x[i] - x[i - 1])/(phi[i] - phi[i - 1])
                xInt = x[i] - slope * (phi[i] - 0.5)
                break
    return xInt, i


@numba.njit
def locateInterfaceTanh(phi, x, interfaceWidth, start=0):
    for i in range(1, x.shape[0]):
        if start == 1:
            if phi[i] < 0.5:
                # xInt = x[i] - 0.5 * interfaceWidth *\
                #     np.arctanh(1 - 2 * phi[i])
                xInt = x[i - 1] - 0.5 * interfaceWidth *\
                    np.arctanh(1 - 2 * phi[i - 1])
                break
        elif start == 0:
            if phi[i] > 0.5:
                xInt = x[i] - 0.5 * interfaceWidth *\
                    np.arctanh(2 * phi[i] - 1)
                break
    return xInt, i


@numba.njit
def locateInterfaceTanhCircle(phi, contactAngle, theta, radius,
                              interfaceWidth, start=0):
    for i in range(1, theta.shape[0]):
        if start == 1:
            if phi[i] < 0.5:
                # thetaInt = theta[i] - 0.5 * interfaceWidth *\
                #     np.arctanh(1 - 2 * phi[i]) / radius
                thetaInt = theta[i - 1] - 0.5 * interfaceWidth *\
                    np.arctanh(1 - 2 * phi[i - 1]) / radius
                slope = (contactAngle[i] - contactAngle[i - 1]) /\
                    (theta[i] - theta[i - 1])
                contactAngleInt = contactAngle[i] - slope *\
                    (theta[i] - thetaInt)
                break
        elif start == 0:
            if phi[i] > 0.5:
                thetaInt = theta[i] - 0.5 * interfaceWidth *\
                    np.arctanh(2 * phi[i] - 1) / radius
                slope = (contactAngle[i] - contactAngle[i - 1]) /\
                    (theta[i] - theta[i - 1])
                contactAngleInt = contactAngle[i] - slope *\
                    (theta[i] - thetaInt)
                break
    return thetaInt, contactAngleInt, i


# @numba.njit
# def locateInterfaceTanhCircle(phi, theta, radius, interfaceWidth, start=0):
#     for i in range(1, theta.shape[0]):
#         if start == 1:
#             if phi[i] < 0.5:
#                 # thetaInt = theta[i] - 0.5 * interfaceWidth *\
#                 #     np.arctanh(1 - 2 * phi[i]) / radius
#                 diffAhead = phi[i + 1] - phi[i]
#                 diffBehind = phi[i] - phi[i - 1]
#                 if diffAhead < 0 and diffBehind < 0:
#                     thetaInt = theta[i - 1] - 0.5 * interfaceWidth *\
#                         np.arctanh(1 - 2 * phi[i - 1]) / radius
#                     break
#         elif start == 0:
#             if phi[i] > 0.5:
#                 diffAhead = phi[i + 1] - phi[i]
#                 diffBehind = phi[i] - phi[i - 1]
#                 if diffAhead > 0 and diffBehind > 0:
#                     thetaInt = theta[i] - 0.5 * interfaceWidth *\
#                         np.arctanh(2 * phi[i] - 1) / radius
#                     break
#     return thetaInt, i


def replaceTanh(Nx, Ny, phi, solidBoundary, interfaceWidth,
                xl_0, xr_0, center_0):
    # phiLine = np.zeros(Nx, dtype=np.float64)
    phiLine = np.zeros((Nx, 2), dtype=np.float64)
    bottomPlate, topPlate = 0, 0
    flag = False
    for j in range(Ny):
        ind = Nx//2 * Ny + j
        if solidBoundary[ind] == 1 and flag is False:
            bottomPlate = j
            flag = True
        elif solidBoundary[ind] == 1 and flag is True:
            topPlate = j
    print(bottomPlate, topPlate)
    for ind in range(Nx * Ny):
        i, j = int(ind / Ny), int(ind % Ny)
        if j == topPlate:
            phiLine[i, 0] = phi[ind]
        if j == bottomPlate:
            phiLine[i, 1] = phi[ind]
        # if j == 61:  # j == 56:
        #     phiLine[i, 0] = phi[ind]
        # elif j == 40:  # j == 46:
        #     phiLine[i, 1] = phi[ind]
    x = np.linspace(0, Nx - 1, Nx)
    # xl_0[0], node = locateInterface(phiLine, x, start=0)
    # xr_0[0], node = locateInterface(phiLine[Nx//2:], x[Nx//2:], start=1)
    xl_0[0], node = locateInterface(phiLine[:, 0], x, start=0)
    xr_0[0], node = locateInterface(phiLine[Nx//2:, 0], x[Nx//2:], start=1)
    xl_0[1], node = locateInterface(phiLine[:, 1], x, start=0)
    xr_0[1], node = locateInterface(phiLine[Nx//2:, 1], x[Nx//2:], start=1)
    # xl_0[0], node = locateInterface(phiLine[:, 0], x, start=1)
    # xr_0[0], node = locateInterface(phiLine[:, 1], x, start=1)
    print(xl_0, xr_0)
    # print(xlBottom, xrBottom)
    for ind in range(Nx * Ny):
        if solidBoundary[ind] == 1:
            i, j = int(ind / Ny), int(ind % Ny)
            if j == topPlate and i < Nx/2:
                phi[ind] = 0.5 * (1 + np.tanh(2 * (i - xl_0[0]) /
                                  interfaceWidth))
            elif j == topPlate and i >= Nx/2:
                phi[ind] = 0.5 * (1 - np.tanh(2 * (i - xr_0[0]) /
                                  interfaceWidth))
            elif j == bottomPlate and i < Nx/2:
                phi[ind] = 0.5 * (1 + np.tanh(2 * (i - xl_0[1]) /
                                  interfaceWidth))
            elif j == bottomPlate and i >= Nx/2:
                phi[ind] = 0.5 * (1 - np.tanh(2 * (i - xr_0[1]) /
                                  interfaceWidth))
            else:
                phi[ind] = 0
            # if j == 61:  # j == 56:
            #     phi[ind] = 0.5 * (1 - np.tanh(2 * (i - xl_0[0]) /
            #                       interfaceWidth))
            # elif j == 40:  # j == 46:
            #     phi[ind] = 0.5 * (1 - np.tanh(2 * (i - xr_0[0]) /
            #                       interfaceWidth))
            # else:
            #     if i < Nx//2:
            #         phi[ind] = 1
            #     else:
            #         phi[ind] = 0
    center_0[0] = Nx // 2
    center_0[1] = Nx // 2
    return x, phiLine


def replaceTanhCircle(Nx, Ny, phi, solidBoundary, interfaceWidth,
                      thetaLeft_0, thetaRight_0, center_0, centerX,
                      centerY, noOfSolidBoundary, radius):
    print(centerX, centerY, radius)
    noOfSolidBoundary = int(np.sum(solidBoundary))
    polarCoordsData = np.zeros((noOfSolidBoundary, 4), dtype=np.float64)
    itr = 0
    for ind in range(Nx * Ny):
        if solidBoundary[ind] == 1:
            i, j = int(ind / Ny), int(ind % Ny)
            polarCoordsData[itr, 0] = \
                np.sqrt((i - centerX) * (i - centerX) +
                        (j - centerY) * (j - centerY))
            polarCoordsData[itr, 1] = np.arctan2(j - centerY,
                                                 i - centerX) + np.pi
            polarCoordsData[itr, 2] = phi[ind]
            polarCoordsData[itr, 3] = itr
            itr += 1
    polarCoordsDataSorted = polarCoordsData[polarCoordsData[:, 1].argsort()]
    phiTheta = polarCoordsDataSorted[:, 2]
    theta = polarCoordsDataSorted[:, 1]
    r = polarCoordsDataSorted[:, 0]
    indexMap = polarCoordsDataSorted[:, 3]
    sortedIndex = np.zeros(noOfSolidBoundary, dtype=np.int64)
    for itr in range(polarCoordsData.shape[0]):
        for indexNo in range(indexMap.shape[0]):
            if polarCoordsData[itr, 3] == indexMap[indexNo]:
                sortedIndex[itr] = indexNo
                break
    thetaInterpolate = np.zeros(3 * noOfSolidBoundary, dtype=np.float64)
    thetaInterpolate[:noOfSolidBoundary] = np.copy(theta)
    thetaInterpolate[noOfSolidBoundary:(2 * noOfSolidBoundary)] =\
        np.copy(theta) + 2 * np.pi
    thetaInterpolate[(2 * noOfSolidBoundary):] = np.copy(theta) + 4 * np.pi
    phiInterpolate = np.zeros(3 * theta.shape[0], dtype=np.float64)
    phiInterpolate[:noOfSolidBoundary] = np.copy(phiTheta)
    phiInterpolate[noOfSolidBoundary:(2 * noOfSolidBoundary)] =\
        np.copy(phiTheta)
    phiInterpolate[(2 * noOfSolidBoundary):] = np.copy(phiTheta)
    if phiInterpolate[noOfSolidBoundary] > 0.5:
        thetaIntRight, nodeRight = \
            locateInterface(phiInterpolate[noOfSolidBoundary:],
                            thetaInterpolate[noOfSolidBoundary:],
                            start=1)
        thetaIntLeft, nodeLeft = \
            locateInterface(phiInterpolate[int(noOfSolidBoundary +
                                           nodeRight + 20):],
                            thetaInterpolate[int(noOfSolidBoundary +
                                             nodeRight + 20):],
                            start=0)
    elif phiInterpolate[noOfSolidBoundary] <= 0.5:
        thetaIntLeft, nodeLeft = \
            locateInterface(phiInterpolate[noOfSolidBoundary:],
                            thetaInterpolate[noOfSolidBoundary:],
                            start=0)
        thetaIntRight, nodeRight = \
            locateInterface(phiInterpolate[int(noOfSolidBoundary +
                                           nodeLeft + 20):],
                            thetaInterpolate[int(noOfSolidBoundary +
                                             nodeLeft + 20):],
                            start=1)
    thetaLeft_0[0] = thetaIntLeft % (2 * np.pi)
    thetaRight_0[0] = thetaIntRight % (2 * np.pi)
    center_0[0] = (thetaLeft_0[0] + thetaRight_0[0]) / 2
    phiThetaSolid = np.zeros_like(phiTheta)
    print(thetaLeft_0, thetaRight_0, center_0)
    itr = 0
    for ind in range(Nx * Ny):
        if solidBoundary[ind] == 1:
            i, j = int(ind / Ny), int(ind % Ny)
            r_local = np.sqrt((i - centerX) * (i - centerX) +
                              (j - centerY) * (j - centerY))
            theta_local = np.arctan2(j - centerY, i - centerX) + np.pi
            distFromLeft, distFromRight = 1e10, 1e10
            distFromLeft_1 = theta_local - thetaLeft_0[0]
            distFromLeft_2 = theta_local - thetaLeft_0[0] -\
                2 * np.pi
            distFromLeft_3 = theta_local + 2 * np.pi -\
                thetaLeft_0[0]
            if np.abs(distFromLeft_1) < np.abs(distFromLeft):
                distFromLeft = distFromLeft_1
            if np.abs(distFromLeft_2) < np.abs(distFromLeft):
                distFromLeft = distFromLeft_2
            if np.abs(distFromLeft_3) < np.abs(distFromLeft):
                distFromLeft = distFromLeft_3
            distFromRight_1 = theta_local - thetaRight_0[0]
            distFromRight_2 = theta_local - thetaRight_0[0] -\
                2 * np.pi
            distFromRight_3 = theta_local + 2 * np.pi -\
                thetaRight_0[0]
            if np.abs(distFromRight_1) < np.abs(distFromRight):
                distFromRight = distFromRight_1
            if np.abs(distFromRight_2) < np.abs(distFromRight):
                distFromRight = distFromRight_2
            if np.abs(distFromRight_3) < np.abs(distFromRight):
                distFromRight = distFromRight_3
            if np.abs(distFromLeft) <= np.abs(distFromRight):
                phi[ind] = 0.5 * (1 + np.tanh(2 * radius *
                                  distFromLeft / interfaceWidth))
            elif np.abs(distFromLeft) > np.abs(distFromRight):
                phi[ind] = 0.5 * (1 - np.tanh(2 * radius *
                                  distFromRight / interfaceWidth))
            phiThetaSolid[int(sortedIndex[itr])] = phi[ind]
            itr += 1
    return r, theta, phiTheta, thetaInterpolate, phiInterpolate, \
        phiThetaSolid, sortedIndex, indexMap


@numba.njit
def sortedIndexMovingBoundary(Nx, Ny, solidBoundary, center):
    noOfSolidBoundary = int(np.sum(solidBoundary))
    polarCoordsData = np.zeros((noOfSolidBoundary, 2), dtype=np.float64)
    itr = 0
    for ind in range(Nx * Ny):
        if solidBoundary[ind] == 1:
            i, j = int(ind / Ny), int(ind % Ny)
            polarCoordsData[itr, 0] = np.arctan2(j - center[1],
                                                 i - center[0]) + np.pi
            polarCoordsData[itr, 1] = itr
            itr += 1
    polarCoordsDataSorted = polarCoordsData[polarCoordsData[:, 0].argsort()]
    thetaSolid = polarCoordsDataSorted[:, 0]
    indexMap = polarCoordsDataSorted[:, 1]
    sortedIndex = np.zeros(noOfSolidBoundary, dtype=np.int64)
    for itr in range(polarCoordsData.shape[0]):
        for indexNo in range(indexMap.shape[0]):
            if polarCoordsData[itr, 1] == indexMap[indexNo]:
                sortedIndex[itr] = indexNo
                break
    return thetaSolid, sortedIndex, noOfSolidBoundary


@numba.njit
def computeContactAngleInterface(Nx, Ny, xl, xr, contactAngleLocalSolid,
                                 topPlate, bottomPlate):
    xl_next = np.ceil(xl)
    xl_prev = np.floor(xl)
    xr_next = np.ceil(xr)
    xr_prev = np.floor(xr)
    # Top left contact angle
    ind_next = int(xl_next[0] * Ny + topPlate)
    ind_prev = int(xl_prev[0] * Ny + topPlate)
    slope = (contactAngleLocalSolid[ind_next] -
             contactAngleLocalSolid[ind_prev]) /\
        (xl_next[0] - xl_prev[0])
    contactAngleTopLeft = contactAngleLocalSolid[ind_prev] +\
        slope * (xl[0] - xl_prev[0])
    # Top right contact angle
    ind_next = int(xr_next[0] * Ny + topPlate)
    ind_prev = int(xr_prev[0] * Ny + topPlate)
    slope = (contactAngleLocalSolid[ind_next] -
             contactAngleLocalSolid[ind_prev]) /\
        (xr_next[0] - xr_prev[0])
    contactAngleTopRight = contactAngleLocalSolid[ind_prev] +\
        slope * (xr[0] - xr_prev[0])
    # Bottom left contact angle
    ind_next = int(xl_next[1] * Ny + topPlate)
    ind_prev = int(xl_prev[1] * Ny + topPlate)
    slope = (contactAngleLocalSolid[ind_next] -
             contactAngleLocalSolid[ind_prev]) /\
        (xl_next[1] - xl_prev[1])
    contactAngleBottomLeft = contactAngleLocalSolid[ind_prev] +\
        slope * (xl[1] - xl_prev[1])
    # Bottom right contact angle
    ind_next = int(xr_next[1] * Ny + topPlate)
    ind_prev = int(xr_prev[1] * Ny + topPlate)
    slope = (contactAngleLocalSolid[ind_next] -
             contactAngleLocalSolid[ind_prev]) /\
        (xr_next[1] - xr_prev[1])
    contactAngleBottomRight = contactAngleLocalSolid[ind_prev] +\
        slope * (xr[1] - xr_prev[1])
    return contactAngleTopLeft, contactAngleTopRight, \
        contactAngleBottomLeft, contactAngleBottomRight


def checkPinningPlate(solidBoundary, deltaFuncSolid, contactAngleLocalSolid,
                      leftReceding, rightReceding, leftAdvancing,
                      rightAdvancing, contactAngleReceding,
                      contactAngleAdvancing, center, Nx, Ny):
    bottomPlate, topPlate = 0, 0
    flag = False
    for j in range(Ny):
        ind = Nx//2 * Ny + j
        if solidBoundary[ind] == 1 and flag is False:
            bottomPlate = j
            flag = True
        elif solidBoundary[ind] == 1 and flag is True:
            topPlate = j
    for ind in range(Nx * Ny):
        if solidBoundary[ind] == 1:
            i, j = int(ind / Ny), int(ind % Ny)
            if j == topPlate and deltaFuncSolid[ind] >= 1e-1:
                if i < center[0]:
                    if contactAngleLocalSolid[ind] < contactAngleReceding:
                        leftReceding[0] = True
                        leftAdvancing[0] = False
                    elif contactAngleLocalSolid[ind] > contactAngleAdvancing:
                        leftReceding[0] = False
                        leftAdvancing[0] = True
                elif i >= center[0]:
                    if contactAngleLocalSolid[ind] < contactAngleReceding:
                        rightReceding[0] = True
                        rightAdvancing[0] = False
                    elif contactAngleLocalSolid[ind] > contactAngleAdvancing:
                        rightReceding[0] = False
                        rightAdvancing[0] = True
            elif j == bottomPlate and deltaFuncSolid[ind] >= 1e-1:
                if i < center[1]:
                    if contactAngleLocalSolid[ind] < contactAngleReceding:
                        leftReceding[1] = True
                        leftAdvancing[1] = False
                    elif contactAngleLocalSolid[ind] > contactAngleAdvancing:
                        leftReceding[1] = False
                        leftAdvancing[1] = True
                elif i >= center[1]:
                    if contactAngleLocalSolid[ind] < contactAngleReceding:
                        rightReceding[1] = True
                        rightAdvancing[1] = False
                    elif contactAngleLocalSolid[ind] > contactAngleAdvancing:
                        rightReceding[1] = False
                        rightAdvancing[1] = True


def checkPinningCircle(solidBoundary, deltaFuncSolid, contactAngleLocalSolid,
                       leftReceding, rightReceding, leftAdvancing,
                       rightAdvancing, contactAngleReceding,
                       contactAngleAdvancing, interfaceWidth,
                       interfaceWidthEffLeft, interfaceWidthEffRight, center,
                       thetaLeft, thetaRight, obsOrigin, Nx, Ny):
    for ind in range(Nx * Ny):
        if solidBoundary[ind] == 1 and deltaFuncSolid[ind] > 1e-1:
            i, j = int(ind / Ny), int(ind % Ny)
            theta_local = np.arctan2(j - obsOrigin[1], i - obsOrigin[0]) +\
                np.pi
            distFromLeft, distFromRight = 1e10, 1e10
            distFromLeft_1 = theta_local - thetaLeft[0]
            distFromLeft_2 = theta_local - thetaLeft[0] -\
                2 * np.pi
            distFromLeft_3 = theta_local + 2 * np.pi -\
                thetaLeft[0]
            if np.abs(distFromLeft_1) < np.abs(distFromLeft):
                distFromLeft = distFromLeft_1
            if np.abs(distFromLeft_2) < np.abs(distFromLeft):
                distFromLeft = distFromLeft_2
            if np.abs(distFromLeft_3) < np.abs(distFromLeft):
                distFromLeft = distFromLeft_3
            distFromRight_1 = theta_local - thetaRight[0]
            distFromRight_2 = theta_local - thetaRight[0] -\
                2 * np.pi
            distFromRight_3 = theta_local + 2 * np.pi -\
                thetaRight[0]
            if np.abs(distFromRight_1) < np.abs(distFromRight):
                distFromRight = distFromRight_1
            if np.abs(distFromRight_2) < np.abs(distFromRight):
                distFromRight = distFromRight_2
            if np.abs(distFromRight_3) < np.abs(distFromRight):
                distFromRight = distFromRight_3
            if np.abs(distFromLeft) <= np.abs(distFromRight):
                if contactAngleLocalSolid[ind] < contactAngleReceding:
                    leftReceding[0] = True
                    leftAdvancing[0] = False
                    # sinThetaLocal = np.sin(contactAngleReceding)
                    # interfaceWidthEffLeft[0] = interfaceWidth /\
                    #     sinThetaLocal
                elif (contactAngleLocalSolid[ind] >
                        contactAngleAdvancing):
                    leftReceding[0] = False
                    leftAdvancing[0] = True
                    # sinThetaLocal = np.sin(contactAngleAdvancing)
                    # interfaceWidthEffLeft[0] = interfaceWidth /\
                    #     sinThetaLocal
            elif np.abs(distFromLeft) > np.abs(distFromRight):
                if contactAngleLocalSolid[ind] < contactAngleReceding:
                    rightReceding[0] = True
                    rightAdvancing[0] = False
                    # sinThetaLocal = np.sin(contactAngleReceding)
                    # interfaceWidthEffRight[0] = interfaceWidth /\
                    #     sinThetaLocal
                elif (contactAngleLocalSolid[ind] >
                        contactAngleAdvancing):
                    rightReceding[0] = False
                    rightAdvancing[0] = True
                    # sinThetaLocal = np.sin(contactAngleAdvancing)
                    # interfaceWidthEffRight[0] = interfaceWidth /\
                    #     sinThetaLocal


def solidPhaseFieldHysteresisPlate(phi, phiAdvect, phiFSolidBoundary,
                                   solidBoundary, interfaceWidth, center,
                                   leftReceding, rightReceding, leftAdvancing,
                                   rightAdvancing, contactAngleReceding,
                                   contactAngleAdvancing, wettingFunc, Nx, Ny):
    bottomPlate, topPlate = 0, 0
    flag = False
    for j in range(Ny):
        ind = Nx//2 * Ny + j
        if solidBoundary[ind] == 1 and flag is False:
            bottomPlate = j
            flag = True
        elif solidBoundary[ind] == 1 and flag is True:
            topPlate = j
    for ind in prange(Nx * Ny):
        if solidBoundary[ind] == 1:
            i, j = int(ind / Ny), int(ind % Ny)
            if j == topPlate and i < center[0]:
                if leftReceding[0] is True and leftAdvancing[0] is False:
                    wettingFunc(ind, phi, phiFSolidBoundary, interfaceWidth,
                                contactAngleReceding)
                elif leftReceding[0] is False and leftAdvancing[0] is True:
                    wettingFunc(ind, phi, phiFSolidBoundary, interfaceWidth,
                                contactAngleAdvancing)
                elif leftReceding[0] is False and leftAdvancing[0] is False:
                    phi[ind] = phiAdvect[ind]
            elif j == topPlate and i >= center[0]:
                if rightReceding[0] is True and rightAdvancing[0] is False:
                    wettingFunc(ind, phi, phiFSolidBoundary, interfaceWidth,
                                contactAngleReceding)
                elif rightReceding[0] is False and rightAdvancing[0] is True:
                    wettingFunc(ind, phi, phiFSolidBoundary, interfaceWidth,
                                contactAngleAdvancing)
                elif rightReceding[0] is False and rightAdvancing[0] is False:
                    phi[ind] = phiAdvect[ind]
            elif j == bottomPlate and i < center[1]:
                if leftReceding[1] is True and leftAdvancing[1] is False:
                    wettingFunc(ind, phi, phiFSolidBoundary, interfaceWidth,
                                contactAngleReceding)
                elif leftReceding[1] is False and leftAdvancing[1] is True:
                    wettingFunc(ind, phi, phiFSolidBoundary, interfaceWidth,
                                contactAngleAdvancing)
                elif leftReceding[1] is False and leftAdvancing[1] is False:
                    phi[ind] = phiAdvect[ind]
            elif j == bottomPlate and i >= center[1]:
                if rightReceding[1] is True and rightAdvancing[1] is False:
                    wettingFunc(ind, phi, phiFSolidBoundary, interfaceWidth,
                                contactAngleReceding)
                elif rightReceding[1] is False and rightAdvancing[1] is True:
                    wettingFunc(ind, phi, phiFSolidBoundary, interfaceWidth,
                                contactAngleAdvancing)
                elif rightReceding[1] is False and rightAdvancing[1] is False:
                    phi[ind] = phiAdvect[ind]


def solidPhaseFieldHysteresisCircle(phi, phiAdvect, phiFSolidBoundary,
                                    solidBoundary, interfaceWidth, center,
                                    thetaLeft, thetaRight, leftReceding,
                                    rightReceding, leftAdvancing,
                                    rightAdvancing, contactAngleReceding,
                                    contactAngleAdvancing, wettingFunc,
                                    obsOrigin, Nx, Ny):
    for ind in range(Nx * Ny):
        if solidBoundary[ind] == 1:
            i, j = int(ind / Ny), int(ind % Ny)
            theta_local = np.arctan2(j - obsOrigin[1], i - obsOrigin[0]) +\
                np.pi
            distFromLeft, distFromRight = 1e10, 1e10
            distFromLeft_1 = theta_local - thetaLeft[0]
            distFromLeft_2 = theta_local - thetaLeft[0] -\
                2 * np.pi
            distFromLeft_3 = theta_local + 2 * np.pi -\
                thetaLeft[0]
            if np.abs(distFromLeft_1) < np.abs(distFromLeft):
                distFromLeft = distFromLeft_1
            if np.abs(distFromLeft_2) < np.abs(distFromLeft):
                distFromLeft = distFromLeft_2
            if np.abs(distFromLeft_3) < np.abs(distFromLeft):
                distFromLeft = distFromLeft_3
            distFromRight_1 = theta_local - thetaRight[0]
            distFromRight_2 = theta_local - thetaRight[0] -\
                2 * np.pi
            distFromRight_3 = theta_local + 2 * np.pi -\
                thetaRight[0]
            if np.abs(distFromRight_1) < np.abs(distFromRight):
                distFromRight = distFromRight_1
            if np.abs(distFromRight_2) < np.abs(distFromRight):
                distFromRight = distFromRight_2
            if np.abs(distFromRight_3) < np.abs(distFromRight):
                distFromRight = distFromRight_3
            if np.abs(distFromLeft) <= np.abs(distFromRight):
                # wettingFunc(ind, phi, phiFSolidBoundary, interfaceWidth,
                #             contactAngleAdvancing)
                if leftReceding[0] is True and leftAdvancing[0] is False:
                    wettingFunc(ind, phi, phiFSolidBoundary, interfaceWidth,
                                contactAngleReceding)
                elif leftReceding[0] is False and leftAdvancing[0] is True:
                    wettingFunc(ind, phi, phiFSolidBoundary, interfaceWidth,
                                contactAngleAdvancing)
                elif leftReceding[0] is False and leftAdvancing[0] is False:
                    phi[ind] = phiAdvect[ind]
            elif np.abs(distFromLeft) > np.abs(distFromRight):
                # wettingFunc(ind, phi, phiFSolidBoundary, interfaceWidth,
                #             contactAngleReceding)
                if rightReceding[0] is True and rightAdvancing[0] is False:
                    wettingFunc(ind, phi, phiFSolidBoundary, interfaceWidth,
                                contactAngleReceding)
                elif rightReceding[0] is False and rightAdvancing[0] is True:
                    wettingFunc(ind, phi, phiFSolidBoundary, interfaceWidth,
                                contactAngleAdvancing)
                elif rightReceding[0] is False and rightAdvancing[0] is False:
                    phi[ind] = phiAdvect[ind]


def updateGhostHysteresis(solidBoundary, phi, phiAdvect, obsU, Nx, Ny,
                          xl, xr, center, xl_0, xr_0, center_0,
                          interfaceWidth, interfaceWidthEffLeft,
                          interfaceWidthEffRight, timeStep, deltaFuncSolid,
                          contactAngleLocalSolid, contactAngleReceding,
                          contactAngleAdvancing, contactAngleLeft,
                          contactAngleRight, smoothening=False):
    # phiLine = np.zeros((Nx, 2), dtype=np.float64)
    bottomPlate, topPlate = 0, 0
    flag = False
    for j in range(Ny):
        ind = Nx//2 * Ny + j
        if solidBoundary[ind] == 1 and flag is False:
            bottomPlate = j
            flag = True
        elif solidBoundary[ind] == 1 and flag is True:
            topPlate = j
    # for ind in range(Nx * Ny):
    #     i, j = int(ind / Ny), int(ind % Ny)
    #     if j == topPlate:
    #         phiLine[i, 0] = phi[ind]
    #     if j == bottomPlate:
    #         phiLine[i, 1] = phi[ind]
    #     # if j == 56:
    #     #     phiLine[i, 0] = phi[ind]
    #     # elif j == 46:
    #     #     phiLine[i, 1] = phi[ind]
    # x = np.linspace(0, Nx - 1, Nx)
    # # xl[0], node = locateInterface(phiLine[:, 0], x, start=0)
    # # xr[0], node = locateInterface(phiLine[Nx//2:, 0], x[Nx//2:], start=1)
    # xl[0], node = locateInterfaceTanh(phiLine[:, 0], x, interfaceWidth,
    #                                   start=0)
    # xr[0], node = locateInterfaceTanh(phiLine[Nx//2:, 0], x[Nx//2:],
    #                                   interfaceWidth, start=1)
    # print(xl[0], xr[0])
    if smoothening is False:
        xl[0] = xl[0] + obsU[0, 0]
        xr[0] = xr[0] + obsU[0, 0]
        xl[1] = xl[1] + obsU[1, 0]
        xr[1] = xr[1] + obsU[1, 0]
        # print(obsU[0])
        # print(xl[0], xr[0])
        center[0] = center[0] + obsU[0, 0]
        center[1] = center[1] + obsU[1, 0]
    # interfaceWidthEffLeft[0] = interfaceWidth
    # interfaceWidthEffLeft[1] = interfaceWidth
    # interfaceWidthEffRight[0] = interfaceWidth
    # interfaceWidthEffRight[1] = interfaceWidth
    if smoothening is True:
        for ind in range(Nx * Ny):
            if solidBoundary[ind] == 1:
                i, j = int(ind / Ny), int(ind % Ny)
                if j == topPlate:  # j == 56:
                    if deltaFuncSolid[ind] >= 1e-1 and i < center[0]:
                        if contactAngleLocalSolid[ind] <= contactAngleReceding:
                            sinThetaLocal = np.sin(contactAngleReceding)
                            interfaceWidthEffLeft[0] = interfaceWidth /\
                                sinThetaLocal
                        elif (contactAngleLocalSolid[ind] >=
                                contactAngleAdvancing):
                            sinThetaLocal = np.sin(contactAngleAdvancing)
                            interfaceWidthEffLeft[0] = interfaceWidth /\
                                sinThetaLocal
                    elif deltaFuncSolid[ind] >= 1e-1 and i >= center[0]:
                        if contactAngleLocalSolid[ind] <= contactAngleReceding:
                            sinThetaLocal = np.sin(contactAngleReceding)
                            interfaceWidthEffRight[0] = interfaceWidth /\
                                sinThetaLocal
                        elif (contactAngleLocalSolid[ind] >=
                                contactAngleAdvancing):
                            sinThetaLocal = np.sin(contactAngleAdvancing)
                            interfaceWidthEffRight[0] = interfaceWidth /\
                                sinThetaLocal
                elif j == bottomPlate:  # j == 56:
                    if deltaFuncSolid[ind] >= 1e-1 and i < center[1]:
                        if contactAngleLocalSolid[ind] <= contactAngleReceding:
                            sinThetaLocal = np.sin(contactAngleReceding)
                            interfaceWidthEffLeft[1] = interfaceWidth /\
                                sinThetaLocal
                        elif (contactAngleLocalSolid[ind] >=
                                contactAngleAdvancing):
                            sinThetaLocal = np.sin(contactAngleAdvancing)
                            interfaceWidthEffLeft[1] = interfaceWidth /\
                                sinThetaLocal
                    elif deltaFuncSolid[ind] >= 1e-1 and i >= center[1]:
                        if contactAngleLocalSolid[ind] <= contactAngleReceding:
                            sinThetaLocal = np.sin(contactAngleReceding)
                            interfaceWidthEffRight[1] = interfaceWidth /\
                                sinThetaLocal
                        elif (contactAngleLocalSolid[ind] >=
                                contactAngleAdvancing):
                            sinThetaLocal = np.sin(contactAngleAdvancing)
                            interfaceWidthEffRight[1] = interfaceWidth /\
                                sinThetaLocal
    for ind in range(Nx * Ny):
        if solidBoundary[ind] == 1:
            i, j = int(ind / Ny), int(ind % Ny)
            if j == topPlate and i < center[0]:
                phi[ind] = 0.5 * (1 + np.tanh(2 * (i - xl[0]) /
                                  interfaceWidthEffLeft[0]))
                if smoothening is False:
                    phiAdvect[ind] = 0.5 * (1 + np.tanh(2 * (i - xl[0]) /
                                            interfaceWidthEffLeft[0]))
            elif j == topPlate and i >= center[0]:
                phi[ind] = 0.5 * (1 - np.tanh(2 * (i - xr[0]) /
                                  interfaceWidthEffRight[0]))
                if smoothening is False:
                    phiAdvect[ind] = 0.5 * (1 - np.tanh(2 * (i - xr[0]) /
                                            interfaceWidthEffRight[0]))
            elif j == bottomPlate and i < center[0]:
                phi[ind] = 0.5 * (1 + np.tanh(2 * (i - xl[1]) /
                                  interfaceWidthEffLeft[1]))
                if smoothening is False:
                    phiAdvect[ind] = 0.5 * (1 + np.tanh(2 * (i - xl[1]) /
                                            interfaceWidthEffLeft[1]))
            elif j == bottomPlate and i >= center[0]:
                phi[ind] = 0.5 * (1 - np.tanh(2 * (i - xr[1]) /
                                  interfaceWidthEffRight[1]))
                if smoothening is False:
                    phiAdvect[ind] = 0.5 * (1 - np.tanh(2 * (i - xr[1]) /
                                            interfaceWidthEffRight[1]))
            else:
                phi[ind] = 0
                phiAdvect[ind] = 0
        else:
            phiAdvect[ind] = 0
            # if j == 61:  # j == 56:
            #     phi[ind] = 0.5 * (1 - np.tanh(2 * (i - xl[0]) /
            #                       interfaceWidthEffLeft))
            #     phiAdvect[ind] = 0.5 * (1 - np.tanh(2 * (i - xl[0]) /
            #                             interfaceWidthEffLeft))
            # elif j == 40:  # j == 56:
            #     phi[ind] = 0.5 * (1 - np.tanh(2 * (i - xr[0]) /
            #                       interfaceWidthEffRight))
            #     phiAdvect[ind] = 0.5 * (1 - np.tanh(2 * (i - xr[0]) /
            #                             interfaceWidthEffRight))
            # else:
            #     if i < Nx//2:
            #         phi[ind] = 1
            #         phiAdvect[ind] = 1
            #     else:
            #         phi[ind] = 0
            #         phiAdvect[ind] = 0


@numba.njit
def avgContactAngle(Nx, Ny, solidBoundary, phi, deltaFuncSolid, delPhiSolid,
                    contactAngleLocalSolid, xl, xr, contactAngleLeft,
                    contactAngleRight):
    numeratorLeft, numeratorRight = 0, 0
    denominatorLeft, denominatorRight = 0, 0
    for ind in range(Nx * Ny):
        if solidBoundary[ind] == 1:
            i, j = int(ind / Ny), int(ind % Ny)
            if j == 61:
                numeratorLeft += contactAngleLocalSolid[ind] *\
                    deltaFuncSolid[ind] * delPhiSolid[ind]
                denominatorLeft += deltaFuncSolid[ind] *\
                    delPhiSolid[ind]
            elif j == 40:
                numeratorRight += contactAngleLocalSolid[ind] *\
                    deltaFuncSolid[ind] * delPhiSolid[ind]
                denominatorRight += deltaFuncSolid[ind] *\
                    delPhiSolid[ind]
    contactAngleLeft[0] = numeratorLeft / denominatorLeft
    contactAngleRight[0] = numeratorRight / denominatorRight


@numba.njit
def interfacePositionUpdate(Nx, Ny, solidBoundary, phi, interfaceWidth, xl, xr,
                            center, contactAngleLeft, contactAngleRight,
                            contactAngleReceding, contactAngleAdvancing,
                            contactAngleLocalSolid, deltaFuncSolid,
                            interfaceWidthEffLeft, interfaceWidthEffRight):
    phiLine = np.zeros((Nx, 2), dtype=np.float64)
    bottomPlate, topPlate = 0, 0
    flag = False
    for j in range(Ny):
        ind = Nx//2 * Ny + j
        if solidBoundary[ind] == 1 and flag is False:
            bottomPlate = j
            flag = True
        elif solidBoundary[ind] == 1 and flag is True:
            topPlate = j
    # print(bottomPlate, topPlate)
    # interfaceWidthEffLeftTop = interfaceWidth
    # interfaceWidthEffLeftBottom = interfaceWidth
    # interfaceWidthEffRightTop = interfaceWidth
    # interfaceWidthEffRightBottom = interfaceWidth
    for ind in range(Nx * Ny):
        i, j = int(ind / Ny), int(ind % Ny)
        if j == topPlate:
            phiLine[i, 0] = phi[ind]
            # if deltaFuncSolid[ind] > 1e-1 and i < center[0]:
            #     if contactAngleLocalSolid[ind] <= contactAngleReceding:
            #         sinThetaLocal = np.sin(contactAngleReceding)
            #         interfaceWidthEffLeftTop = interfaceWidth /\
            #             sinThetaLocal
            #     elif (contactAngleLocalSolid[ind] >=
            #             contactAngleAdvancing):
            #         sinThetaLocal = np.sin(contactAngleAdvancing)
            #         interfaceWidthEffLeftTop = interfaceWidth /\
            #             sinThetaLocal
            # elif deltaFuncSolid[ind] > 1e-1 and i >= center[0]:
            #     if contactAngleLocalSolid[ind] <= contactAngleReceding:
            #         sinThetaLocal = np.sin(contactAngleReceding)
            #         interfaceWidthEffRightTop = interfaceWidth /\
            #             sinThetaLocal
            #     elif (contactAngleLocalSolid[ind] >=
            #             contactAngleAdvancing):
            #         sinThetaLocal = np.sin(contactAngleAdvancing)
            #         interfaceWidthEffRightTop = interfaceWidth /\
            #             sinThetaLocal
        if j == bottomPlate:
            phiLine[i, 1] = phi[ind]
            # if deltaFuncSolid[ind] > 1e-1 and i < center[1]:
            #     if contactAngleLocalSolid[ind] <= contactAngleReceding:
            #         sinThetaLocal = np.sin(contactAngleReceding)
            #         interfaceWidthEffLeftBottom = interfaceWidth /\
            #             sinThetaLocal
            #     elif (contactAngleLocalSolid[ind] >=
            #             contactAngleAdvancing):
            #         sinThetaLocal = np.sin(contactAngleAdvancing)
            #         interfaceWidthEffLeftBottom = interfaceWidth /\
            #             sinThetaLocal
            # elif deltaFuncSolid[ind] > 1e-1 and i >= center[1]:
            #     if contactAngleLocalSolid[ind] <= contactAngleReceding:
            #         sinThetaLocal = np.sin(contactAngleReceding)
            #         interfaceWidthEffRightBottom = interfaceWidth /\
            #             sinThetaLocal
            #     elif (contactAngleLocalSolid[ind] >=
            #             contactAngleAdvancing):
            #         sinThetaLocal = np.sin(contactAngleAdvancing)
            #         interfaceWidthEffRightBottom = interfaceWidth /\
            #             sinThetaLocal
    # contactAngleTopLeft, contactAngleTopRight, contactAngleBottomLeft, \
    #     contactAngleBottomRight = \
    #     computeContactAngleInterface(Nx, Ny, xl, xr, contactAngleLocalSolid,
    #                                  topPlate, bottomPlate)
    # if contactAngleTopLeft <= contactAngleReceding:
    #     sinThetaLocal = np.sin(contactAngleReceding)
    #     interfaceWidthEffLeftTop = interfaceWidth /\
    #         sinThetaLocal
    # elif contactAngleTopLeft >= contactAngleAdvancing:
    #     sinThetaLocal = np.sin(contactAngleAdvancing)
    #     interfaceWidthEffLeftTop = interfaceWidth /\
    #         sinThetaLocal
    # if contactAngleTopRight <= contactAngleReceding:
    #     sinThetaLocal = np.sin(contactAngleReceding)
    #     interfaceWidthEffRightTop = interfaceWidth /\
    #         sinThetaLocal
    # elif contactAngleTopRight >= contactAngleAdvancing:
    #     sinThetaLocal = np.sin(contactAngleAdvancing)
    #     interfaceWidthEffRightTop = interfaceWidth /\
    #         sinThetaLocal
    # if contactAngleBottomLeft <= contactAngleReceding:
    #     sinThetaLocal = np.sin(contactAngleReceding)
    #     interfaceWidthEffLeftBottom = interfaceWidth /\
    #         sinThetaLocal
    # elif contactAngleBottomLeft >= contactAngleAdvancing:
    #     sinThetaLocal = np.sin(contactAngleAdvancing)
    #     interfaceWidthEffLeftBottom = interfaceWidth /\
    #         sinThetaLocal
    # if contactAngleBottomRight <= contactAngleReceding:
    #     sinThetaLocal = np.sin(contactAngleReceding)
    #     interfaceWidthEffRightBottom = interfaceWidth /\
    #         sinThetaLocal
    # elif contactAngleBottomRight >= contactAngleAdvancing:
    #     sinThetaLocal = np.sin(contactAngleAdvancing)
    #     interfaceWidthEffRightBottom = interfaceWidth /\
    #         sinThetaLocal
    x = np.linspace(0, Nx - 1, Nx)
    # xl[0], node = locateInterfaceTanh(phiLine[:, 0], x, interfaceWidth,
    #                                   start=0)
    # xr[0], node = locateInterfaceTanh(phiLine[Nx//2:, 0], x[Nx//2:],
    #                                   interfaceWidth, start=1)
    # if contactAngleLeft[0] <= contactAngleReceding:
    #     sinThetaLocal = np.sin(contactAngleReceding)
    # elif contactAngleLeft[0] >= contactAngleAdvancing:
    #     sinThetaLocal = np.sin(contactAngleAdvancing)
    # else:
    #     sinThetaLocal = 1
    xl[0], node = locateInterfaceTanh(phiLine[:, 0], x,
                                      interfaceWidthEffLeft[0], start=0)
    xr[0], node = locateInterfaceTanh(phiLine[int(center[0]):, 0],
                                      x[int(center[0]):],
                                      interfaceWidthEffRight[0], start=1)
    # xl[0], node = locateInterface(phiLine[:, 0], x, start=0)
    # xr[0], node = locateInterface(phiLine[int(center[0]):, 0],
    #                               x[int(center[0]):], start=1)
    # if contactAngleRight[0] <= contactAngleReceding:
    #     sinThetaLocal = np.sin(contactAngleReceding)
    # elif contactAngleRight[0] >= contactAngleAdvancing:
    #     sinThetaLocal = np.sin(contactAngleAdvancing)
    # else:
    #     sinThetaLocal = 1
    xl[1], node = locateInterfaceTanh(phiLine[:, 1], x,
                                      interfaceWidthEffLeft[1], start=0)
    xr[1], node = locateInterfaceTanh(phiLine[int(center[1]):, 1],
                                      x[int(center[1]):],
                                      interfaceWidthEffRight[1], start=1)
    # xl[1], node = locateInterface(phiLine[:, 1], x, start=0)
    # xr[1], node = locateInterface(phiLine[int(center[1]):, 1],
    #                               x[int(center[1]):], start=1)
    # xl[0], node = locateInterface(phiLine[:, 0], x, start=1)
    # xr[0], node = locateInterface(phiLine[:, 1], x, start=1)


@numba.njit
def interfacePositionUpdateCircle(thetaSolid, phiThetaSolid, thetaLeft,
                                  thetaRight, noOfSolidBoundary, radius,
                                  interfaceWidth, interfaceWidthEffLeft,
                                  interfaceWidthEffRight,
                                  contactAngleLeft, contactAngleRight,
                                  contactAngleThetaSolid):
    thetaInterpolate = np.zeros(3 * noOfSolidBoundary, dtype=np.float64)
    thetaInterpolate[:noOfSolidBoundary] = np.copy(thetaSolid)
    thetaInterpolate[noOfSolidBoundary:(2 * noOfSolidBoundary)] =\
        np.copy(thetaSolid) + 2 * np.pi
    thetaInterpolate[(2 * noOfSolidBoundary):] = np.copy(thetaSolid) +\
        4 * np.pi
    phiInterpolate = np.zeros(3 * noOfSolidBoundary, dtype=np.float64)
    phiInterpolate[:noOfSolidBoundary] = np.copy(phiThetaSolid)
    phiInterpolate[noOfSolidBoundary:(2 * noOfSolidBoundary)] =\
        np.copy(phiThetaSolid)
    phiInterpolate[(2 * noOfSolidBoundary):] = np.copy(phiThetaSolid)
    contactAngleInterpolate = np.zeros(3 * noOfSolidBoundary, dtype=np.float64)
    contactAngleInterpolate[:noOfSolidBoundary] =\
        np.copy(contactAngleThetaSolid)
    contactAngleInterpolate[noOfSolidBoundary:(2 * noOfSolidBoundary)] =\
        np.copy(contactAngleThetaSolid)
    contactAngleInterpolate[(2 * noOfSolidBoundary):] =\
        np.copy(contactAngleThetaSolid)
    if phiInterpolate[noOfSolidBoundary] > 0.5:
        thetaRight[0], contactAngleRight[0], nodeRight = \
            locateInterfaceTanhCircle(phiInterpolate[noOfSolidBoundary:],
                                      contactAngleInterpolate
                                      [noOfSolidBoundary:],
                                      thetaInterpolate[noOfSolidBoundary:],
                                      radius, interfaceWidthEffRight[0],
                                      start=1)
        thetaLeft[0], contactAngleLeft[0], nodeLeft = \
            locateInterfaceTanhCircle(phiInterpolate[int(noOfSolidBoundary +
                                                         nodeRight + 20):],
                                      contactAngleInterpolate
                                      [int(noOfSolidBoundary +
                                       nodeRight + 20):],
                                      thetaInterpolate[int(noOfSolidBoundary +
                                                           nodeRight + 20):],
                                      radius, interfaceWidthEffLeft[0],
                                      start=0)
    elif phiInterpolate[noOfSolidBoundary] <= 0.5:
        thetaLeft[0], contactAngleLeft[0], nodeLeft = \
            locateInterfaceTanhCircle(phiInterpolate[noOfSolidBoundary:],
                                      contactAngleInterpolate
                                      [noOfSolidBoundary:],
                                      thetaInterpolate[noOfSolidBoundary:],
                                      radius, interfaceWidthEffLeft[0],
                                      start=0)
        thetaRight[0], contactAngleRight[0], nodeRight = \
            locateInterfaceTanhCircle(phiInterpolate[int(noOfSolidBoundary +
                                                         nodeLeft + 20):],
                                      contactAngleInterpolate
                                      [int(noOfSolidBoundary +
                                       nodeLeft + 20):],
                                      thetaInterpolate[int(noOfSolidBoundary +
                                                           nodeLeft + 20):],
                                      radius, interfaceWidthEffRight[0],
                                      start=1)


def updatePhiThetaSolid(solidBoundary, phi, contactAngleLocalSolid,
                        phiThetaSolid, sortedIndex, contactAngleThetaSolid,
                        Nx, Ny):
    itr = 0
    for ind in range(Nx * Ny):
        if solidBoundary[ind] == 1:
            phiThetaSolid[int(sortedIndex[itr])] = phi[ind]
            contactAngleThetaSolid[int(sortedIndex[itr])] =\
                contactAngleLocalSolid[ind]
            itr += 1


def updateGhostHysteresisCircle(solidBoundary, phi, deltaFuncSolid,
                                contactAngleLocalSolid, phiAdvect, obsOmega,
                                contactAngleAdvancing, contactAngleReceding,
                                Nx, Ny, thetaSolid, phiThetaSolid, thetaLeft,
                                thetaRight, center, thetaLeft_0, thetaRight_0,
                                center_0, interfaceWidth, obsOrigin, radius,
                                interfaceWidthEffLeft, interfaceWidthEffRight,
                                sortedIndex, noOfSolidBoundary, timeStep,
                                smoothening=False):
    # interfacePositionUpdate(thetaSolid, phiThetaSolid, thetaLeft, thetaRight,
    #                         noOfSolidBoundary, radius, interfaceWidth)
    if smoothening is False:
        thetaLeft[0] = (thetaLeft[0] + obsOmega) % (2 * np.pi)
        thetaRight[0] = (thetaRight[0] + obsOmega) % (2 * np.pi)
        center[0] = (center[0] + obsOmega) % (2 * np.pi)
    else:
        thetaLeft[0] = thetaLeft[0] % (2 * np.pi)
        thetaRight[0] = thetaRight[0] % (2 * np.pi)
        center[0] = center[0] % (2 * np.pi)
    # interfaceWidthEffLeft[0] = interfaceWidth
    # interfaceWidthEffRight[0] = interfaceWidth
    # if smoothening is True:
    #     for ind in range(Nx * Ny):
    #         if solidBoundary[ind] == 1:
    #             if deltaFuncSolid[ind] > 1e-1:
    #                 i, j = int(ind / Ny), int(ind % Ny)
    #                 theta_local = np.arctan2(j - obsOrigin[1], i - obsOrigin[0]) +\
    #                     np.pi
    #                 distFromLeft, distFromRight = 1e10, 1e10
    #                 distFromLeft_1 = theta_local - thetaLeft[0]
    #                 distFromLeft_2 = theta_local - thetaLeft[0] -\
    #                     2 * np.pi
    #                 distFromLeft_3 = theta_local + 2 * np.pi -\
    #                     thetaLeft[0]
    #                 if np.abs(distFromLeft_1) < np.abs(distFromLeft):
    #                     distFromLeft = distFromLeft_1
    #                 if np.abs(distFromLeft_2) < np.abs(distFromLeft):
    #                     distFromLeft = distFromLeft_2
    #                 if np.abs(distFromLeft_3) < np.abs(distFromLeft):
    #                     distFromLeft = distFromLeft_3
    #                 distFromRight_1 = theta_local - thetaRight[0]
    #                 distFromRight_2 = theta_local - thetaRight[0] -\
    #                     2 * np.pi
    #                 distFromRight_3 = theta_local + 2 * np.pi -\
    #                     thetaRight[0]
    #                 if np.abs(distFromRight_1) < np.abs(distFromRight):
    #                     distFromRight = distFromRight_1
    #                 if np.abs(distFromRight_2) < np.abs(distFromRight):
    #                     distFromRight = distFromRight_2
    #                 if np.abs(distFromRight_3) < np.abs(distFromRight):
    #                     distFromRight = distFromRight_3
    #                 if np.abs(distFromLeft) <= np.abs(distFromRight):
    #                     if contactAngleLocalSolid[ind] <= contactAngleReceding:
    #                         sinThetaLocal = np.sin(contactAngleReceding)
    #                         interfaceWidthEffLeft[0] = interfaceWidth /\
    #                             sinThetaLocal
    #                     elif (contactAngleLocalSolid[ind] >=
    #                             contactAngleAdvancing):
    #                         sinThetaLocal = np.sin(contactAngleAdvancing)
    #                         interfaceWidthEffLeft[0] = interfaceWidth /\
    #                             sinThetaLocal
    #                 elif np.abs(distFromLeft) > np.abs(distFromRight):
    #                     if contactAngleLocalSolid[ind] <= contactAngleReceding:
    #                         sinThetaLocal = np.sin(contactAngleReceding)
    #                         interfaceWidthEffRight[0] = interfaceWidth /\
    #                             sinThetaLocal
    #                     elif (contactAngleLocalSolid[ind] >=
    #                             contactAngleAdvancing):
    #                         sinThetaLocal = np.sin(contactAngleAdvancing)
    #                         interfaceWidthEffRight[0] = interfaceWidth /\
    #                             sinThetaLocal
    itr = 0
    for ind in range(Nx * Ny):
        if solidBoundary[ind] == 1:
            i, j = int(ind / Ny), int(ind % Ny)
            # r_local = np.sqrt((i - obsOrigin[0]) * (i - obsOrigin[0]) +
            #                   (j - obsOrigin[1]) * (j - obsOrigin[1]))
            theta_local = np.arctan2(j - obsOrigin[1], i - obsOrigin[0]) +\
                np.pi
            distFromLeft, distFromRight = 1e10, 1e10
            distFromLeft_1 = theta_local - thetaLeft[0]
            distFromLeft_2 = theta_local - thetaLeft[0] -\
                2 * np.pi
            distFromLeft_3 = theta_local + 2 * np.pi -\
                thetaLeft[0]
            if np.abs(distFromLeft_1) < np.abs(distFromLeft):
                distFromLeft = distFromLeft_1
            if np.abs(distFromLeft_2) < np.abs(distFromLeft):
                distFromLeft = distFromLeft_2
            if np.abs(distFromLeft_3) < np.abs(distFromLeft):
                distFromLeft = distFromLeft_3
            distFromRight_1 = theta_local - thetaRight[0]
            distFromRight_2 = theta_local - thetaRight[0] -\
                2 * np.pi
            distFromRight_3 = theta_local + 2 * np.pi -\
                thetaRight[0]
            if np.abs(distFromRight_1) < np.abs(distFromRight):
                distFromRight = distFromRight_1
            if np.abs(distFromRight_2) < np.abs(distFromRight):
                distFromRight = distFromRight_2
            if np.abs(distFromRight_3) < np.abs(distFromRight):
                distFromRight = distFromRight_3
            if np.abs(distFromLeft) <= np.abs(distFromRight):
                phi[ind] = 0.5 * (1 + np.tanh(2 * radius *
                                  distFromLeft / interfaceWidthEffLeft[0]))
            elif np.abs(distFromLeft) > np.abs(distFromRight):
                phi[ind] = 0.5 * (1 - np.tanh(2 * radius *
                                  distFromRight / interfaceWidthEffRight[0]))
            phiThetaSolid[int(sortedIndex[itr])] = phi[ind]
            if smoothening is False:
                phiAdvect[ind] = phi[ind]
            itr += 1
        # else:
        #     phiAdvect[ind] = 0


def obtainSolidPhiData(solidNodes, phiFSolidBoundary, normalPhiDotNormalSolid,
                       boundaryNode, procBoundary, surfaceNormals,
                       phi, phi_temp, normalPhi, normalPhi_temp, gradPhi,
                       gradPhi_temp, Nx, Ny, size, contactAngleLocalSolid,
                       delPhiSolid, deltaFuncSolid, solidBoundary):
    for itr in prange(solidNodes.shape[0]):
        ind = solidNodes[itr]
        i, j = int(ind / Ny), int(ind % Ny)
        i_nb = i + surfaceNormals[itr, 0]
        j_nb = j + surfaceNormals[itr, 1]
        i_nb_normals = i + 0.5 * surfaceNormals[itr, 0]
        j_nb_normals = j + 0.5 * surfaceNormals[itr, 1]
        i_next, j_next = int(np.ceil(i_nb)), int(np.ceil(j_nb))
        i_prev, j_prev = int(np.floor(i_nb)), int(np.floor(j_nb))
        i_next_normals = int(np.ceil(i_nb_normals))
        j_next_normals = int(np.ceil(j_nb_normals))
        i_prev_normals = int(np.floor(i_nb_normals))
        j_prev_normals = int(np.floor(j_nb_normals))
        if i_next == i_prev and j_next == j_prev:
            deltaX, deltaY = i_next - i, j_next - j
            deltaX_normals = i_next_normals - i
            deltaY_normals = j_next_normals - j
        else:
            deltaX, deltaY = i_next - i_prev, j_next - j_prev
            deltaX_normals = i_next_normals - i_prev_normals
            deltaY_normals = j_next_normals - j_prev_normals
        i_next_phi, j_next_phi = i_next, j_next
        i_prev_phi, j_prev_phi = i_prev, j_prev
        i_next_phi_normals = i_next_normals
        j_next_phi_normals = j_next_normals
        i_prev_phi_normals = i_prev_normals
        j_prev_phi_normals = j_prev_normals
        if size == 1 and boundaryNode[ind] == 2:
            if i_next == Nx - 1:
                i_next_phi = (i_next + 2 + Nx) % Nx
                i_next_phi_normals = (i_next_normals + 2 + Nx) % Nx
            if j_next == Ny - 1:
                j_next_phi = (j_next + 2 + Ny) % Ny
                j_next_phi_normals = (j_next_normals + 2 + Ny) % Ny
            if i_prev == 0:
                i_prev_phi = (i_prev - 2 + Nx) % Nx
                i_prev_phi_normals = (i_prev_normals - 2 + Nx) % Nx
            if j_prev == 0:
                j_prev_phi = (j_prev - 2 + Ny) % Ny
                j_prev_phi_normals = (j_prev_normals - 2 + Ny) % Ny
        elif size > 1 and boundaryNode[ind] == 2:
            if i_next == Nx - 2:
                i_next_phi = i_next + 1
                i_next_phi_normals = i_next_normals + 1
            if j_next == Ny - 2:
                j_next_phi = j_next + 1
                j_next_phi_normals = j_next_normals + 1
            if i_prev == 1:
                i_prev_phi = i_prev - 1
                i_prev_phi_normals = i_prev_normals - 1
            if j_prev == 1:
                j_prev_phi = j_prev - 1
                j_prev_phi_normals = j_prev_normals - 1
        solidNormalDotNormalPhi = 0
        if deltaX == 0:
            if surfaceNormals[itr, 1] < 0:
                normalPhi_fluid_x = \
                    (normalPhi_temp[int(i_prev_phi * Ny + j_prev_phi), 0] +
                     normalPhi_temp[ind, 0]) / 2
                normalPhi_fluid_y = \
                    (normalPhi_temp[int(i_prev_phi * Ny + j_prev_phi), 1] +
                     normalPhi_temp[ind, 1]) / 2
                gradPhi_fluid_x = \
                    (gradPhi_temp[int(i_prev_phi * Ny + j_prev_phi), 0] +
                     gradPhi_temp[ind, 0]) / 2
                gradPhi_fluid_y = \
                    (gradPhi_temp[int(i_prev_phi * Ny + j_prev_phi), 1] +
                     gradPhi_temp[ind, 1]) / 2
                # normalPhi_fluid_x = \
                #     normalPhi_temp[int(i_prev_phi * Ny + j_prev_phi), 0]
                # normalPhi_fluid_y = \
                #     normalPhi_temp[int(i_prev_phi * Ny + j_prev_phi), 1]
                # gradPhi_fluid_x = \
                #     gradPhi_temp[int(i_prev_phi * Ny + j_prev_phi), 0]
                # gradPhi_fluid_y = \
                #     gradPhi_temp[int(i_prev_phi * Ny + j_prev_phi), 1]
            elif surfaceNormals[itr, 1] > 0:
                normalPhi_fluid_x = \
                    (normalPhi_temp[int(i_prev_phi * Ny + j_next_phi), 0] +
                     normalPhi_temp[ind, 0]) / 2
                normalPhi_fluid_y = \
                    (normalPhi_temp[int(i_prev_phi * Ny + j_next_phi), 1] +
                     normalPhi_temp[ind, 1]) / 2
                gradPhi_fluid_x = \
                    (gradPhi_temp[int(i_prev_phi * Ny + j_next_phi), 0] +
                     gradPhi_temp[ind, 0]) / 2
                gradPhi_fluid_y = \
                    (gradPhi_temp[int(i_prev_phi * Ny + j_next_phi), 1] +
                     gradPhi_temp[ind, 1]) / 2
                # normalPhi_fluid_x = \
                #     normalPhi_temp[int(i_prev_phi * Ny + j_next_phi), 0]
                # normalPhi_fluid_y = \
                #     normalPhi_temp[int(i_prev_phi * Ny + j_next_phi), 1]
                # gradPhi_fluid_x = \
                #     gradPhi_temp[int(i_prev_phi * Ny + j_next_phi), 0]
                # gradPhi_fluid_y = \
                #     gradPhi_temp[int(i_prev_phi * Ny + j_next_phi), 1]
        elif deltaY == 0:
            if surfaceNormals[itr, 0] < 0:
                normalPhi_fluid_x = \
                    (normalPhi_temp[int(i_prev_phi * Ny + j_prev_phi), 0] +
                     normalPhi_temp[ind, 0]) / 2
                normalPhi_fluid_y = \
                    (normalPhi_temp[int(i_prev_phi * Ny + j_prev_phi), 1] +
                     normalPhi_temp[ind, 1]) / 2
                gradPhi_fluid_x = \
                    (gradPhi_temp[int(i_prev_phi * Ny + j_prev_phi), 0] +
                     gradPhi_temp[ind, 0]) / 2
                gradPhi_fluid_y = \
                    (gradPhi_temp[int(i_prev_phi * Ny + j_prev_phi), 1] +
                     gradPhi_temp[ind, 1]) / 2
                # normalPhi_fluid_x = \
                #     normalPhi_temp[int(i_prev_phi * Ny + j_prev_phi), 0]
                # normalPhi_fluid_y = \
                #     normalPhi_temp[int(i_prev_phi * Ny + j_prev_phi), 1]
                # gradPhi_fluid_x = \
                #     gradPhi_temp[int(i_prev_phi * Ny + j_prev_phi), 0]
                # gradPhi_fluid_y = \
                #     gradPhi_temp[int(i_prev_phi * Ny + j_prev_phi), 1]
            elif surfaceNormals[itr, 0] > 0:
                normalPhi_fluid_x = \
                    (normalPhi_temp[int(i_next_phi * Ny + j_prev_phi), 0] +
                     normalPhi_temp[ind, 0]) / 2
                normalPhi_fluid_y = \
                    (normalPhi_temp[int(i_next_phi * Ny + j_prev_phi), 1] +
                     normalPhi_temp[ind, 1]) / 2
                gradPhi_fluid_x = \
                    (gradPhi_temp[int(i_next_phi * Ny + j_prev_phi), 0] +
                     gradPhi_temp[ind, 0]) / 2
                gradPhi_fluid_y = \
                    (gradPhi_temp[int(i_next_phi * Ny + j_prev_phi), 1] +
                     gradPhi_temp[ind, 1]) / 2
                # normalPhi_fluid_x = \
                #     normalPhi_temp[int(i_next_phi * Ny + j_prev_phi), 0]
                # normalPhi_fluid_y = \
                #     normalPhi_temp[int(i_next_phi * Ny + j_prev_phi), 1]
                # gradPhi_fluid_x = \
                #     gradPhi_temp[int(i_next_phi * Ny + j_prev_phi), 0]
                # gradPhi_fluid_y = \
                #     gradPhi_temp[int(i_next_phi * Ny + j_prev_phi), 1]
        elif deltaX != 0 and deltaY != 0:
            # normalPhi_fluid_x =\
            #     ((i_next - i_nb) * (j_next - j_nb) *
            #         normalPhi_temp[i_prev_phi * Ny + j_prev_phi, 0] +
            #         (i_next - i_nb) * (j_nb - j_prev) *
            #         normalPhi_temp[i_prev_phi * Ny + j_next_phi, 0] +
            #         (i_nb - i_prev) * (j_next - j_nb) *
            #         normalPhi_temp[i_next_phi * Ny + j_prev_phi, 0] +
            #         (i_nb - i_prev) * (j_nb - j_prev) *
            #         normalPhi_temp[i_next_phi * Ny + j_next_phi, 0])
            # normalPhi_fluid_y =\
            #     ((i_next - i_nb) * (j_next - j_nb) *
            #         normalPhi_temp[i_prev_phi * Ny + j_prev_phi, 1] +
            #         (i_next - i_nb) * (j_nb - j_prev) *
            #         normalPhi_temp[i_prev_phi * Ny + j_next_phi, 1] +
            #         (i_nb - i_prev) * (j_next - j_nb) *
            #         normalPhi_temp[i_next_phi * Ny + j_prev_phi, 1] +
            #         (i_nb - i_prev) * (j_nb - j_prev) *
            #         normalPhi_temp[i_next_phi * Ny + j_next_phi, 1])
            # gradPhi_fluid_x =\
            #     ((i_next - i_nb) * (j_next - j_nb) *
            #         gradPhi_temp[i_prev_phi * Ny + j_prev_phi, 0] +
            #         (i_next - i_nb) * (j_nb - j_prev) *
            #         gradPhi_temp[i_prev_phi * Ny + j_next_phi, 0] +
            #         (i_nb - i_prev) * (j_next - j_nb) *
            #         gradPhi_temp[i_next_phi * Ny + j_prev_phi, 0] +
            #         (i_nb - i_prev) * (j_nb - j_prev) *
            #         gradPhi_temp[i_next_phi * Ny + j_next_phi, 0])
            # gradPhi_fluid_y =\
            #     ((i_next - i_nb) * (j_next - j_nb) *
            #         gradPhi_temp[i_prev_phi * Ny + j_prev_phi, 1] +
            #         (i_next - i_nb) * (j_nb - j_prev) *
            #         gradPhi_temp[i_prev_phi * Ny + j_next_phi, 1] +
            #         (i_nb - i_prev) * (j_next - j_nb) *
            #         gradPhi_temp[i_next_phi * Ny + j_prev_phi, 1] +
            #         (i_nb - i_prev) * (j_nb - j_prev) *
            #         gradPhi_temp[i_next_phi * Ny + j_next_phi, 1])
            normalPhi_fluid_x =\
                ((i_next_normals - i_nb) * (j_next_normals - j_nb) *
                 normalPhi_temp[i_prev_phi_normals * Ny + j_prev_phi_normals, 0] +
                 (i_next_normals - i_nb) * (j_nb - j_prev_normals) *
                 normalPhi_temp[i_prev_phi_normals * Ny + j_next_phi_normals, 0] +
                 (i_nb - i_prev_normals) * (j_next_normals - j_nb) *
                 normalPhi_temp[i_next_phi_normals * Ny + j_prev_phi_normals, 0] +
                 (i_nb - i_prev_normals) * (j_nb - j_prev_normals) *
                 normalPhi_temp[i_next_phi_normals * Ny + j_next_phi_normals, 0])
            normalPhi_fluid_y =\
                ((i_next_normals - i_nb) * (j_next_normals - j_nb) *
                 normalPhi_temp[i_prev_phi_normals * Ny + j_prev_phi_normals, 1] +
                 (i_next_normals - i_nb) * (j_nb - j_prev_normals) *
                 normalPhi_temp[i_prev_phi_normals * Ny + j_next_phi_normals, 1] +
                 (i_nb - i_prev_normals) * (j_next_normals - j_nb) *
                 normalPhi_temp[i_next_phi_normals * Ny + j_prev_phi_normals, 1] +
                 (i_nb - i_prev_normals) * (j_nb - j_prev_normals) *
                 normalPhi_temp[i_next_phi_normals * Ny + j_next_phi_normals, 1])
            gradPhi_fluid_x =\
                ((i_next_normals - i_nb) * (j_next_normals - j_nb) *
                 gradPhi_temp[i_prev_phi_normals * Ny + j_prev_phi_normals, 0] +
                 (i_next_normals - i_nb) * (j_nb - j_prev_normals) *
                 gradPhi_temp[i_prev_phi_normals * Ny + j_next_phi_normals, 0] +
                 (i_nb - i_prev_normals) * (j_next_normals - j_nb) *
                 gradPhi_temp[i_next_phi_normals * Ny + j_prev_phi_normals, 0] +
                 (i_nb - i_prev_normals) * (j_nb - j_prev_normals) *
                 gradPhi_temp[i_next_phi_normals * Ny + j_next_phi_normals, 0])
            gradPhi_fluid_y =\
                ((i_next_normals - i_nb) * (j_next_normals - j_nb) *
                 gradPhi_temp[i_prev_phi_normals * Ny + j_prev_phi_normals, 1] +
                 (i_next_normals - i_nb) * (j_nb - j_prev_normals) *
                 gradPhi_temp[i_prev_phi_normals * Ny + j_next_phi_normals, 1] +
                 (i_nb - i_prev_normals) * (j_next_normals - j_nb) *
                 gradPhi_temp[i_next_phi_normals * Ny + j_prev_phi_normals, 1] +
                 (i_nb - i_prev_normals) * (j_nb - j_prev_normals) *
                 gradPhi_temp[i_next_phi_normals * Ny + j_next_phi_normals, 1])
            normalPhi_fluid_x = normalPhi_fluid_x / (deltaX * deltaY)
            normalPhi_fluid_y = normalPhi_fluid_y / (deltaX * deltaY)
            gradPhi_fluid_x = gradPhi_fluid_x / (deltaX * deltaY)
            gradPhi_fluid_y = gradPhi_fluid_y / (deltaX * deltaY)
        normalPhi[ind, 0] = normalPhi_fluid_x
        normalPhi[ind, 1] = normalPhi_fluid_y
        gradPhi[ind, 0] = gradPhi_fluid_x
        gradPhi[ind, 1] = gradPhi_fluid_y
        surfaceTangent_x = -surfaceNormals[itr, 1]
        surfaceTangent_y = surfaceNormals[itr, 0]
        solidTangentDotGradPhi =\
            np.abs(surfaceTangent_x * gradPhi_fluid_x +
                   surfaceTangent_y * gradPhi_fluid_y)
        solidNormalDotGradPhi = \
            surfaceNormals[itr, 0] * gradPhi_fluid_x +\
            surfaceNormals[itr, 1] * gradPhi_fluid_y
        solidNormalDotNormalPhi = \
            surfaceNormals[itr, 0] * normalPhi_fluid_x +\
            surfaceNormals[itr, 1] * normalPhi_fluid_y
        normalPhiDotNormalSolid[ind] = solidNormalDotNormalPhi
        # normalPhiDotNormalSolid[ind] = - solidNormalDotGradPhi /\
        #     solidTangentDotGradPhi
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
        phiFSolidBoundary[ind] = phi_fluid
        contactAngleLocalSolid[ind] = np.arccos(-solidNormalDotNormalPhi)
        deltaFuncSolid[ind] = 6 * phi[ind] * (1 - phi[ind])
        # deltaFuncSolid[ind] = 6 * phi_fluid * (1 - phi_fluid)
        delx, dely = 0, 0
        if solidBoundary[int((i + 1) * Ny + j)] == 1:
            delx += 0.5
        if solidBoundary[int((i - 1) * Ny + j)] == 1:
            delx += 0.5
        if solidBoundary[int(i * Ny + j + 1)] == 1:
            dely += 0.5
        if solidBoundary[int(i * Ny + j - 1)] == 1:
            dely += 0.5
        delPhiSolid[ind] = np.abs(gradPhi_fluid_x) * delx +\
            np.abs(gradPhi_fluid_y) * dely


def setContactAngle(solidNodes, phi, normalPhiDotNormalSolid,
                    phiFSolidBoundary, contactAngle, interfaceWidth,
                    Nx, Ny, wettingFunc, size, hysteresis=False,
                    contactAngleReceding=0, contactAngleAdvancing=0,
                    initial=False):
    for itr in prange(solidNodes.shape[0]):
        ind = solidNodes[itr]
        updateContactAngleValue = contactAngle
        updateContactAngle = True
        contactAngleLocal = np.arccos(-normalPhiDotNormalSolid[ind])
        if hysteresis is True and initial is False:
            if contactAngleLocal < contactAngleReceding:
                updateContactAngleValue = contactAngleReceding
                updateContactAngle = True
            elif contactAngleLocal > contactAngleAdvancing:
                updateContactAngleValue = contactAngleAdvancing
                updateContactAngle = True
            else:
                updateContactAngleValue = contactAngle
                updateContactAngle = False
        if updateContactAngle is True:
            wettingFunc(ind, phi, phiFSolidBoundary, interfaceWidth,
                        updateContactAngleValue)


@numba.njit
def fakhariWetting(ind, phi, phiFSolidBoundary, interfaceWidth,
                   updateContactAngleValue):
    epsilon = - 2 * np.cos(updateContactAngleValue) / interfaceWidth
    epsilon_1 = 1/(epsilon + 1e-17)
    phi[ind] =\
        np.abs(epsilon_1 * (1 + epsilon - np.sqrt((1 + epsilon)
               * (1 + epsilon) - 4 * epsilon * phiFSolidBoundary[ind]))
               - phiFSolidBoundary[ind])


@numba.njit
def regularized(ind, phi, phiFSolidBoundary, interfaceWidth,
                updateContactAngleValue):
    epsilon = np.exp(- 4 * np.cos(updateContactAngleValue) /
                     interfaceWidth)
    phi[ind] = np.abs(phiFSolidBoundary[ind] / (epsilon *
                      (1.0 - phiFSolidBoundary[ind])
                      + phiFSolidBoundary[ind]))


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
               normalPhi, stressTensor, procBoundary, boundaryNode, gravity,
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
