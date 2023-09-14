import os
import numpy as np
import numba
from numba import prange


class phaseFieldDef:
    def __init__(self, phaseDict, lattice, transport, rank, precision):
        try:
            self.M = precision(phaseDict['M'])
            self.interfaceWidth = precision(phaseDict['interfaceWidth'])
            self.viscInterpolation = phaseDict['nuInterpolation']
            if self.viscInterpolation == 'linear':
                self.computeViscFunc = computeViscLinear
            else:
                self.computeViscFunc = computeViscHarmonic
            self.contactAngle = phaseDict['contactAngle']
            if self.contactAngle is not None:
                if (not isinstance(self.contactAngle, int) and
                        not isinstance(self.contactAngle, float)):
                    if rank == 0:
                        print("ERROR! contact angle must be a " +
                              "float/int representing angle in degrees!",
                              flush=True)
                    os._exit(1)
                self.contactAngle = self.contactAngle / 180 * np.pi
            self.displayMass = False
            try:
                self.displayMass = phaseDict['displayMass']
            except KeyError:
                pass
            if self.displayMass is True:
                self.mass = np.zeros(1, dtype=precision)
            self.diffusionTime = int(2 * (self.interfaceWidth * self.
                                     interfaceWidth) / self.M)
            self.kappa = 1.5 * transport.sigma * self.interfaceWidth
            self.beta = 12 * transport.sigma / self.interfaceWidth
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
        self.correctNormal =  \
            numba.njit(correctNormal, parallel=parallel,
                       cache=False, nogil=True)
        self.setSolidPhi =  \
            numba.njit(setSolidPhi, parallel=parallel,
                       cache=False, nogil=True)
        self.computeMass =  \
            numba.njit(computeMass, parallel=parallel,
                       cache=False, nogil=True)
        self.liangWetting =  \
            numba.njit(liangWetting, parallel=parallel,
                       cache=False, nogil=True)
        self.leclaireWetting =  \
            numba.njit(leclaireWetting, parallel=parallel,
                       cache=False, nogil=True)
        self.simpleWetting =  \
            numba.njit(simpleWetting, parallel=parallel,
                       cache=False, nogil=True)

    def setSolidPhaseField(self, options, fields, lattice, mesh, size):
        for itr in range(options.noOfSurfaces):
            pass
            # self.setSolidPhi(options.solidNbNodes[itr], fields.phi,
            #                  fields.solid, fields.boundaryNode,
            #                  fields.procBoundary, lattice.w, lattice.c,
            #                  lattice.noOfDirections, mesh.Nx, mesh.Ny,
            #                  size)
            # if itr == 0:
            #     self.liangWetting(options.solidNbNodes[itr], fields.phi,
            #                       self.contactAngle, mesh.Nx, mesh.Ny)

    def correctNormalPhi(self, options, fields, precision):
        for itr in range(options.noOfSurfaces):
            # self.correctNormal(fields.normalPhi, fields.gradPhi,
            #                    options.surfaceNodes[itr],
            #                    options.surfaceNormals[itr], self.contactAngle)
            # self.leclaireWetting(options.surfaceNodes[itr],
            #                      options.surfaceNormals[itr],
            #                      fields.normalPhi, fields.gradPhi,
            #                      self.contactAngle, precision)
            self.simpleWetting(options.surfaceNodes[itr],
                               options.surfaceNormals[itr],
                               fields.normalPhi, fields.gradPhi,
                               self.contactAngle, precision)


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
        if (procBoundary[ind] != 1 or solid[ind, 0] != 1 or
                boundaryNode[ind] != 1):
            i, j = int(ind / Ny), int(ind % Ny)
            gradPhiSum_x, gradPhiSum_y = 0., 0.
            lapPhiSum = 0.
            denominator = 0.
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
                if initial is False:
                    gradPhiSum_x += c[k, 0] * w[k] * phi[ind_nb]
                    gradPhiSum_y += c[k, 1] * w[k] * phi[ind_nb]
                    lapPhiSum += w[k] * (phi[ind_nb] - phi[ind])
                else:
                    gradPhiSum_x += c[k, 0] * w[k] * phi[ind_nb] * \
                        (1 - solid[ind_nb, 0])
                    gradPhiSum_y += c[k, 1] * w[k] * phi[ind_nb] * \
                        (1 - solid[ind_nb, 0])
                    lapPhiSum += w[k] * (phi[i_nb * Ny + j_nb] - phi[ind]) * \
                        (1 - solid[ind_nb, 0])
                    denominator += w[k] * (1 - solid[ind_nb, 0])
            if initial is False:
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


def setSolidPhi(solidNodes, phi, solid, boundaryNode, procBoundary, w,
                c, noOfDirections, Nx, Ny, size):
    for itr in prange(solidNodes.shape[0]):
        ind = solidNodes[itr]
        if procBoundary[ind] != 1:
            i, j = int(ind / Ny), int(ind % Ny)
            phiSum, denominator = 0., 0.
            for k in range(noOfDirections):
                i_nb = i + int(c[k, 0])
                j_nb = j + int(c[k, 1])
                # if size == 1 and boundaryNode[ind] == 2:
                #     if i + int(2 * c[k, 0]) < 0 or i + int(2 * c[k, 0]) >= Nx:
                #         i_nb = (i_nb + int(2 * c[k, 0]) + Nx) % Nx
                #     else:
                #         i_nb = (i_nb + Nx) % Nx
                #     if j + int(2 * c[k, 1]) < 0 or j + int(2 * c[k, 1]) >= Ny:
                #         j_nb = (j_nb + int(2 * c[k, 1]) + Ny) % Ny
                #     else:
                #         j_nb = (j_nb + Ny) % Ny
                # elif size > 1 and boundaryNode[ind] == 2:
                #     if (i + int(2 * c[k, 0]) == 0 or
                #             i + int(2 * c[k, 0]) == Nx - 1):
                #         i_nb = i_nb + int(c[k, 0])
                #     if (j + int(2 * c[k, 1]) == 0 or
                #             j + int(2 * c[k, 1]) == Ny - 1):
                #         j_nb = j_nb + int(c[k, 1])
                ind_nb = int(i_nb * Ny + j_nb)
                if solid[ind_nb, 0] == 0:
                    phiSum += w[k] * phi[ind_nb]
                    denominator += w[k]
            phi[ind] = phiSum / (denominator + 1e-17)


@numba.njit
def contactAngleFunc(contactAngle, normalPhi, surfaceNormals):
    normalPhiDotSolidNormal = normalPhi[0] * surfaceNormals[0] +\
        normalPhi[1] * surfaceNormals[1]
    magNormalPhi = np.sqrt(normalPhi[0] * normalPhi[0] +
                           normalPhi[1] * normalPhi[1])
    return normalPhiDotSolidNormal - magNormalPhi * np.cos(contactAngle)


def leclaireWetting(surfaceNodes, surfaceNormals, normalPhi, gradPhi,
                    contactAngle, precision):
    for itr in prange(surfaceNodes.shape[0]):
        ind = surfaceNodes[itr]
        normalPhi_1 = np.zeros(2, dtype=precision)
        normalPhi_2 = np.zeros(2, dtype=precision)
        normalPhi_1[0] = normalPhi[ind, 0] - 0.5 * (normalPhi[ind, 0] +
                                                    surfaceNormals[itr, 0])
        normalPhi_1[1] = normalPhi[ind, 1] - 0.5 * (normalPhi[ind, 1] +
                                                    surfaceNormals[itr, 1])
        f_0 = contactAngleFunc(contactAngle, normalPhi[ind],
                               surfaceNormals[itr])
        f_1 = contactAngleFunc(contactAngle, normalPhi_1,
                               surfaceNormals[itr])
        normalPhi_2[0] = (normalPhi[ind, 0] * f_1 - normalPhi_1[0] * f_0) /\
            (f_1 - f_0 + 1e-17)
        normalPhi_2[1] = (normalPhi[ind, 1] * f_1 - normalPhi_1[1] * f_0) /\
            (f_1 - f_0 + 1e-17)
        magNormal = np.sqrt(normalPhi_2[0] * normalPhi_2[0] +
                            normalPhi_2[1] * normalPhi_2[1])
        normalPhi[ind, 0] = normalPhi_2[0] / (magNormal + 1e-17)
        normalPhi[ind, 1] = normalPhi_2[1] / (magNormal + 1e-17)
        magGradPhi = np.sqrt(gradPhi[ind, 0] * gradPhi[ind, 0] +
                             gradPhi[ind, 1] * gradPhi[ind, 1])
        gradPhi[ind, 0] = magGradPhi * normalPhi[ind, 0]
        gradPhi[ind, 1] = magGradPhi * normalPhi[ind, 1]


def liangWetting(solidNodes, phi, contactAngle, Nx, Ny):
    for itr in range(solidNodes.shape[0]):
        ind = solidNodes[itr]
        i, j = int(ind / Ny), int(ind % Ny)
        if j == 3:
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


def correctNormal(normalPhi, gradPhi, surfaceNodes, surfaceNormals,
                  contactAngle):
    for itr in prange(surfaceNodes.shape[0]):
        normal_1 = np.zeros(2)
        normal_2 = np.zeros(2)
        ind = surfaceNodes[itr]
        # normal_1[0] = surfaceNormals[itr, 0] * np.cos(contactAngle) -\
        #     surfaceNormals[itr, 1] * np.sin(contactAngle)
        # normal_1[1] = surfaceNormals[itr, 1] * np.cos(contactAngle) +\
        #     surfaceNormals[itr, 0] * np.sin(contactAngle)
        # normal_2[0] = surfaceNormals[itr, 0] * np.cos(contactAngle) +\
        #     surfaceNormals[itr, 1] * np.sin(contactAngle)
        # normal_2[1] = surfaceNormals[itr, 1] * np.cos(contactAngle) -\
        #     surfaceNormals[itr, 0] * np.sin(contactAngle)
        surfaceNormalDotNormalPhi = (surfaceNormals[itr, 0] *
                                     normalPhi[ind, 0] +
                                     surfaceNormals[itr, 1] *
                                     normalPhi[ind, 1])
        thetaPrime = np.arccos(surfaceNormalDotNormalPhi)
        coeff_1 = (np.cos(contactAngle) - (np.sin(contactAngle) *
                   np.cos(thetaPrime)) / np.sin(thetaPrime))
        coeff_2 = np.sin(contactAngle) / np.sin(thetaPrime)
        normal_1[0] = coeff_1 * surfaceNormals[itr, 0] + coeff_2 *\
            normalPhi[ind, 0]
        normal_1[1] = coeff_1 * surfaceNormals[itr, 1] + coeff_2 *\
            normalPhi[ind, 1]
        coeff_1 = (np.cos(-contactAngle) - (np.sin(-contactAngle) *
                   np.cos(thetaPrime)) / np.sin(thetaPrime))
        coeff_2 = np.sin(-contactAngle) / np.sin(thetaPrime)
        normal_2[0] = coeff_1 * surfaceNormals[itr, 0] + coeff_2 *\
            normalPhi[ind, 0]
        normal_2[1] = coeff_1 * surfaceNormals[itr, 1] + coeff_2 *\
            normalPhi[ind, 1]
        distanceFromNormal_1 = ((normalPhi[ind, 0] - normal_1[0]) *
                                (normalPhi[ind, 0] - normal_1[0])) + \
                               ((normalPhi[ind, 1] - normal_1[1]) *
                                (normalPhi[ind, 1] - normal_1[1]))
        distanceFromNormal_2 = ((normalPhi[ind, 0] - normal_2[0]) *
                                (normalPhi[ind, 0] - normal_2[0])) + \
                               ((normalPhi[ind, 1] - normal_2[1]) *
                                (normalPhi[ind, 1] - normal_2[1]))
        if distanceFromNormal_1 < distanceFromNormal_2:
            normalPhi[ind, 0] = normal_1[0]
            normalPhi[ind, 1] = normal_1[1]
        elif distanceFromNormal_1 > distanceFromNormal_2:
            normalPhi[ind, 0] = normal_2[0]
            normalPhi[ind, 1] = normal_2[1]
        else:
            normalPhi[ind, 0] = surfaceNormals[itr, 0]
            normalPhi[ind, 1] = surfaceNormals[itr, 1]
        magGradPhi = np.sqrt(gradPhi[ind, 0] * gradPhi[ind, 0] +
                             gradPhi[ind, 1] * gradPhi[ind, 1])
        gradPhi[ind, 0] = magGradPhi * normalPhi[ind, 0]
        gradPhi[ind, 1] = magGradPhi * normalPhi[ind, 1]


def simpleWetting(surfaceNodes, surfaceNormals, normalPhi, gradPhi,
                  contactAngle, precision):
    for itr in prange(surfaceNodes.shape[0]):
        ind = surfaceNodes[itr]
        tangent = np.zeros(2, dtype=precision)
        tangent[0] = -surfaceNormals[itr, 1]
        tangent[1] = surfaceNormals[itr, 0]
        gradPhiDotTangent = gradPhi[ind, 0] * tangent[0] +\
            gradPhi[ind, 1] * tangent[1]
        gradPhiDotSolidNormal = -np.tan((np.pi/2 - contactAngle)) *\
            np.abs(gradPhiDotTangent)
        gradPhi[ind, 0] = gradPhiDotSolidNormal * surfaceNormals[itr, 0] +\
            gradPhiDotTangent * tangent[0]
        gradPhi[ind, 1] = gradPhiDotSolidNormal * surfaceNormals[itr, 1] +\
            gradPhiDotTangent * tangent[1]


def computeMass(mass, phi, solid, boundaryNode, procBoundary, Nx, Ny):
    temp = 0.
    for ind in prange(Nx * Ny):
        if (solid[ind, 0] == 0 and boundaryNode[ind] == 0 and
                procBoundary[ind] == 0):
            temp += phi[ind]
    mass[0] = temp
    return mass


def forceFluid(f_new, f_eq, forceField, rho, p, phi, solid, lapPhi, gradPhi,
               stressTensor, procBoundary, boundaryNode, gravity,
               preFactorFluid, forcingPreFactor, constructOperatorFunc,
               collisionOperatorArgs, constructStressTensorFunc,
               computeViscFunc, beta, kappa, mu_l, mu_g, rho_l, rho_g, phi_l,
               phi_g, cs, noOfDirections, c, cs_2, Nx, Ny):
    for ind in prange(Nx * Ny):
        if (solid[ind, 0] != 1 and procBoundary[ind] != 1
                and boundaryNode[ind] != 1):
            nu = computeViscFunc(mu_l, mu_g, phi_g, phi[ind], rho[ind],
                                 rho_l, rho_g)
            constructOperatorFunc(preFactorFluid[ind], forcingPreFactor[ind],
                                  nu, *collisionOperatorArgs)
            constructStressTensorFunc(f_new[ind], f_eq[ind], stressTensor[ind],
                                      preFactorFluid[ind], noOfDirections, c,
                                      cs_2, nu)
            chemPotential = 4 * beta * (phi[ind] - phi_g) * (phi[ind] - phi_l)\
                * (phi[ind] - 0.5) - kappa * lapPhi[ind]
            surfaceTensionForce_x = chemPotential * gradPhi[ind, 0]
            surfaceTensionForce_y = chemPotential * gradPhi[ind, 1]
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
