from numba import prange
import numpy as np


def fixedU(f, f_new, rho, u, solid, faceList, outDirections, invDirections,
           c, w, cs, Nx, Ny, phase):
    cs_2 = 1/(cs * cs)
    for itr in prange(faceList.shape[0]):
        ind = faceList[itr]
        if solid[ind, 0] != 1:
            if phase is False:
                rhoWall = rho[ind]
            else:
                rhoWall = 1
            for direction in range(invDirections.shape[0]):
                i, j = int(ind / Ny), int(ind % Ny)
                i_nb = int(i + c[outDirections[direction], 0])
                j_nb = int(j + c[outDirections[direction], 1])
                ind_nb = int(i_nb * Ny + j_nb)
                preFactor = 2 * w[outDirections[direction]] * rhoWall *\
                    ((c[outDirections[direction], 0] * u[ind_nb, 0] +
                     c[outDirections[direction], 1] * u[ind_nb, 1])) * cs_2
                f_new[ind, invDirections[direction]] = \
                    f[ind, outDirections[direction]] - preFactor


def variableU(f, f_new, rho, u, solid, faceList, outDirections, invDirections,
              c, w, cs, Nx, Ny, phase):
    cs_2 = 1/(cs * cs)
    for itr in prange(faceList.shape[0]):
        ind = faceList[itr]
        if solid[ind, 0] != 1:
            if phase is False:
                rhoWall = rho[ind]
            else:
                rhoWall = 1
            for direction in range(invDirections.shape[0]):
                i_nb = int(int(ind / Ny) + c[outDirections[direction], 0])
                j_nb = int(int(ind % Ny) + c[outDirections[direction], 1])
                ind_nb = int(i_nb * Ny + j_nb)
                preFactor = 2 * w[outDirections[direction]] * rhoWall *\
                    ((c[outDirections[direction], 0] * u[ind_nb, 0] +
                     c[outDirections[direction], 1] * u[ind_nb, 1])) * cs_2
                f_new[ind, invDirections[direction]] = \
                    f[ind, outDirections[direction]] - preFactor


def fixedPressure(f, f_new, rho, p, u, solid, rho_ref, faceList, outDirections,
                  invDirections, nbList, c, w, cs, Nx, Ny, phase):
    cs_2 = 1/(cs * cs)
    cs_4 = cs_2 * cs_2
    for itr in prange(faceList.shape[0]):
        ind = faceList[itr]
        if solid[ind, 0] != 1:
            i, j = int(ind / Ny), int(ind % Ny)
            for direction in invDirections:
                c_mag = int(np.ceil((c[direction, 0] * c[direction, 0]
                            + c[direction, 1] * c[direction, 1])))
                if c_mag == 1:
                    i_nb = int(i + c[direction, 0])
                    j_nb = int(j + c[direction, 1])
                    ind_nb = int(i_nb * Ny + j_nb)
                    break
            # u_nb = u[ind, :] + 0.5 * (u[ind, :] - u[ind_nb, :])
            u_nb_x = u[ind, 0] + 0.5 * (u[ind, 0] - u[ind_nb, 0])
            u_nb_y = u[ind, 1] + 0.5 * (u[ind, 1] - u[ind_nb, 1])
            if phase is True:
                rho_nb = rho[ind] + 0.5 * (rho[ind] - rho[ind_nb])
            # u_nb_2 = u_nb[0] * u_nb[0] + u_nb[1] * u_nb[1]
            u_nb_2 = u_nb_x * u_nb_x + u_nb_y * u_nb_y
            for direction in range(invDirections.shape[0]):
                # c_dot_u = (c[outDirections[direction], 0] * u_nb[0] +
                #            c[outDirections[direction], 1] * u_nb[1])
                c_dot_u = (c[outDirections[direction], 0] * u_nb_x +
                           c[outDirections[direction], 1] * u_nb_y)
                if phase is False and rho_ref is None:
                    preFactor = \
                        2 * w[outDirections[direction]] * rho[nbList[itr]]\
                        * (1. + c_dot_u * c_dot_u * 0.5 * cs_4
                           - u_nb_2 * 0.5 * cs_2)
                elif phase is False and rho_ref is not None:
                    preFactor = 2 * w[outDirections[direction]] *\
                        (rho[nbList[itr]] + rho_ref * (c_dot_u * c_dot_u *
                         0.5 * cs_4 - u_nb_2 * 0.5 * cs_2))
                elif phase is True:
                    preFactor = 2 * w[outDirections[direction]] *\
                        (p[nbList[itr]] * cs_2 / rho_nb + c_dot_u * c_dot_u
                         * 0.5 * cs_4 - u_nb_2 * 0.5 * cs_2)
                f_new[ind, invDirections[direction]] = \
                    - f[ind, outDirections[direction]] + preFactor


def fixedPressureRegularized(f, f_new, f_eq, rho, p, u, solid, boundaryNode,
                             rho_ref, faceList, outDirections, invDirections,
                             normalBoundary, nbList, c, w, cs, noOfDirections,
                             Nx, Ny, equilibriumFunc, equilibriumArgs, phase):
    cs_4 = 1/(cs * cs * cs * cs)
    cs_2 = 1 / (cs * cs)
    for itr in prange(faceList.shape[0]):
        ind = faceList[itr]
        i, j = int(ind / Ny), int(ind % Ny)
        if phase is True:
            normalU = -p[nbList[itr]]
            for k in range(noOfDirections):
                normalU += (1 + c[k, 0] * normalBoundary[0] +
                            c[k, 1] * normalBoundary[1]) * f_new[ind, k]
        else:
            normalU = 0
            for k in range(noOfDirections):
                normalU += (1 + c[k, 0] * normalBoundary[0] +
                            c[k, 1] * normalBoundary[1]) * f_new[ind, k]
                # print(i, k, 1 + c[k, 0] * normalBoundary[0] + c[k, 1] *
                #       normalBoundary[1], f_new[ind, k])
            # print()
            normalU = normalU / rho[nbList[itr]] - 1
        u_nb = np.zeros(2)
        u_nb[0] = normalU * normalBoundary[0]
        u_nb[1] = normalU * normalBoundary[1]
        if phase is True:
            # rho_nb = rhoSum / (denominator + 1e-17)
            equilibriumFunc(f_eq[ind], u_nb, p[nbList[itr]],
                            *equilibriumArgs)
        else:
            equilibriumFunc(f_eq[ind], u_nb, rho[nbList[itr]],
                            *equilibriumArgs)
        for direction in range(invDirections.shape[0]):
            f_new[ind, invDirections[direction]] = \
                f_eq[ind, invDirections[direction]] +\
                f_new[ind, outDirections[direction]] -\
                f_eq[ind, outDirections[direction]]
        secondOrderTensor_xx, secondOrderTensor_xy = 0, 0
        secondOrderTensor_yx, secondOrderTensor_yy = 0, 0
        for k in range(noOfDirections):
            secondOrderTensor_xx += c[k, 0] * c[k, 0] *\
                (f_new[ind, k] - f_eq[ind, k])
            secondOrderTensor_xy += c[k, 0] * c[k, 1] *\
                (f_new[ind, k] - f_eq[ind, k])
            secondOrderTensor_yx += c[k, 1] * c[k, 0] *\
                (f_new[ind, k] - f_eq[ind, k])
            secondOrderTensor_yy += c[k, 1] * c[k, 1] *\
                (f_new[ind, k] - f_eq[ind, k])
        for k in range(noOfDirections):
            f_new[ind, k] = f_eq[ind, k] + 0.5 * w[k] * cs_4 * \
                ((c[k, 0] * c[k, 0] - cs * cs) * secondOrderTensor_xx +
                 c[k, 0] * c[k, 1] * secondOrderTensor_yx +
                 c[k, 1] * c[k, 0] * secondOrderTensor_xy +
                 (c[k, 1] * c[k, 1] - cs * cs) * secondOrderTensor_yy)


def fixedPressureRegularizedMRT(f, f_new, f_eq, rho, p, u, solid, boundaryNode,
                                rho_ref, faceList, outDirections,
                                invDirections, normalBoundary, nbList, c, w,
                                cs, noOfDirections, Nx, Ny, equilibriumFunc,
                                equilibriumArgs, preFactorFluid,
                                preFactorFluidInv, tau_nu, phase):
    cs_4 = 1/(cs * cs * cs * cs)
    cs_2 = 1 / (cs * cs)
    for itr in prange(faceList.shape[0]):
        ind = faceList[itr]
        i, j = int(ind / Ny), int(ind % Ny)
        if phase is True:
            normalU = -p[nbList[itr]]
            for k in range(noOfDirections):
                normalU += (1 + c[k, 0] * normalBoundary[0] +
                            c[k, 1] * normalBoundary[1]) * f_new[ind, k]
        else:
            normalU = 0
            for k in range(noOfDirections):
                normalU += (1 + c[k, 0] * normalBoundary[0] +
                            c[k, 1] * normalBoundary[1]) * f_new[ind, k]
                # print(i, k, 1 + c[k, 0] * normalBoundary[0] + c[k, 1] *
                #       normalBoundary[1], f_new[ind, k])
            # print()
            normalU = normalU / rho[nbList[itr]] - 1
        u_nb = np.zeros(2)
        u_nb[0] = normalU * normalBoundary[0]
        u_nb[1] = normalU * normalBoundary[1]
        if phase is True:
            # rho_nb = rhoSum / (denominator + 1e-17)
            equilibriumFunc(f_eq[ind], u_nb, p[nbList[itr]],
                            *equilibriumArgs)
        else:
            equilibriumFunc(f_eq[ind], u_nb, rho[nbList[itr]],
                            *equilibriumArgs)
        for direction in range(invDirections.shape[0]):
            f_new[ind, invDirections[direction]] = \
                f_eq[ind, invDirections[direction]] +\
                f_new[ind, outDirections[direction]] -\
                f_eq[ind, outDirections[direction]]
        secondOrderTensor = np.zeros((2, 2))
        # for x in range(2):
        #     for y in range(2):
        #         tensorSum = 0
        #         for k in range(noOfDirections):
        #             temp = 0
        #             for m in range(noOfDirections):
        #                 temp += preFactorFluid[k, m] *\
        #                     (f_new[ind, m] - f_eq[ind, m])
        #             tensorSum += c[k, x] * c[k, y] * temp
        #         secondOrderTensor[x, y] = tau_nu * tensorSum
        for x in range(2):
            for y in range(2):
                tensorSum = 0
                for k in range(noOfDirections):
                    tensorSum += c[k, x] * c[k, y] *\
                        (f_new[ind, k] - f_eq[ind, k])
                secondOrderTensor[x, y] = tensorSum
        for k in range(noOfDirections):
            f_new[ind, k] = f_eq[ind, k] + 0.5 * w[k] * cs_4 * \
                ((c[k, 0] * c[k, 0] - cs * cs) * secondOrderTensor[0, 0] +
                 c[k, 0] * c[k, 1] * secondOrderTensor[1, 0] +
                 c[k, 1] * c[k, 0] * secondOrderTensor[0, 1] +
                 (c[k, 1] * c[k, 1] - cs * cs) * secondOrderTensor[1, 1])
        # for k in range(noOfDirections):
        #     f_nonEq_sum = 0
        #     for m in range(noOfDirections):
        #         f_nonEq_sum += preFactorFluidInv[k, m] *\
        #             ((c[m, 0] * c[m, 0] - cs * cs) * secondOrderTensor[0, 0] +
        #              c[m, 0] * c[m, 1] * secondOrderTensor[1, 0] +
        #              c[m, 1] * c[m, 0] * secondOrderTensor[0, 1] +
        #              (c[m, 1] * c[m, 1] - cs * cs) * secondOrderTensor[1, 1])
        #     f_new[ind, k] = f_eq[ind, k] + 0.5 * w[k] * cs_4 * f_nonEq_sum


def bounceBack(f, f_new, solid, faceList, outDirections, invDirections):
    for ind in prange(faceList.shape[0]):
        if solid[faceList[ind], 0] != 1:
            for direction in range(invDirections.shape[0]):
                f_new[faceList[ind], invDirections[direction]] = \
                    f[faceList[ind], outDirections[direction]]


def zeroGradient(f, f_new, solid, faceList, outDirections, invDirections,
                 c, Nx, Ny):
    for itr in prange(faceList.shape[0]):
        ind = faceList[itr]
        if solid[ind, 0] != 1:
            i, j = int(ind / Ny), int(ind % Ny)
            for direction in invDirections:
                c_mag = int(np.ceil((c[direction, 0] * c[direction, 0]
                            + c[direction, 1] * c[direction, 1])))
                if c_mag == 1:
                    i_nb = int(i + c[direction, 0])
                    j_nb = int(j + c[direction, 1])
                    ind_nb = int(i_nb * Ny + j_nb)
                    break
            for k in range(f.shape[1]):
                f_new[ind, k] = f_new[ind_nb, k]


def periodic():
    pass
