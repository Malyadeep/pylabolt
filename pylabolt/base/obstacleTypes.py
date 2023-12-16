import numpy as np
import numba


def circle(center, radius, solid, initialFields, rho_s, mesh, obsNo,
           velType='fixedTranslational', velValue=[0.0, 0.0],
           velOrigin=0, velOmega=0, periodic=False):
    obsNodes = []
    center_idx = np.int64(np.divide(center, mesh.delX)) + \
        np.ones(2, dtype=np.int64)
    radius_idx = np.int64(radius / mesh.delX)
    origin_idx = np.int64(np.divide(velOrigin, mesh.delX)) + \
        np.ones(2, dtype=np.int64)
    for i in range(mesh.Nx_global):
        for j in range(mesh.Ny_global):
            cond1 = ((i - center_idx[0])*(i - center_idx[0]) +
                     (j - center_idx[1])*(j - center_idx[1]) <
                     radius_idx * radius_idx)
            cond2, cond3 = False, False
            cond4, cond5 = False, False
            cond6, cond7 = False, False
            cond8, cond9 = False, False
            if periodic is True:
                cond2 = ((i - center_idx[0] - mesh.Nx_global + 2) *
                         (i - center_idx[0] - mesh.Nx_global + 2) +
                         (j - center_idx[1]) *
                         (j - center_idx[1]) <
                         radius_idx * radius_idx)
                cond3 = ((i - center_idx[0] + mesh.Nx_global - 2) *
                         (i - center_idx[0] + mesh.Nx_global - 2) +
                         (j - center_idx[1]) *
                         (j - center_idx[1]) <
                         radius_idx * radius_idx)
                cond4 = ((i - center_idx[0]) *
                         (i - center_idx[0]) +
                         (j - center_idx[1] - mesh.Ny_global + 2) *
                         (j - center_idx[1] - mesh.Ny_global + 2) <
                         radius_idx * radius_idx)
                cond5 = ((i - center_idx[0]) *
                         (i - center_idx[0]) +
                         (j - center_idx[1] + mesh.Ny_global - 2) *
                         (j - center_idx[1] + mesh.Ny_global - 2) <
                         radius_idx * radius_idx)
                cond6 = ((i - center_idx[0] - mesh.Nx_global + 2) *
                         (i - center_idx[0] - mesh.Nx_global + 2) +
                         (j - center_idx[1] - mesh.Ny_global + 2) *
                         (j - center_idx[1] - mesh.Ny_global + 2) <
                         radius_idx * radius_idx)
                cond7 = ((i - center_idx[0] + mesh.Nx_global - 2) *
                         (i - center_idx[0] + mesh.Nx_global - 2) +
                         (j - center_idx[1] - mesh.Ny_global + 2) *
                         (j - center_idx[1] - mesh.Ny_global + 2) <
                         radius_idx * radius_idx)
                cond8 = ((i - center_idx[0] - mesh.Nx_global + 2) *
                         (i - center_idx[0] - mesh.Nx_global + 2) +
                         (j - center_idx[1] + mesh.Ny_global - 2) *
                         (j - center_idx[1] + mesh.Ny_global - 2) <
                         radius_idx * radius_idx)
                cond9 = ((i - center_idx[0] + mesh.Nx_global - 2) *
                         (i - center_idx[0] + mesh.Nx_global - 2) +
                         (j - center_idx[1] + mesh.Ny_global - 2) *
                         (j - center_idx[1] + mesh.Ny_global - 2) <
                         radius_idx * radius_idx)
            if (cond1 or cond2 or cond3 or cond4 or cond5 or cond6 or
                    cond7 or cond8 or cond9):
                ind = i * mesh.Ny_global + j
                obsNodes.append(ind)
                solid[ind, 0] = 1
                solid[ind, 1] = obsNo
                for field in initialFields.fieldList:
                    if field == 'rho':
                        initialFields.rho[ind] = rho_s
                    if field == 'u':
                        if velType == 'fixedTranslational':
                            initialFields.u[ind, 0] = velValue[0]
                            initialFields.u[ind, 1] = velValue[1]
                        elif (velType == 'fixedRotational' or
                                velType == 'calculatedRotational' or
                                velType == 'calculated'):
                            x = i - origin_idx[0]
                            y = j - origin_idx[1]
                            theta = np.arctan2(y, x)
                            r = np.sqrt(x**2 + y**2)
                            if velType == 'calculated':
                                initialFields.u[ind, 0] = velValue[0] - r *\
                                    velOmega * np.sin(theta)
                                initialFields.u[ind, 1] = velValue[1] + r *\
                                    velOmega * np.cos(theta)
                            else:
                                initialFields.u[ind, 0] = - r * velOmega *\
                                    np.sin(theta)
                                initialFields.u[ind, 1] = r * velOmega *\
                                    np.cos(theta)
                    if field == 'p':
                        initialFields.p[ind] = 0
                    if field == 'phi':
                        initialFields.phi[ind] = 0
    obsNodes = np.array(obsNodes, dtype=np.int64)
    volume = np.pi * radius_idx**2 * 1
    mass = rho_s * volume
    momentofInertia = mass * radius_idx**2 / 2
    return obsNodes, momentofInertia, mass, rho_s


def rectangle(boundingBox, solid, initialFields, rho_s, mesh, obsNo,
              velType='fixedTranslational', velValue=[0.0, 0.0],
              velOrigin=0, velOmega=0, periodic=False):
    obsNodes = []
    boundingBox_idx = np.int64(np.divide(boundingBox, mesh.delX)) +\
        np.ones((2, 2), dtype=np.int64)
    origin_idx = np.int64(np.divide(velOrigin, mesh.delX)) + \
        np.ones(2, dtype=np.int64)
    for i in range(mesh.Nx_global):
        for j in range(mesh.Ny_global):
            if (i >= boundingBox_idx[0, 0] and i <= boundingBox_idx[1, 0]
                    and j >= boundingBox_idx[0, 1] and
                    j <= boundingBox_idx[1, 1]):
                ind = i * mesh.Ny_global + j
                obsNodes.append(ind)
                solid[ind, 0] = 1
                solid[ind, 1] = obsNo
                for field in initialFields.fieldList:
                    if field == 'rho':
                        initialFields.rho[ind] = rho_s
                    if field == 'u':
                        if velType == 'fixedTranslational':
                            initialFields.u[ind, 0] = velValue[0]
                            initialFields.u[ind, 1] = velValue[1]
                        elif (velType == 'fixedRotational' or
                                velType == 'calculatedRotational'):
                            x = i - origin_idx[0]
                            y = j - origin_idx[1]
                            theta = np.arctan2(y, x)
                            r = np.sqrt(x**2 + y**2)
                            initialFields.u[ind, 0] = - r * velOmega *\
                                np.sin(theta)
                            initialFields.u[ind, 1] = r * velOmega *\
                                np.cos(theta)
                    if field == 'p':
                        initialFields.p[ind] = 0
                    if field == 'phi':
                        initialFields.phi[ind] = 0
    obsNodes = np.array(obsNodes, dtype=np.int64)
    dist_x = (np.abs(boundingBox_idx[1, 0] - boundingBox_idx[0, 0]))/mesh.delX
    dist_y = (np.abs(boundingBox_idx[1, 1] - boundingBox_idx[0, 1]))/mesh.delX
    volume = dist_x * dist_y * 1
    mass = rho_s * volume
    momentofInertia = mass * (dist_x**2 + dist_y**2) / 12
    return obsNodes, momentofInertia, mass, rho_s


def inclinedRectangle(centerLine, width, solid, initialFields,
                      rho_s, mesh, obsNo, comm, velType='fixedTranslational',
                      velValue=[0.0, 0.0], velOrigin=0, velOmega=0,
                      periodic=False):
    obsNodes = []
    centerLine_idx = np.int64(np.divide(centerLine, mesh.delX)) +\
        np.ones((2, 2), dtype=np.int64)
    origin_idx = np.int64(np.divide(velOrigin, mesh.delX)) + \
        np.ones(2, dtype=np.int64)
    Nx_i, Nx_f = centerLine_idx[0, 0], centerLine_idx[1, 0]
    Ny_i, Ny_f = centerLine_idx[0, 1], centerLine_idx[1, 1]
    if Nx_f - Nx_i == 0 or Ny_f - Ny_i == 0:
        print("ERROR! For vertical or horizontal rectangular obstacle" +
              " use 'rectangle' option!")
        comm.Abort(1)
    else:
        slope = (Ny_f - Ny_i)/(Nx_f - Nx_i)
        intercept = Ny_i - slope * Nx_i
        # print(slope, intercept)
        inclinationAngle = np.arctan(slope)
        sinAlpha = np.sin(inclinationAngle)
    for i in range(mesh.Nx_global):
        for j in range(mesh.Ny_global):
            dist = np.abs(slope * i - j + intercept) /\
                np.sqrt(slope * slope + 1)
            point = False
            if dist <= width:
                y = slope * i + intercept
                if j < y:
                    if (i - dist * sinAlpha <= Nx_f and
                            i - dist * sinAlpha >= Nx_i):
                        point = True
                elif j > y:
                    if (i + dist * sinAlpha <= Nx_f and
                            i + dist * sinAlpha >= Nx_i):
                        point = True
                elif j == y:
                    if (i <= Nx_f and i >= Nx_i):
                        point = True
            if point is True:
                ind = i * mesh.Ny_global + j
                obsNodes.append(ind)
                solid[ind, 0] = 1
                solid[ind, 1] = obsNo
                for field in initialFields.fieldList:
                    if field == 'rho':
                        initialFields.rho[ind] = rho_s
                    if field == 'u':
                        if velType == 'fixedTranslational':
                            initialFields.u[ind, 0] = velValue[0]
                            initialFields.u[ind, 1] = velValue[1]
                        elif (velType == 'fixedRotational' or
                                velType == 'calculatedRotational'):
                            x = i - origin_idx[0]
                            y = j - origin_idx[1]
                            theta = np.arctan2(y, x)
                            r = np.sqrt(x**2 + y**2)
                            initialFields.u[ind, 0] = - r * velOmega *\
                                np.sin(theta)
                            initialFields.u[ind, 1] = r * velOmega *\
                                np.cos(theta)
                    if field == 'p':
                        initialFields.p[ind] = 0
                    if field == 'phi':
                        initialFields.phi[ind] = 0
    obsNodes = np.array(obsNodes, dtype=np.int64)
    dist_x = np.sqrt((Nx_f - Nx_i) ** 2 + (Ny_f - Ny_i) ** 2)
    dist_y = width
    volume = dist_x * dist_y * 1
    mass = rho_s * volume
    momentofInertia = mass * (dist_x**2 + dist_y**2) / 12
    return obsNodes, momentofInertia, mass, rho_s


def circularConfinement(center, radius, solid, initialFields, rho_s, mesh, obsNo,
                        velType='fixedTranslational', velValue=[0.0, 0.0],
                        velOrigin=0, velOmega=0, periodic=True):
    obsNodes = []
    center_idx = np.int64(np.divide(center, mesh.delX)) + \
        np.ones(2, dtype=np.int64)
    radius_idx = np.int64(radius/mesh.delX)
    origin_idx = np.int64(np.divide(velOrigin, mesh.delX)) + \
        np.ones(2, dtype=np.int64)
    for i in range(mesh.Nx_global):
        for j in range(mesh.Ny_global):
            if ((i - center_idx[0])*(i - center_idx[0]) +
                    (j - center_idx[1])*(j - center_idx[1]) >
                    radius_idx*radius_idx):
                ind = i * mesh.Ny_global + j
                obsNodes.append(ind)
                solid[ind, 0] = 1
                solid[ind, 1] = obsNo
                for field in initialFields.fieldList:
                    if field == 'rho':
                        initialFields.rho[ind] = rho_s
                    if field == 'u':
                        if velType == 'fixedTranslational':
                            initialFields.u[ind, 0] = velValue[0]
                            initialFields.u[ind, 1] = velValue[1]
                        elif velType == 'fixedRotational':
                            x = i - origin_idx[0]
                            y = j - origin_idx[1]
                            theta = np.arctan2(y, x)
                            r = np.sqrt(x**2 + y**2)
                            initialFields.u[ind, 0] = - r * velOmega *\
                                np.sin(theta)
                            initialFields.u[ind, 1] = r * velOmega *\
                                np.cos(theta)
                    if field == 'p':
                        initialFields.p[ind] = 0
                    if field == 'phi':
                        initialFields.phi[ind] = 0
    obsNodes = np.array(obsNodes, dtype=np.int64)
    mass, momentOfInertia = 0, 0
    return obsNodes, momentOfInertia, mass, rho_s


@numba.njit
def reconstructCircle(i_global, j_global, Nx_global, Ny_global,
                      obsOrigin, x_periodic, y_periodic, radius,
                      center):
    x = i_global - obsOrigin[0]
    y = j_global - obsOrigin[1]
    leastDist = x * x + y * y
    cond1 = (leastDist < radius * radius)
    cond2, cond3 = False, False
    cond4, cond5 = False, False
    cond6, cond7 = False, False
    cond8, cond9 = False, False
    insideSolid = False
    if x_periodic is True and y_periodic is False:
        dist = ((i_global - obsOrigin[0] - Nx_global + 2) *
                (i_global - obsOrigin[0] - Nx_global + 2) +
                (j_global - obsOrigin[1]) *
                (j_global - obsOrigin[1]))
        cond2 = dist < radius * radius
        if cond2:
            insideSolid = True
            leastDist = dist
            x = i_global - obsOrigin[0] - Nx_global + 2
            y = j_global - obsOrigin[1]
        dist = ((i_global - obsOrigin[0] + Nx_global - 2) *
                (i_global - obsOrigin[0] + Nx_global - 2) +
                (j_global - obsOrigin[1]) *
                (j_global - obsOrigin[1]))
        cond3 = dist < radius * radius
        if cond3:
            insideSolid = True
            leastDist = dist
            x = i_global - obsOrigin[0] + Nx_global - 2
            y = j_global - obsOrigin[1]
        if cond1 or cond2 or cond3:
            insideSolid = True
    elif x_periodic is False and y_periodic is True:
        dist = ((i_global - obsOrigin[0]) *
                (i_global - obsOrigin[0]) +
                (j_global - obsOrigin[1] - Ny_global + 2) *
                (j_global - obsOrigin[1] - Ny_global + 2))
        cond2 = dist < radius * radius
        if cond2:
            insideSolid = True
            leastDist = dist
            x = i_global - obsOrigin[0]
            y = j_global - obsOrigin[1] - Ny_global + 2
        dist = ((i_global - obsOrigin[0]) *
                (i_global - obsOrigin[0]) +
                (j_global - obsOrigin[1] + Ny_global - 2) *
                (j_global - obsOrigin[1] + Ny_global - 2))
        cond3 = dist < radius * radius
        if cond3:
            insideSolid = True
            leastDist = dist
            x = i_global - obsOrigin[0]
            y = j_global - obsOrigin[1] + Ny_global - 2
        if cond1 or cond2 or cond3:
            insideSolid = True
    elif x_periodic is True and y_periodic is True:
        dist = ((i_global - obsOrigin[0] - Nx_global + 2) *
                (i_global - obsOrigin[0] - Nx_global + 2) +
                (j_global - obsOrigin[1]) *
                (j_global - obsOrigin[1]))
        cond2 = dist < radius * radius
        if cond2:
            insideSolid = True
            leastDist = dist
            x = i_global - obsOrigin[0] - Nx_global + 2
            y = j_global - obsOrigin[1]
        dist = ((i_global - obsOrigin[0] + Nx_global - 2) *
                (i_global - obsOrigin[0] + Nx_global - 2) +
                (j_global - obsOrigin[1]) *
                (j_global - obsOrigin[1]))
        cond3 = dist < radius * radius
        if cond3:
            insideSolid = True
            leastDist = dist
            x = i_global - obsOrigin[0] + Nx_global - 2
            y = j_global - obsOrigin[1]
        dist = ((i_global - obsOrigin[0]) *
                (i_global - obsOrigin[0]) +
                (j_global - obsOrigin[1] - Ny_global + 2) *
                (j_global - obsOrigin[1] - Ny_global + 2))
        cond4 = dist < radius * radius
        if cond4:
            insideSolid = True
            leastDist = dist
            x = i_global - obsOrigin[0]
            y = j_global - obsOrigin[1] - Ny_global + 2
        dist = ((i_global - obsOrigin[0]) *
                (i_global - obsOrigin[0]) +
                (j_global - obsOrigin[1] + Ny_global - 2) *
                (j_global - obsOrigin[1] + Ny_global - 2))
        cond5 = dist < radius * radius
        if cond5:
            insideSolid = True
            leastDist = dist
            x = i_global - obsOrigin[0]
            y = j_global - obsOrigin[1] + Ny_global - 2
        dist = ((i_global - obsOrigin[0] - Nx_global + 2) *
                (i_global - obsOrigin[0] - Nx_global + 2) +
                (j_global - obsOrigin[1] - Ny_global + 2) *
                (j_global - obsOrigin[1] - Ny_global + 2))
        cond6 = dist < radius * radius
        if cond6:
            insideSolid = True
            leastDist = dist
            x = i_global - obsOrigin[0] - Nx_global + 2
            y = j_global - obsOrigin[1] - Ny_global + 2
        dist = ((i_global - obsOrigin[0] + Nx_global - 2) *
                (i_global - obsOrigin[0] + Nx_global - 2) +
                (j_global - obsOrigin[1] - Ny_global + 2) *
                (j_global - obsOrigin[1] - Ny_global + 2))
        cond7 = dist < radius * radius
        if cond7:
            insideSolid = True
            leastDist = dist
            x = i_global - obsOrigin[0] + Nx_global - 2
            y = j_global - obsOrigin[1] - Ny_global + 2
        dist = ((i_global - obsOrigin[0] - Nx_global + 2) *
                (i_global - obsOrigin[0] - Nx_global + 2) +
                (j_global - obsOrigin[1] + Ny_global - 2) *
                (j_global - obsOrigin[1] + Ny_global - 2))
        cond8 = dist < radius * radius
        if cond8:
            insideSolid = True
            leastDist = dist
            x = i_global - obsOrigin[0] - Nx_global + 2
            y = j_global - obsOrigin[1] + Ny_global - 2
        dist = ((i_global - obsOrigin[0] + Nx_global - 2) *
                (i_global - obsOrigin[0] + Nx_global - 2) +
                (j_global - obsOrigin[1] + Ny_global - 2) *
                (j_global - obsOrigin[1] + Ny_global - 2))
        cond9 = dist < radius * radius
        if cond9:
            insideSolid = True
            leastDist = dist
            x = i_global - obsOrigin[0] + Nx_global - 2
            y = j_global - obsOrigin[1] + Ny_global - 2
        if (cond1 or cond2 or cond3 or cond4 or cond5 or cond6 or
                cond7 or cond8 or cond9):
            insideSolid = True
        # if i_global == 100 and j_global == 50:
        #     print(cond1, ', ', cond2, ', ', cond3, ', ', cond4,
        #           ', ', cond5, ', ', cond6, ', ', cond7, ', ', cond8,
        #           ', ', cond9)
    else:
        insideSolid = cond1
    return insideSolid, x, y, leastDist


@numba.njit
def reconstructRectangle(i_global, j_global, Nx_global, Ny_global,
                         obsOrigin, x_periodic, y_periodic, length,
                         breadth):
    x = i_global - obsOrigin[0]
    y = j_global - obsOrigin[1]
    leastDist = x * x + y * y
    cond1 = (i_global <= obsOrigin[0] + length / 2 and
             i_global >= obsOrigin[0] - length / 2 and
             j_global <= obsOrigin[1] + breadth / 2 and
             j_global >= obsOrigin[1] - breadth / 2)
    cond2, cond3 = False, False
    cond4, cond5 = False, False
    cond6, cond7 = False, False
    cond8, cond9 = False, False
    insideSolid = False
    if x_periodic is True and y_periodic is False:
        dist = ((i_global - obsOrigin[0] - Nx_global + 2) *
                (i_global - obsOrigin[0] - Nx_global + 2) +
                (j_global - obsOrigin[1]) *
                (j_global - obsOrigin[1]))
        cond2 = (i_global - Nx_global + 2 <= obsOrigin[0] + length / 2 and
                 i_global - Nx_global + 2 >= obsOrigin[0] - length / 2 and
                 j_global <= obsOrigin[1] + breadth / 2 and
                 j_global >= obsOrigin[1] - breadth / 2)
        if cond2:
            insideSolid = True
            leastDist = dist
            x = i_global - obsOrigin[0] - Nx_global + 2
            y = j_global - obsOrigin[1]
        dist = ((i_global - obsOrigin[0] + Nx_global - 2) *
                (i_global - obsOrigin[0] + Nx_global - 2) +
                (j_global - obsOrigin[1]) *
                (j_global - obsOrigin[1]))
        cond3 = (i_global + Nx_global - 2 <= obsOrigin[0] + length / 2 and
                 i_global + Nx_global - 2 >= obsOrigin[0] - length / 2 and
                 j_global <= obsOrigin[1] + breadth / 2 and
                 j_global >= obsOrigin[1] - breadth / 2)
        if cond3:
            insideSolid = True
            leastDist = dist
            x = i_global - obsOrigin[0] - Nx_global + 2
            y = j_global - obsOrigin[1]
        if cond1 or cond2 or cond3:
            insideSolid = True
    elif x_periodic is False and y_periodic is True:
        dist = ((i_global - obsOrigin[0]) *
                (i_global - obsOrigin[0]) +
                (j_global - obsOrigin[1] - Ny_global + 2) *
                (j_global - obsOrigin[1] - Ny_global + 2))
        cond2 = (i_global <= obsOrigin[0] + length / 2 and
                 i_global >= obsOrigin[0] - length / 2 and
                 j_global - Ny_global + 2 <= obsOrigin[1] + breadth / 2 and
                 j_global - Ny_global + 2 >= obsOrigin[1] - breadth / 2)
        if cond2:
            insideSolid = True
            leastDist = dist
            x = i_global - obsOrigin[0]
            y = j_global - obsOrigin[1] - Ny_global + 2
        dist = ((i_global - obsOrigin[0]) *
                (i_global - obsOrigin[0]) +
                (j_global - obsOrigin[1] + Ny_global - 2) *
                (j_global - obsOrigin[1] + Ny_global - 2))
        cond3 = (i_global <= obsOrigin[0] + length / 2 and
                 i_global >= obsOrigin[0] - length / 2 and
                 j_global + Ny_global - 2 <= obsOrigin[1] + breadth / 2 and
                 j_global + Ny_global - 2 >= obsOrigin[1] - breadth / 2)
        if cond3:
            insideSolid = True
            leastDist = dist
            x = i_global - obsOrigin[0]
            y = j_global - obsOrigin[1] + Ny_global - 2
        if cond1 or cond2 or cond3:
            insideSolid = True
    elif x_periodic is True and y_periodic is True:
        dist = ((i_global - obsOrigin[0] - Nx_global + 2) *
                (i_global - obsOrigin[0] - Nx_global + 2) +
                (j_global - obsOrigin[1]) *
                (j_global - obsOrigin[1]))
        cond2 = (i_global - Nx_global + 2 <= obsOrigin[0] + length / 2 and
                 i_global - Nx_global + 2 >= obsOrigin[0] - length / 2 and
                 j_global <= obsOrigin[1] + breadth / 2 and
                 j_global >= obsOrigin[1] - breadth / 2)
        if cond2:
            insideSolid = True
            leastDist = dist
            x = i_global - obsOrigin[0] - Nx_global + 2
            y = j_global - obsOrigin[1]
        dist = ((i_global - obsOrigin[0] + Nx_global - 2) *
                (i_global - obsOrigin[0] + Nx_global - 2) +
                (j_global - obsOrigin[1]) *
                (j_global - obsOrigin[1]))
        cond3 = (i_global + Nx_global - 2 <= obsOrigin[0] + length / 2 and
                 i_global + Nx_global - 2 >= obsOrigin[0] - length / 2 and
                 j_global <= obsOrigin[1] + breadth / 2 and
                 j_global >= obsOrigin[1] - breadth / 2)
        if cond3:
            insideSolid = True
            leastDist = dist
            x = i_global - obsOrigin[0] + Nx_global - 2
            y = j_global - obsOrigin[1]
        dist = ((i_global - obsOrigin[0]) *
                (i_global - obsOrigin[0]) +
                (j_global - obsOrigin[1] - Ny_global + 2) *
                (j_global - obsOrigin[1] - Ny_global + 2))
        cond4 = (i_global <= obsOrigin[0] + length / 2 and
                 i_global >= obsOrigin[0] - length / 2 and
                 j_global - Ny_global + 2 <= obsOrigin[1] + breadth / 2 and
                 j_global - Ny_global + 2 >= obsOrigin[1] - breadth / 2)
        if cond4:
            insideSolid = True
            leastDist = dist
            x = i_global - obsOrigin[0]
            y = j_global - obsOrigin[1] - Ny_global + 2
        dist = ((i_global - obsOrigin[0]) *
                (i_global - obsOrigin[0]) +
                (j_global - obsOrigin[1] + Ny_global - 2) *
                (j_global - obsOrigin[1] + Ny_global - 2))
        cond5 = (i_global <= obsOrigin[0] + length / 2 and
                 i_global >= obsOrigin[0] - length / 2 and
                 j_global + Ny_global - 2 <= obsOrigin[1] + breadth / 2 and
                 j_global + Ny_global - 2 >= obsOrigin[1] - breadth / 2)
        if cond5:
            insideSolid = True
            leastDist = dist
            x = i_global - obsOrigin[0]
            y = j_global - obsOrigin[1] + Ny_global - 2
        dist = ((i_global - obsOrigin[0] - Nx_global + 2) *
                (i_global - obsOrigin[0] - Nx_global + 2) +
                (j_global - obsOrigin[1] - Ny_global + 2) *
                (j_global - obsOrigin[1] - Ny_global + 2))
        cond6 = (i_global - Nx_global + 2 <= obsOrigin[0] + length / 2 and
                 i_global - Nx_global + 2 >= obsOrigin[0] - length / 2 and
                 j_global - Ny_global + 2 <= obsOrigin[1] + breadth / 2 and
                 j_global - Ny_global + 2 >= obsOrigin[1] - breadth / 2)
        if cond6:
            insideSolid = True
            leastDist = dist
            x = i_global - obsOrigin[0] - Nx_global + 2
            y = j_global - obsOrigin[1] - Ny_global + 2
        dist = ((i_global - obsOrigin[0] + Nx_global - 2) *
                (i_global - obsOrigin[0] + Nx_global - 2) +
                (j_global - obsOrigin[1] - Ny_global + 2) *
                (j_global - obsOrigin[1] - Ny_global + 2))
        cond7 = (i_global + Nx_global - 2 <= obsOrigin[0] + length / 2 and
                 i_global + Nx_global - 2 >= obsOrigin[0] - length / 2 and
                 j_global - Ny_global + 2 <= obsOrigin[1] + breadth / 2 and
                 j_global - Ny_global + 2 >= obsOrigin[1] - breadth / 2)
        if cond7:
            insideSolid = True
            leastDist = dist
            x = i_global - obsOrigin[0] + Nx_global - 2
            y = j_global - obsOrigin[1] - Ny_global + 2
        dist = ((i_global - obsOrigin[0] - Nx_global + 2) *
                (i_global - obsOrigin[0] - Nx_global + 2) +
                (j_global - obsOrigin[1] + Ny_global - 2) *
                (j_global - obsOrigin[1] + Ny_global - 2))
        cond8 = (i_global - Nx_global + 2 <= obsOrigin[0] + length / 2 and
                 i_global - Nx_global + 2 >= obsOrigin[0] - length / 2 and
                 j_global + Ny_global - 2 <= obsOrigin[1] + breadth / 2 and
                 j_global + Ny_global - 2 >= obsOrigin[1] - breadth / 2)
        if cond8:
            insideSolid = True
            leastDist = dist
            x = i_global - obsOrigin[0] - Nx_global + 2
            y = j_global - obsOrigin[1] + Ny_global - 2
        dist = ((i_global - obsOrigin[0] + Nx_global - 2) *
                (i_global - obsOrigin[0] + Nx_global - 2) +
                (j_global - obsOrigin[1] + Ny_global - 2) *
                (j_global - obsOrigin[1] + Ny_global - 2))
        cond9 = (i_global + Nx_global - 2 <= obsOrigin[0] + length / 2 and
                 i_global + Nx_global - 2 >= obsOrigin[0] - length / 2 and
                 j_global + Ny_global - 2 <= obsOrigin[1] + breadth / 2 and
                 j_global + Ny_global - 2 >= obsOrigin[1] - breadth / 2)
        if cond9:
            insideSolid = True
            leastDist = dist
            x = i_global - obsOrigin[0] + Nx_global - 2
            y = j_global - obsOrigin[1] + Ny_global - 2
        if (cond1 or cond2 or cond3 or cond4 or cond5 or cond6 or
                cond7 or cond8 or cond9):
            insideSolid = True
        # if i_global == 100 and j_global == 50:
        #     print(cond1, ', ', cond2, ', ', cond3, ', ', cond4,
        #           ', ', cond5, ', ', cond6, ', ', cond7, ', ', cond8,
        #           ', ', cond9)
    else:
        insideSolid = cond1
    return insideSolid, x, y, leastDist
