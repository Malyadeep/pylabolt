import numpy as np


def circle(center, radius, solid, u, rho, rho_s, mesh, obsNo,
           velType='fixedTranslational', velValue=[0.0, 0.0],
           velOrigin=0, velOmega=0):
    obsNodes = []
    center_idx = np.int64(np.divide(center, mesh.delX)) + \
        np.ones(2, dtype=np.int64)
    radius_idx = np.int64(radius/mesh.delX)
    origin_idx = np.int64(np.divide(velOrigin, mesh.delX)) + \
        np.ones(2, dtype=np.int64)
    for i in range(mesh.Nx_global):
        for j in range(mesh.Ny_global):
            if ((i - center_idx[0])*(i - center_idx[0]) +
                    (j - center_idx[1])*(j - center_idx[1]) <
                    radius_idx*radius_idx):
                ind = i * mesh.Ny_global + j
                obsNodes.append(ind)
                solid[ind, 0] = 1
                solid[ind, 1] = obsNo
                rho[ind] = rho_s
                if velType == 'fixedTranslational':
                    u[ind, 0] = velValue[0]
                    u[ind, 1] = velValue[1]
                elif (velType == 'fixedRotational' or
                      velType == 'calculatedRotational'):
                    x = i - origin_idx[0]
                    y = j - origin_idx[1]
                    theta = np.arctan2(y, x)
                    r = np.sqrt(x**2 + y**2)
                    u[ind, 0] = - r * velOmega * np.sin(theta)
                    u[ind, 1] = r * velOmega * np.cos(theta)
    obsNodes = np.array(obsNodes, dtype=np.int64)
    volume = np.pi * radius_idx**2 * 1
    mass = rho_s * volume
    momentofInertia = mass * radius_idx**2 / 2
    return obsNodes, momentofInertia


def rectangle(boundingBox, solid, u, rho, rho_s, mesh, obsNo,
              velType='fixedTranslational', velValue=[0.0, 0.0],
              velOrigin=0, velOmega=0):
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
                rho[ind] = rho_s
                if velType == 'fixedTranslational':
                    u[ind, 0] = velValue[0]
                    u[ind, 1] = velValue[1]
                elif (velType == 'fixedRotational' or
                      velType == 'calculatedRotational'):
                    x = i - origin_idx[0]
                    y = j - origin_idx[1]
                    theta = np.arctan2(y, x)
                    r = np.sqrt(x**2 + y**2)
                    u[ind, 0] = - r * velOmega * np.sin(theta)
                    u[ind, 1] = r * velOmega * np.cos(theta)
    obsNodes = np.array(obsNodes, dtype=np.int64)
    dist_x = (np.abs(boundingBox_idx[1, 0] - boundingBox_idx[0, 0]))/mesh.delX
    dist_y = (np.abs(boundingBox_idx[1, 1] - boundingBox_idx[0, 1]))/mesh.delX
    volume = dist_x * dist_y * 1
    mass = rho_s * volume
    momentofInertia = mass * (dist_x**2 + dist_y**2) / 12
    return obsNodes, momentofInertia


def inclinedRectangle(centerLine, width, solid, u, rho,
                      rho_s, mesh, obsNo, comm, velType='fixedTranslational',
                      velValue=[0.0, 0.0], velOrigin=0, velOmega=0):
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
                rho[ind] = rho_s
                if velType == 'fixedTranslational':
                    u[ind, 0] = velValue[0]
                    u[ind, 1] = velValue[1]
                elif (velType == 'fixedRotational' or
                      velType == 'calculatedRotational'):
                    x = i - origin_idx[0]
                    y = j - origin_idx[1]
                    theta = np.arctan2(y, x)
                    r = np.sqrt(x**2 + y**2)
                    u[ind, 0] = - r * velOmega * np.sin(theta)
                    u[ind, 1] = r * velOmega * np.cos(theta)
    obsNodes = np.array(obsNodes, dtype=np.int64)
    dist_x = np.sqrt((Nx_f - Nx_i) ** 2 + (Ny_f - Ny_i) ** 2)
    dist_y = width
    volume = dist_x * dist_y * 1
    mass = rho_s * volume
    momentofInertia = mass * (dist_x**2 + dist_y**2) / 12
    return obsNodes, momentofInertia


def circularConfinement(center, radius, solid, u, rho, rho_s, mesh, obsNo,
                        velType='fixedTranslational', velValue=[0.0, 0.0],
                        velOrigin=0, velOmega=0):
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
                rho[ind] = rho_s
                if velType == 'fixedTranslational':
                    u[ind, 0] = velValue[0]
                    u[ind, 1] = velValue[1]
                elif velType == 'fixedRotational':
                    x = i - origin_idx[0]
                    y = j - origin_idx[1]
                    theta = np.arctan2(y, x)
                    r = np.sqrt(x**2 + y**2)
                    u[ind, 0] = - r * velOmega * np.sin(theta)
                    u[ind, 1] = r * velOmega * np.cos(theta)
    obsNodes = np.array(obsNodes, dtype=np.int64)
    momentOfInertia = 0
    return obsNodes, momentOfInertia
