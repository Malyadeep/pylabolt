import numpy as np


def circle(center, radius, solid, u, rho, rho_s, mesh, obsNo,
           velType='fixedTranslational', velValue=[0.0, 0.0],
           velOrigin=0, velOmega=0):
    obsNodes = []
    center_idx = np.int64(np.divide(center, mesh.delX))
    radius_idx = np.int64(radius/mesh.delX)
    origin_idx = np.int64(np.divide(velOrigin, mesh.delX))
    for i in range(mesh.Nx_global):
        for j in range(mesh.Ny_global):
            if ((i - center_idx[0])*(i - center_idx[0]) +
                    (j - center_idx[1])*(j - center_idx[1]) <=
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
    boundingBox_idx = np.int32(np.divide(boundingBox, mesh.delX))
    origin_idx = np.int64(np.divide(velOrigin, mesh.delX))
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


def circularConfinement(center, radius, solid, u, rho, rho_s, mesh, obsNo,
                        velType='fixedTranslational', velValue=[0.0, 0.0],
                        velOrigin=0, velOmega=0):
    obsNodes = []
    center_idx = np.int64(np.divide(center, mesh.delX))
    radius_idx = np.int64(radius/mesh.delX)
    origin_idx = np.int64(np.divide(velOrigin, mesh.delX))
    for i in range(mesh.Nx_global):
        for j in range(mesh.Ny_global):
            if ((i - center_idx[0])*(i - center_idx[0]) +
                    (j - center_idx[1])*(j - center_idx[1]) >=
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
