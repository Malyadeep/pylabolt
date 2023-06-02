import numpy as np


def line(fields, mesh, precision, region):
    try:
        point_0 = region['point_0']
        point_1 = region['point_1']
        if not isinstance(point_0, list):
            print('ERROR! coordinates in point_0 must be list [x, y]')
            print('Cannot set fields in region for region type line\n')
            return 0
        if not isinstance(point_1, list):
            print('ERROR! coordinates in point_1 must be list [x, y]')
            print('Cannot set fields in region for region type line\n')
            return 0
        Nx_i, Nx_f = int(point_0[0]/mesh.delX), int(point_1[0]/mesh.delX)
        Ny_i, Ny_f = int(point_0[1]/mesh.delX), int(point_1[1]/mesh.delX)
        u_initial = region['fields']['u']
        v_initial = region['fields']['v']
        U_initial = np.array([u_initial, v_initial],
                             dtype=precision)
        rho_initial = precision(region['fields']['rho'])
    except KeyError as e:
        print('ERROR! Keyword ' + str(e) +
              ' missing in internalFields')
        return 0
    if Nx_f - Nx_i == 0 and Ny_f - Ny_i != 0:
        for j in range(Ny_i, Ny_f + 1):
            ind = Nx_i * mesh.Ny_global + j
            fields.u[ind, 0] = U_initial[0]
            fields.u[ind, 1] = U_initial[1]
            fields.rho[ind] = rho_initial
    elif Ny_f - Ny_i == 0 and Nx_f - Nx_i != 0:
        for i in range(Nx_i, Nx_f + 1):
            ind = i * mesh.Ny_global + Ny_i
            fields.u[ind, 0] = U_initial[0]
            fields.u[ind, 1] = U_initial[1]
            fields.rho[ind] = rho_initial
    elif Ny_f - Ny_i == 0 and Nx_f - Nx_i == 0:
        ind = Nx_i * mesh.Ny_global + Ny_i
        fields.u[ind, 0] = U_initial[0]
        fields.u[ind, 1] = U_initial[1]
        fields.rho[ind] = rho_initial
    else:
        slope = (Ny_f - Ny_i)/(Nx_f - Nx_i)
        intercept = Ny_i - slope * Nx_i
        for i in range(Nx_i, Nx_f + 1):
            j = int(slope * i + intercept)
            ind = i * mesh.Ny_global + j
            fields.u[ind, 0] = U_initial[0]
            fields.u[ind, 1] = U_initial[1]
            fields.rho[ind] = rho_initial
    return 1
