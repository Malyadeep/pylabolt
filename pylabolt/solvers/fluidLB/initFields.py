import numpy as np
import os

import pylabolt.base.setRegionFields as setRegionFields


class initialFields:
    def __init__(self, Nx, Ny, precision):
        self.u = np.zeros((Nx * Ny, 2),
                          dtype=precision)
        self.rho = np.zeros((Nx * Ny),
                            dtype=precision)


def setFields(fields, U_initial, rho_initial, mesh):
    for ind in range(mesh.Nx_global * mesh.Ny_global):
        fields.u[ind, 0] = U_initial[0]
        fields.u[ind, 1] = U_initial[1]
        fields.rho[ind] = rho_initial


def initFields(internalFields, mesh, precision, rank, comm):
    if rank == 0:
        fields = initialFields(mesh.Nx_global, mesh.Ny_global,
                               precision)
    internalFieldsKeys = list(internalFields.keys())
    try:
        u_initial = internalFields['default']['u']
        v_initial = internalFields['default']['v']
        U_initial = np.array([u_initial, v_initial],
                             dtype=precision)
        rho_initial = precision(internalFields['default']['rho'])
    except KeyError as e:
        if rank == 0:
            print('ERROR! Keyword ' + str(e) +
                  ' missing in internalFields')
        os._exit(1)

    if rank == 0:
        setFields(fields, U_initial, rho_initial, mesh)

    if len(internalFieldsKeys) == 1:
        if rank == 0:
            return fields
        else:
            return 0
    else:
        for key in internalFieldsKeys:
            if key == 'default':
                continue
            region = internalFields[key]
            try:
                regionType = region['type']
                if regionType == 'line':
                    if rank == 0:
                        status = setRegionFields.line(fields, mesh, precision,
                                                      region)
                        if status == 0:
                            comm.Abort(1)
                    comm.Barrier()
                else:
                    if rank == 0:
                        print('ERROR! Unsupported region type - ', regionType)
                    os._exit(1)
            except KeyError as e:
                if rank == 0:
                    print('ERROR! Keyword ' + str(e) +
                          ' missing in internalFields')
                os._exit(1)
    if rank == 0:
        return fields
    else:
        return 0
