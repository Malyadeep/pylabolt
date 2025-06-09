import numpy as np
import os

from pylabolt.base.setRegionFields import initializeFields


class initialFields:
    def __init__(self, Nx, Ny, precision):
        self.fieldList = ['u', 'rho', 'p', 'phi', 'boundaryNode']
        self.defaultU = np.zeros(2, dtype=precision)
        self.defaultRho = 1.
        self.defaultPhi = 0.
        self.u = np.zeros((Nx * Ny, 2),
                          dtype=precision)
        self.rho = np.zeros((Nx * Ny),
                            dtype=precision)
        self.p = np.zeros((Nx * Ny),
                          dtype=precision)
        self.phi = np.zeros((Nx * Ny),
                            dtype=precision)
        self.boundaryNode = np.zeros((Nx * Ny),
                                     dtype=np.int32)


def setFields(fields, rho_initial, phi_initial, mesh, forcingScheme,
              lattice, velType, velValue=[0., 0.], velOmega=0., x_ref=[0, 0],
              pressure_initialization=False, direction='y',
              pressureRefValue=0, pressureRefHeight=0):
    if velType == 'translational':
        for ind in range(mesh.Nx_global * mesh.Ny_global):
            fields.u[ind, 0] = velValue[0]
            fields.u[ind, 1] = velValue[1]
            fields.rho[ind] = rho_initial
            fields.phi[ind] = phi_initial
            if pressure_initialization is True:
                i, j = int(ind / mesh.Ny_global), int(ind % mesh.Ny_global)
                if direction == 'y':
                    fields.p[ind] = (j - pressureRefHeight) *\
                        forcingScheme.gravity[1] * lattice.cs_2 -\
                        0.5 * (-pressureRefHeight) * forcingScheme.gravity[1]\
                        * lattice.cs_2
                elif direction == 'x':
                    fields.p[ind] = (i - pressureRefHeight) *\
                        forcingScheme.gravity[0] * lattice.cs_2 -\
                        0.5 * (-pressureRefHeight) * forcingScheme.gravity[0]\
                        * lattice.cs_2
            else:
                fields.p[ind] = 0
    elif velType == 'rotational':
        for i in range(mesh.Nx_global):
            for j in range(mesh.Ny_global):
                ind = int(i * mesh.Ny_global + j)
                x = i - x_ref[0]
                y = j - x_ref[1]
                r = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x)
                fields.u[ind, 0] = -r * velOmega * np.sin(theta)
                fields.u[ind, 1] = r * velOmega * np.cos(theta)
                fields.rho[ind] = rho_initial
                fields.phi[ind] = phi_initial
                if pressure_initialization is True:
                    if direction == 'y':
                        fields.p[ind] = (j - pressureRefHeight) *\
                            forcingScheme.gravity[1] * rho_initial
                    elif direction == 'x':
                        fields.p[ind] = (i - pressureRefHeight) *\
                            forcingScheme.gravity[0] * rho_initial
                else:
                    fields.p[ind] = 0


def readFields(internalFields, mesh, lattice, forcingScheme, precision, rank,
               comm, phaseField):
    if rank == 0:
        fields = initialFields(mesh.Nx_global, mesh.Ny_global,
                               precision)
    try:
        velDict = internalFields['default']['U']
        if velDict['type'] == 'translational':
            velType = 'translational'
            u_initial = velDict['value']
            if not isinstance(u_initial, list):
                print('ERROR! velocity must be a list of components [x, y]')
                os._exit(1)
            else:
                u_initial = np.array(u_initial, dtype=precision)
        elif velDict['type'] == 'rotational':
            velType = 'rotational'
            omega_initial = velDict['omega']
            x_ref = velDict['x_ref']
            if not isinstance(omega_initial, float):
                print('ERROR! angular velocity must be float')
                os._exit(1)
            if not isinstance(x_ref, list):
                print('ERROR! reference point must be a list of ' +
                      'coordinates [x, y]')
                os._exit(1)
            omega_initial = precision(omega_initial)
            x_ref = np.array(x_ref, dtype=np.int64)
            x_ref_idx = np.int64(np.divide(x_ref, mesh.delX)) +\
                np.ones(2, dtype=np.int64)
        else:
            print("ERROR!")
            print("Unsupported velocity initialization!", flush=True)
            os._exit(1)
        rho_initial = precision(internalFields['default']['rho'])
        phi_initial = precision(internalFields['default']['phi'])
        try:
            pressureDict = internalFields['default']['p']
            pressure_initialization = True
        except KeyError:
            pressure_initialization = False
            print("Pressure initialized to zero everywhere")
        if pressure_initialization is True:
            try:
                pressureType = pressureDict['type']
                if pressureType == 'hydrostatic':
                    direction = pressureDict['direction']
                    if direction != 'x' and direction != 'y':
                        if rank == 0:
                            print("ERROR! 'direction' can be either " +
                                  "'x' or 'y'!")
                        os._exit(1)
                    pressureRefValue = pressureDict['ref_value']
                    pressureRefHeight = pressureDict['ref_height']
                    if ((not isinstance(pressureRefValue, float) and
                            not isinstance(pressureRefValue, int)) or
                            (not isinstance(pressureRefHeight, float) and
                             not isinstance(pressureRefHeight, int))):
                        if rank == 0:
                            print("ERROR! 'ref_value' and 'ref_height' " +
                                  "must be 'int' or 'float' type!")
                        os._exit(1)
                else:
                    if rank == 0:
                        print('ERROR! Unidentified type of pressure ' +
                              'initialization!')
                    os._exit(1)
            except KeyError as e:
                if rank == 0:
                    print("ERROR! Keyword " + str(e) +
                          " missing in 'internalFields' - 'p'")
                os._exit(1)
    except KeyError as e:
        if rank == 0:
            print('ERROR! Keyword ' + str(e) +
                  ' missing in internalFields')
        os._exit(1)

    if rank == 0:
        if pressure_initialization is False:
            direction = 'y'
            pressureRefHeight = 0
            pressureRefValue = 0
        if velType == 'translational':
            setFields(fields, rho_initial, phi_initial, mesh, forcingScheme,
                      lattice, velType, velValue=u_initial,
                      pressure_initialization=pressure_initialization,
                      direction=direction, pressureRefValue=pressureRefValue,
                      pressureRefHeight=pressureRefHeight)
        elif velType == 'rotational':
            setFields(fields, rho_initial, phi_initial, mesh, forcingScheme,
                      lattice, velType, velOmega=omega_initial,
                      x_ref=x_ref_idx,
                      pressure_initialization=pressure_initialization,
                      direction=direction, pressureRefValue=pressureRefValue,
                      pressureRefHeight=pressureRefHeight)

    if rank == 0:
        initializeFields(internalFields, fields, mesh, precision, comm)

    if rank == 0:
        return fields
    else:
        return 0
