import numpy as np
import os


def initializeFields(internalFields, fields, mesh, precision, comm,
                     phaseField):
    internalFieldsKeys = list(internalFields.keys())
    if len(internalFieldsKeys) == 1:
        return fields
    else:
        for key in internalFieldsKeys:
            if key == 'default':
                continue
            region = internalFields[key]
            try:
                regionType = region['type']
                if regionType == 'line':
                    status = line(fields, mesh, precision, region)
                    if status == 0:
                        comm.Abort(1)
                elif regionType == 'circle':
                    status = circle(fields, mesh, precision, region,
                                    phaseField)
                    if status == 0:
                        comm.Abort(1)
                elif regionType == 'rectangle':
                    status = rectangle(fields, mesh, precision, region)
                    if status == 0:
                        comm.Abort(1)
                else:
                    print('ERROR! Unsupported region type - ', regionType)
                    comm.Abort(1)
            except KeyError as e:
                print('ERROR! Keyword ' + str(e) +
                      ' missing in internalFields')
                comm.Abort(1)


def setFields(fields, ind, fieldsInitial, mesh, velType, x_ref=[1, 1],
              eta=None):
    for itr, field in enumerate(fields.fieldList):
        if field == 'u' and velType == 'translational':
            fields.u[ind, 0] = fieldsInitial[itr][0]
            fields.u[ind, 1] = fieldsInitial[itr][1]
        elif field == 'u' and velType == 'rotational':
            i, j = int(ind / mesh.Ny_global), int(ind % mesh.Ny_global)
            x = i - x_ref[0]
            y = j - x_ref[1]
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            fields.u[ind, 0] = -r * fieldsInitial[itr] * np.sin(theta)
            fields.u[ind, 1] = r * fieldsInitial[itr] * np.cos(theta)
        if field == 'rho':
            fields.rho[ind] = fieldsInitial[itr]
        if field == 'phi':
            fields.phi[ind] = fieldsInitial[itr]
            # fields.phi[ind] = 0.5 * (1 + np.tanh(2 * eta))


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
        Nx_i = int(point_0[0]/mesh.delX) + 1
        Nx_f = int(point_1[0]/mesh.delX) + 1
        Ny_i = int(point_0[1]/mesh.delX) + 1
        Ny_f = int(point_1[1]/mesh.delX) + 1
        fieldsInitial = []
        for field in fields.fieldList:
            if field == 'u':
                x_ref_idx = np.ones(2, dtype=np.int64)
                velDict = region['fields']['U']
                if velDict['type'] == 'translational':
                    velType = 'translational'
                    u_initial = velDict['value']
                    if not isinstance(u_initial, list):
                        print('ERROR! velocity must be a list of ' +
                              'components [x, y]')
                        os._exit(1)
                    else:
                        initial = np.array(u_initial, dtype=precision)
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
                    initial = precision(omega_initial)
                    x_ref = np.array(x_ref, dtype=precision)
                    x_ref_idx = np.int64(np.divide(x_ref, mesh.delX)) +\
                        np.ones(2, dtype=np.int64)
                else:
                    print("ERROR!")
                    print("Unsupported velocity initialization!", flush=True)
                    os._exit(1)
            if field == 'rho':
                initial = precision(region['fields']['rho'])
            if field == 'phi':
                initial = precision(region['fields']['phi'])
            fieldsInitial.append(initial)
    except KeyError as e:
        print('ERROR! Keyword ' + str(e) +
              ' missing in internalFields')
        return 0
    if Nx_f - Nx_i == 0 and Ny_f - Ny_i != 0:
        for j in range(Ny_i, Ny_f + 1):
            ind = Nx_i * mesh.Ny_global + j
            setFields(fields, ind, fieldsInitial, mesh, velType,
                      x_ref=x_ref_idx)
    elif Ny_f - Ny_i == 0 and Nx_f - Nx_i != 0:
        for i in range(Nx_i, Nx_f + 1):
            ind = i * mesh.Ny_global + Ny_i
            setFields(fields, ind, fieldsInitial, mesh, velType,
                      x_ref=x_ref_idx)
    elif Ny_f - Ny_i == 0 and Nx_f - Nx_i == 0:
        ind = Nx_i * mesh.Ny_global + Ny_i
        setFields(fields, ind, fieldsInitial, mesh, velType,
                  x_ref=x_ref_idx)
    else:
        slope = (Ny_f - Ny_i)/(Nx_f - Nx_i)
        intercept = Ny_i - slope * Nx_i
        for i in range(Nx_i, Nx_f + 1):
            j = int(slope * i + intercept)
            setFields(fields, ind, fieldsInitial, mesh, velType,
                      x_ref=x_ref_idx)
    return 1


def circle(fields, mesh, precision, region, phaseField):
    try:
        center = region['center']
        radius = region['radius']
        if not isinstance(center, list):
            print('ERROR! coordinates in point_0 must be list [x, y]')
            print('Cannot set fields in region for region type circle\n')
            return 0
        if not isinstance(radius, float):
            print('ERROR! radius must be float for region type circle')
            print('Cannot set fields in region for region type circle\n')
            return 0
        center_idx = np.int64(np.array(center)/mesh.delX) +\
            np.ones(2, dtype=np.int64)
        radius_idx = radius/mesh.delX
        circleType = region['circleType']
        if circleType == 'half':
            point_0 = region['diameterPoint_0']
            point_1 = region['diameterPoint_1']
            activeDomain = region['activeDomain']
            if activeDomain != '+' and activeDomain != '-':
                print("ERROR! activeDomain must be '+' or '-'!")
                return 0
            if not isinstance(point_0, list):
                print('ERROR! coordinates in diameterPoint_0 must be list')
                print('Cannot set fields in region for region type circle\n')
                return 0
            if not isinstance(point_1, list):
                print('ERROR! coordinates in diameterPoint_1 must be list')
                print('Cannot set fields in region for region type circle\n')
                return 0
            Nx_i = int(point_0[0]/mesh.delX) + 1
            Nx_f = int(point_1[0]/mesh.delX) + 1
            Ny_i = int(point_0[1]/mesh.delX) + 1
            Ny_f = int(point_1[1]/mesh.delX) + 1
            if Ny_f - Ny_i == 0 and Nx_f - Nx_i == 0:
                print('ERROR! diameter cannot be a point')
            if Ny_f - Ny_i != 0 and Nx_f - Nx_i != 0:
                slope = (Ny_f - Ny_i)/(Nx_f - Nx_i)
                intercept = Ny_i - slope * Nx_i
        fieldsInitial = []
        for field in fields.fieldList:
            if field == 'u':
                x_ref_idx = np.ones(2, dtype=np.int64)
                velDict = region['fields']['U']
                if velDict['type'] == 'translational':
                    velType = 'translational'
                    u_initial = velDict['value']
                    if not isinstance(u_initial, list):
                        print('ERROR! velocity must be a list of ' +
                              'components [x, y]')
                        os._exit(1)
                    else:
                        initial = np.array(u_initial, dtype=precision)
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
                    initial = precision(omega_initial)
                    x_ref = np.array(x_ref, dtype=precision)
                    x_ref_idx = np.int64(np.divide(x_ref, mesh.delX)) +\
                        np.ones(2, dtype=np.int64)
                else:
                    print("ERROR!")
                    print("Unsupported velocity initialization!", flush=True)
                    os._exit(1)
            if field == 'rho':
                initial = precision(region['fields']['rho'])
            if field == 'phi':
                initial = precision(region['fields']['phi'])
            fieldsInitial.append(initial)
    except KeyError as e:
        print('ERROR! Keyword ' + str(e) +
              ' missing in internalFields')
        return 0
    if circleType == 'full':
        for i in range(mesh.Nx_global):
            for j in range(mesh.Ny_global):
                if (np.sqrt((i - center_idx[0])*(i - center_idx[0]) +
                            (j - center_idx[1])*(j - center_idx[1])) <=
                        radius_idx):
                    # distFromCenter = np.sqrt((i - center_idx[0]) *
                    #                         (i - center_idx[0]) +
                    #                         (j - center_idx[1]) *
                    #                         (j - center_idx[1]))
                    # eta = (distFromCenter - radius_idx) / \
                    #     phaseField.interfaceWidth
                    ind = i * mesh.Ny_global + j
                    setFields(fields, ind, fieldsInitial, mesh, velType,
                            x_ref=x_ref_idx, eta=None)
    elif circleType == 'half':
        for i in range(mesh.Nx_global):
            for j in range(mesh.Ny_global):
                if ((i - center_idx[0])*(i - center_idx[0]) +
                        (j - center_idx[1])*(j - center_idx[1]) <
                        radius_idx*radius_idx):
                    if Nx_f - Nx_i == 0 and Ny_f - Ny_i != 0:
                        if activeDomain == '+' and i >= Nx_f:
                            ind = i * mesh.Ny_global + j
                            setFields(fields, ind, fieldsInitial, mesh,
                                      velType, x_ref=x_ref_idx)
                        elif activeDomain == '-' and i <= Nx_f:
                            ind = i * mesh.Ny_global + j
                            setFields(fields, ind, fieldsInitial, mesh,
                                      velType, x_ref=x_ref_idx)
                    elif Ny_f - Ny_i == 0 and Nx_f - Nx_i != 0:
                        if activeDomain == '+' and j >= Ny_f:
                            ind = i * mesh.Ny_global + j
                            setFields(fields, ind, fieldsInitial, mesh,
                                      velType, x_ref=x_ref_idx)
                        elif activeDomain == '-' and j <= Ny_f:
                            ind = i * mesh.Ny_global + j
                            setFields(fields, ind, fieldsInitial, mesh,
                                      velType, x_ref=x_ref_idx)
                    else:
                        j_ref = int(slope * i + intercept)
                        if activeDomain == '+' and j >= j_ref:
                            ind = i * mesh.Ny_global + j
                            setFields(fields, ind, fieldsInitial, mesh,
                                      velType, x_ref=x_ref_idx)
                        elif activeDomain == '-' and j <= j_ref:
                            ind = i * mesh.Ny_global + j
                            setFields(fields, ind, fieldsInitial, mesh,
                                      velType, x_ref=x_ref_idx)
    return 1


def rectangle(fields, mesh, precision, region):
    try:
        boundingBox = region['boundingBox']
        if not isinstance(boundingBox, list):
            print('ERROR! For rectangle, boundingBox must contain ' +
                  'coordinates of the diagonal as a list')
            print('Cannot set fields in region for region type rectangle\n')
            return 0
        boundingBox = np.array(boundingBox, dtype=np.float64)
        boundingBox_idx = np.int64(np.divide(boundingBox, mesh.delX)) +\
            np.ones((2, 2), dtype=np.int64)
        fieldsInitial = []
        for field in fields.fieldList:
            if field == 'u':
                x_ref_idx = np.ones(2, dtype=np.int64)
                velDict = region['fields']['U']
                if velDict['type'] == 'translational':
                    velType = 'translational'
                    u_initial = velDict['value']
                    if not isinstance(u_initial, list):
                        print('ERROR! velocity must be a list of ' +
                              'components [x, y]')
                        os._exit(1)
                    else:
                        initial = np.array(u_initial, dtype=precision)
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
                    initial = precision(omega_initial)
                    x_ref = np.array(x_ref, dtype=precision)
                    x_ref_idx = np.int64(np.divide(x_ref, mesh.delX)) +\
                        np.ones(2, dtype=np.int64)
                else:
                    print("ERROR!")
                    print("Unsupported velocity initialization!", flush=True)
                    os._exit(1)
            if field == 'rho':
                initial = precision(region['fields']['rho'])
            if field == 'phi':
                initial = precision(region['fields']['phi'])
            fieldsInitial.append(initial)
    except KeyError as e:
        print('ERROR! Keyword ' + str(e) +
              ' missing in internalFields')
        return 0
    for i in range(mesh.Nx_global):
        for j in range(mesh.Ny_global):
            if (i >= boundingBox_idx[0, 0] and i <= boundingBox_idx[1, 0]
                    and j >= boundingBox_idx[0, 1] and
                    j <= boundingBox_idx[1, 1]):
                ind = int(i * mesh.Ny_global + j)
                setFields(fields, ind, fieldsInitial, mesh, velType,
                          x_ref=x_ref_idx)
    return 1
