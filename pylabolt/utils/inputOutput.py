import os
import pickle
import numpy as np


def writeFields(timeStep, fields, lattice, mesh):
    u = False
    rho = False
    phi = False
    T = False
    p = False
    if not os.path.isdir('output'):
        os.makedirs('output')
    if not os.path.isdir('output/' + str(timeStep)):
        os.makedirs('output/' + str(timeStep))
    pointsFile = open('output/' + str(timeStep) + '/points.dat', 'w')
    solidFile = open('output/' + str(timeStep) + '/solid.dat', 'w')
    for field in fields.fieldList:
        if field == 'u':
            u = True
            uFile = open('output/' + str(timeStep) + '/u.dat', 'w')
        if field == 'rho':
            rho = True
            rhoFile = open('output/' + str(timeStep) + '/rho.dat', 'w')
        if field == 'p':
            p = True
            pFile = open('output/' + str(timeStep) + '/p.dat', 'w')
        if field == 'phi':
            phi = True
            phiFile = open('output/' + str(timeStep) + '/phi.dat', 'w')
        if field == 'T':
            T = True
            TFile = open('output/' + str(timeStep) + '/T.dat', 'w')
    for ind in range(mesh.Nx * mesh.Ny):
        if fields.boundaryNode[ind] != 1:
            pointsFile.write(str(np.round(ind, 10)).ljust(12) + '\t' +
                             str(np.round(mesh.x[ind], 10)).ljust(12) + '\t' +
                             str(np.round(mesh.y[ind], 10)).ljust(12) + '\n')
            if u is True:
                uFile.\
                    write(str(np.round(fields.u[ind, 0], 10)).ljust(12) + '\t'
                          + str(np.round(fields.u[ind, 1], 10)).ljust(12) +
                          '\n')
            if rho is True:
                rhoFile.write(str(np.round(fields.rho[ind], 10)).ljust(12) +
                              '\n')
            if p is True:
                pFile.write(str(np.round(fields.p[ind] * fields.rho[ind]
                            * lattice.cs * lattice.cs, 10)).ljust(12) +
                            '\n')
            if phi is True:
                phiFile.write(str(np.round(fields.phi[ind], 10)).ljust(12) +
                              '\n')
            if T is True:
                TFile.write(str(np.round(fields.T[ind], 10)).ljust(12) +
                            '\n')
            solidFile.write(str(np.round(fields.solid[ind, 0], 10)).ljust(12)
                            + '\n')
    pointsFile.close()
    solidFile.close()
    if u is True:
        uFile.close()
    if rho is True:
        rhoFile.close()
    if p is True:
        pFile.close()
    if phi is True:
        phiFile.close()
    if T is True:
        TFile.close()


def saveState(timeStep, simulation):
    if not os.path.isdir('states'):
        os.makedirs('states')
    if not os.path.isdir('states/' + str(timeStep)):
        os.makedirs('states/' + str(timeStep))
    fieldsFile = open('states/' + str(timeStep) + '/fields.pkl', 'wb')
    pickle.dump(simulation.fields, fieldsFile,
                protocol=pickle.HIGHEST_PROTOCOL)


def loadState(timeStep):
    if not os.path.isdir('states'):
        print('ERROMPIR! no previous states present!')
    else:
        try:
            fileName = 'states/' + str(timeStep) + '/fields.pkl'
            fields = pickle.load(fileName)
            return fields
        except Exception:
            print('No saved states at time ' + str(timeStep) + ' present')
            print('creating new initial state')
            return None


def copyFields_cuda(device, fields, flag):
    if flag == 'standard':
        device.u.copy_to_host(fields.u)
        device.rho.copy_to_host(fields.rho)
    elif flag == 'all':
        device.f.copy_to_host(fields.f)
        device.f_new.copy_to_host(fields.f_new)
        device.f_eq.copy_to_host(fields.f_eq)
    pass


def writeFields_mpi(timeStep, fields, lattice, mesh, rank, comm):
    u = False
    rho = False
    phi = False
    T = False
    p = False
    if not os.path.isdir('procs'):
        os.makedirs('procs')
    if not os.path.isdir('procs/proc_' + str(rank)):
        os.makedirs('procs/proc_' + str(rank))
    if not os.path.isdir('procs/proc_' + str(rank) + '/' + str(timeStep)):
        os.makedirs('procs/proc_' + str(rank) + '/' + str(timeStep))
    pointsFile = open('procs/proc_' + str(rank) + '/' +
                      str(timeStep) + '/points.dat', 'w')
    solidFile = open('procs/proc_' + str(rank) + '/' +
                     str(timeStep) + '/solid.dat', 'w')
    boundaryNodeFile = open('procs/proc_' + str(rank) + '/' +
                            str(timeStep) + '/boundaryNode.dat', 'w')
    for field in fields.fieldList:
        if field == 'u':
            u = True
            uFile = open('procs/proc_' + str(rank) + '/' +
                         str(timeStep) + '/u.dat', 'w')
        if field == 'rho':
            rho = True
            rhoFile = open('procs/proc_' + str(rank) + '/' +
                           str(timeStep) + '/rho.dat', 'w')
        if field == 'p':
            p = True
            pFile = open('procs/proc_' + str(rank) + '/' +
                         str(timeStep) + '/p.dat', 'w')
        if field == 'phi':
            phi = True
            phiFile = open('procs/proc_' + str(rank) + '/' +
                           str(timeStep) + '/phi.dat', 'w')
        if field == 'T':
            T = True
            TFile = open('procs/proc_' + str(rank) + '/' +
                         str(timeStep) + '/T.dat', 'w')
    for ind in range(mesh.Nx * mesh.Ny):
        pointsFile.write(str(np.round(ind, 10)).ljust(12) + '\n')
        if u is True:
            uFile.write(str(np.round(fields.u[ind, 0], 10)).ljust(12) + '\t' +
                        str(np.round(fields.u[ind, 1], 10)).ljust(12) + '\n')
        if rho is True:
            rhoFile.write(str(np.round(fields.rho[ind], 10)).ljust(12) + '\n')
        if p is True:
            pFile.write(str(np.round(fields.p[ind] * fields.rho[ind]
                        * lattice.cs * lattice.cs, 10)).ljust(12) + '\n')
        if phi is True:
            phiFile.write(str(np.round(fields.phi[ind], 10)).ljust(12) + '\n')
        if T is True:
            TFile.write(str(np.round(fields.T[ind], 10)).ljust(12) + '\n')
        solidFile.write(str(np.round(fields.solid[ind, 0], 10)).ljust(12)
                        + '\n')
        boundaryNodeFile.write(str(np.round(fields.boundaryNode[ind], 10)).
                               ljust(12)
                               + '\n')
    pointsFile.close()
    solidFile.close()
    boundaryNodeFile.close()
    if u is True:
        uFile.close()
    if rho is True:
        rhoFile.close()
    if p is True:
        pFile.close()
    if phi is True:
        phiFile.close()
    if T is True:
        TFile.close()
