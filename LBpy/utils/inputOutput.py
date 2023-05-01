import os
import pickle


def writeFields(timeStep, fields):
    if not os.path.isdir('output'):
        os.makedirs('output')
    if not os.path.isdir('output/' + str(timeStep)):
        os.makedirs('output/' + str(timeStep))
    writeFile = open('output/' + str(timeStep) + '/fields.dat', 'w')
    for ind in range(fields.u.shape[0]):
        writeFile.write(str(round(ind, 10)).ljust(12) + '\t' +
                        str(round(fields.x[ind], 10)).ljust(12) + '\t' +
                        str(round(fields.y[ind], 10)).ljust(12) + '\t' +
                        str(round(fields.rho[ind], 10)).ljust(12) + '\t' +
                        str(round(fields.u[ind, 0], 10)).ljust(12) + '\t' +
                        str(round(fields.u[ind, 1], 10)).ljust(12) + '\t' +
                        str(round(fields.solid[ind], 10)).ljust(12) + '\n')
    writeFile.close()


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
        print('ERROR! no previous states present!')
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
