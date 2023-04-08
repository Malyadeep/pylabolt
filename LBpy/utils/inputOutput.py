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
                        str(round(fields.u[ind, 1], 10)).ljust(12) + '\n')
    writeFile.close()


def saveState(timeStep, simulation):
    if not os.path.isdir('states'):
        os.makedirs('states')
    fileName = 'states/' + str(timeStep) + '.pkl'
    stateFile = open(fileName, 'w')
    pickle.dump(simulation, stateFile, protocol=pickle.HIGHEST_PROTOCOL)


def loadState(timeStep):
    if not os.path.isdir('states'):
        print('ERROR! no previous states present!')
    else:
        try:
            fileName = 'states/' + str(timeStep) + '.pkl'
            simulation = pickle.load(fileName)
            return simulation
        except Exception:
            print('No saved states at time ' + str(timeStep) + ' present')
            print('creating new initial state')
            return None
