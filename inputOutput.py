import os
import pickle


def writeFields(timeStep, elements):
    if not os.path.isdir('output'):
        os.makedirs('output')
    if not os.path.isdir('output/' + str(timeStep)):
        os.makedirs('output/' + str(timeStep))
    writeFile = open('output/' + str(timeStep) + '/fields.dat', 'w')
    for element in elements:
        writeFile.write(str(round(element.id, 10)).ljust(12) + '\t' +
                        str(round(element.x, 10)).ljust(12) + '\t' +
                        str(round(element.y, 10)).ljust(12) + '\t' +
                        str(round(element.rho, 10)).ljust(12) + '\t' +
                        str(round(element.u[0], 10)).ljust(12) + '\t' +
                        str(round(element.u[1], 10)).ljust(12) + '\n')
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
