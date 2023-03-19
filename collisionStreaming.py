import os
import collisionModels
import streaming


class collisionStreaming:
    def __init__(self, latticeDict, c, collisionDict):
        try:
            self.deltaX = latticeDict['deltaX']
            self.deltaT = latticeDict['deltaT']
            if collisionDict['model'] == 'BGK':
                self.collisionFunc = collisionModels.BGK
            else:
                print("ERROR! Unsupported collision model : " +
                      collisionDict['model'])
        except KeyError as e:
            print("ERROR! Keyword: " + str(e) + " missing in 'latticeDict'")
            os._exit()
        try:
            self.tau = collisionDict['tau']
        except KeyError as e:
            print("ERROR! Keyword: " + str(e) + " missing in 'collisionDict'")
            os._exit()
        self.dtdx = self.deltaT/self.deltaX
        self.c = c
        self.preFactor = self.deltaT/self.tau

    def collide(self, f, f_new, f_eq, nodeType):
        self.collisionFunc(f, f_new, f_eq, nodeType, self.preFactor)

    def propagate(self, f, f_new, nodeType):
        streaming.stream(f, f_new, nodeType, self.dtdx, self.c)


if __name__ == '__main__':
    print('module implementing various collision and streaming steps')
