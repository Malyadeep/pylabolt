controlDict = {
    'startTime': 0,
    'endTime': 300000,
    'stdOutputInterval': 100,
    'saveInterval': 300000,
    'saveStateInterval': None,
    'relTolU': 1e-9,
    'relTolV': 1e-9,
    'relTolRho': 1e-7,
    'precision': 'double'
}

options = {
    'computeForces': True
}

internalFields = {
    'default': {
        'u': 0,
        'v': 0,
        'rho': 1
    }
}

boundaryDict = {
    'outlet': {
        'type': 'periodic',
        'entity': 'patch',
        'points_0': [[3, 0], [3, 1]]
    },
    'walls': {
        'type': 'bounceBack',
        'entity': 'wall',
        'points_0': [[0, 0], [3, 0]],
        'points_1': [[0, 1], [3, 1]]
    },
    'inlet': {
        'type': 'periodic',
        'entity': 'patch',
        'points_0': [[0, 0], [0, 1]]
    }
}

forcingDict = {
    'model': 'Guo',
    'value': [8e-6, 0]
}

collisionDict = {
    'model': 'BGK',
    'tau': 0.8,
    'equilibrium': 'secondOrder',
    'rho_ref': 1
}

latticeDict = {
    'latticeType': 'D2Q9'
}

meshDict = {
    'grid': [301, 101],
    'boundingBox': [[0, 0], [3, 1]]
}

decomposeDict = {
    'nx': 3,
    'ny': 2
}
