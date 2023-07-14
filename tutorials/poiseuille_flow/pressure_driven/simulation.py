controlDict = {
    'startTime': 0,
    'endTime': 100000,
    'stdOutputInterval': 100,
    'saveInterval': 10000,
    'saveStateInterval': None,
    'relTolU': 1e-9,
    'relTolV': 1e-9,
    'relTolRho': 1e-7,
    'precision': 'double'
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
        'type': 'fixedPressure',
        'entity': 'patch',
        'value': 0.3333333,
        'points_0': [[3, 0], [3, 1]]
    },
    'walls': {
        'type': 'bounceBack',
        'entity': 'wall',
        'points_0': [[0, 0], [3, 0]],
        'points_1': [[0, 1], [3, 1]]
    },
    'inlet': {
        'type': 'fixedPressure',
        'entity': 'patch',
        'value': 0.3383333,
        'points_0': [[0, 0], [0, 1]]
    }
}

collisionDict = {
    'model': 'BGK',
    'nu': 0.1,
    'equilibrium': 'secondOrder',
    'rho_ref': 1
}

latticeDict = {
    'latticeType': 'D2Q9'
}

meshDict = {
    'grid': [121, 41],
    'boundingBox': [[0, 0], [3, 1]]
}

decomposeDict = {
    'nx': 3,
    'ny': 1
}
