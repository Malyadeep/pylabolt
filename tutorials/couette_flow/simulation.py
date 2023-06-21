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
    'walls': {
        'type': 'bounceBack',
        'entity': 'wall',
        'points_1': [[0, 0], [1, 0]]
    },
    'movingWall': {
        'type': 'fixedU',
        'entity': 'wall',
        'value': [0.1, 0],
        'points_0': [[0, 1], [1, 1]]
    },
    'Periodic': {
        'type': 'periodic',
        'entity': 'patch',
        'value': 0,
        'points_0': [[0, 0], [0, 1]],
        'points_1': [[1, 0], [1, 1]]
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
    'grid': [101, 101],
    'boundingBox': [[0, 0], [1, 1]]
}
