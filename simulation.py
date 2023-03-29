controlDict = {
    'startTime': 0,
    'endTime': 100,
    'stdOutputInterval': 5,
    'saveInterval': 10,
    'saveStateInterval': 100,
    'relTolU': 1e-6,
    'relTolV': 1e-6,
    'relTolRho': 1e-6,
}

internalFields = {
    'u': 0,
    'v': 0,
    'rho': 1
}

boundaryDict = {
    'top': {
        'type': 'fixedU',
        'value': [0.1, 0],
        'points_0': [[0, 0], [2, 0]]
    },

    'walls': {
        'type': 'bounceBack',
        'points_0': [[0, 0], [0, 1]],
        'points_1': [[0, 1], [2, 1]],
        'points_2': [[2, 0], [2, 1]],
    }
}

collisionDict = {
    'model': 'BGK',
    'tau': 1,
    'equilibrium': 'secondOrder'
}

latticeDict = {
    'latticeType': 'D2Q9'
}

meshDict = {
    'grid': [201, 101],
    'boundingBox': [[0, 0], [2, 1]]
}

obstacle = {
}
