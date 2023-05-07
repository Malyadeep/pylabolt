controlDict = {
    'startTime': 0,
    'endTime': 50000,
    'stdOutputInterval': 100,
    'saveInterval': 100,
    'saveStateInterval': None,
    'relTolU': 1e-9,
    'relTolV': 1e-9,
    'relTolRho': 1e-7,
    'precision': 'double'
}

internalFields = {
    'u': 0,
    'v': 0,
    'rho': 1
}

boundaryDict = {
    'outlet': {
        'type': 'zeroGradient',
        'points_2': [[10, 0], [10, 1]]
    },
    'periodic': {
        'type': 'periodic',
        'value': 0.,
        'points_0': [[0, 0], [10, 0]],
        'points_1': [[0, 1], [10, 1]]
    },
    'inlet': {
        'type': 'fixedU',
        'value': [0.1, 0],
        'points_1': [[0, 0], [0, 1]]
    }
}

collisionDict = {
    'model': 'BGK',
    'tau': 0.65,
    'equilibrium': 'secondOrder'
}

latticeDict = {
    'latticeType': 'D2Q9'
}

meshDict = {
    'grid': [1001, 101],
    'boundingBox': [[0, 0], [10, 1]]
}

obstacle = {
    'type': 'circle',
    'center': [2, 0.5],
    'radius': 0.25
}

decomposeDict = {
    'nx': 4,
    'ny': 2
}
