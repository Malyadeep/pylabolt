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
    'default': {
        'u': 0,
        'v': 0,
        'rho': 1
    }
}

boundaryDict = {
    'outlet': {
        'type': 'zeroGradient',
        'entity': 'patch',
        'points_0': [[10, 0], [10, 1]]
    },
    'periodic': {
        'type': 'periodic',
        'entity': 'patch',
        'value': 0.,
        'points_0': [[0, 0], [10, 0]],
        'points_1': [[0, 1], [10, 1]]
    },
    'inlet': {
        'type': 'fixedU',
        'entity': 'patch',
        'value': [0.1, 0],
        'points_0': [[0, 0], [0, 1]]
    }
}

collisionDict = {
    'model': 'MRT',
    'nu': 0.01,
    'nu_B': 0.1,
    'S_q': 1.,
    'S_epsilon': 1.,
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
    'cylinder': {
        'type': 'circle',
        'center': [2, 0.5],
        'radius': 0.25,
        'static': True
    }
}

decomposeDict = {
    'nx': 5,
    'ny': 2
}
