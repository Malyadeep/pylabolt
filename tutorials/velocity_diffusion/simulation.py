controlDict = {
    'startTime': 0,
    'endTime': 500,
    'stdOutputInterval': 50,
    'saveInterval': 50,
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
    },
    'region_0': {
        'type': 'line',
        'point_0': [0.5, 0],
        'point_1': [0.5, 1],
        'fields': {
            'u': 0,
            'v': 0.01,
            'rho': 1
        }
    }
}

boundaryDict = {
    'outlet': {
        'type': 'periodic',
        'points_0': [[1, 0], [1, 1]]
    },
    'walls': {
        'type': 'periodic',
        'points_0': [[0, 0], [1, 0]],
        'points_1': [[0, 1], [1, 1]]
    },
    'inlet': {
        'type': 'periodic',
        'points_0': [[0, 0], [0, 1]]
    }
}

collisionDict = {
    'model': 'BGK',
    'tau': 0.8,
    'equilibrium': 'stokesLinear',
    'rho_ref': 1
}

latticeDict = {
    'latticeType': 'D2Q9'
}

meshDict = {
    'grid': [101, 101],
    'boundingBox': [[0, 0], [1, 1]]
}

obstacle = {
    # 'type': 'circle',
    # 'center': [2, 0.5],
    # 'radius': 0.25
}

decomposeDict = {
    'nx': 3,
    'ny': 2
}
