controlDict = {
    'startTime': 0,
    'endTime': 100000,
    'stdOutputInterval': 100,
    'saveInterval': 100,
    'saveStateInterval': None,
    'relTolU': 1e-10,
    'relTolV': 1e-10,
    'relTolRho': 1e-10,
    'precision': 'double'
}

options = {
    'computeForces': True,
    'computeTorque': True,
    'x_ref': [0.5, 0.5]
}


internalFields = {
    'default': {
        'u': 0,
        'v': 0,
        'rho': 1
    }
}

boundaryDict = {
    'inletOutlet': {
        'type': 'periodic',
        'entity': 'patch',
        'points_0': [[0, 0], [0, 1]],
        'points_1': [[1, 0], [1, 1]]
    },
    'topPlate': {
        'type': 'fixedU',
        'entity': 'wall',
        'value': [0.0025, 0],
        'points_0': [[0, 1], [1, 1]]
    },
    'bottomPlate': {
        'type': 'bounceBack',
        'entity': 'wall',
        'points_0': [[0, 0], [1, 0]]
    }
}

collisionDict = {
    'model': 'MRT',
    'nu': 1.0,
    'nu_B': 1.0,
    'S_epsilon': 1.0,
    'S_q': 1.0,
    'equilibrium': 'secondOrder',
    'rho_ref': 1
}

latticeDict = {
    'latticeType': 'D2Q9'
}

meshDict = {
    'grid': [161, 161],
    'boundingBox': [[0, 0], [1, 1]]
}

obstacle = {
    'cylinder': {
        'type': 'circle',
        'center': [0.5, 0.5],
        'radius': 0.125,
        'static': False,
        'rho_s': 100,
        'U_def': {
            'type': 'calculatedRotational',
            'origin': [0.5, 0.5],
            'angularVelocity': 0,
            'write': True,
            'writeInterval': 10
        }
    }
}

decomposeDict = {
    'nx': 3,
    'ny': 3
}
