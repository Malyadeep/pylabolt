controlDict = {
    'startTime': 0,
    'endTime': 100000,
    'stdOutputInterval': 100,
    'saveInterval': 100000,
    'saveStateInterval': None,
    'relTolU': 1e-7,
    'relTolV': 1e-7,
    'relTolRho': 1e-7,
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
    'all': {
        'type': 'periodic',
        'entity': 'patch',
        'points_0': [[0, 0], [1, 0]],
        'points_1': [[0, 1], [1, 1]],
        'points_2': [[1, 0], [1, 1]],
        'points_3': [[0, 0], [0, 1]]
    }
}

collisionDict = {
    'model': 'MRT',
    'nu': 0.1,
    'nu_B': 0.1,
    'S_q': 1.,
    'S_epsilon': 1.,
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

obstacle = {
    'annulus_inner': {
        'type': 'circle',
        'center': [0.5, 0.5],
        'radius': 0.2,
        'rho_s': 0.01,
        'static': False,
        'U_def': {
            'type': 'fixedRotational',
            'origin': [0.5, 0.5],
            'angularVelocity': 1e-5
        }
    },
    'annulus_outer': {
        'type': 'circularConfinement',
        'center': [0.5, 0.5],
        'radius': 0.4,
        'rho_s': 0.01,
        'static': False,
        'U_def': {
            'type': 'fixedRotational',
            'origin': [0.5, 0.5],
            'angularVelocity': 4e-5
        }
    }
}

decomposeDict = {
    'nx': 5,
    'ny': 2
}
