controlDict = {
    'startTime': 0,
    'endTime': 100,
    'saveInterval': 10
}

boundaryDict = {
    'top': {
        'type': 'fixedScalar',
        'value': 0.0,
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
    'latticeType': 'D2Q9',
    'deltaT': 1,
    'deltaX': 1
}

meshDict = {
    'grid': [201, 101],
    'boundingBox': [[0, 0], [2, 1]]
}

obstacle = {
}
