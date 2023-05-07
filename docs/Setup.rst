=======================
Setting up simulations
=======================
This section walks you through the setup of the simulations, to run using PyLaBolt.
For this purpose we shall look into the `lid driven cavity tutorial <https://github.com/Malyadeep/pylabolt/tree/main/tutorials/cavity/Re_100>`_,
provided in the `PyLaBolt Github repository <https://github.com/Malyadeep/pylabolt/tree/main>`_.

-----------------------
``simulation.py`` file
-----------------------
All the input data from the user is provided in the ``simulation.py`` file which
must be present in the working directory. A sample ``simulation.py`` file is looks
like::

    controlDict = {
        'startTime': 0,
        'endTime': 50000,
        'stdOutputInterval': 100,
        'saveInterval': 10000,
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
        'walls': {
            'type': 'bounceBack',
            'points_2': [[1, 0], [1, 1]],
            'points_0': [[0, 0], [1, 0]],
            'points_1': [[0, 0], [0, 1]]
        },
        'lid': {
            'type': 'fixedU',
            'value': [0.1, 0],
            'points_1': [[0, 1], [1, 1]]
        }
    }

    collisionDict = {
        'model': 'BGK',
        'tau': 0.8,
        'equilibrium': 'secondOrder'
    }

    latticeDict = {
        'latticeType': 'D2Q9'
    }

    meshDict = {
        'grid': [101, 101],
        'boundingBox': [[0, 0], [1, 1]]
    }

We shall use this file to run the simulations for lid driven cavity. The following
subsections expand on the various dictionaries present in ``simulation.py`` file.

++++++++++++++++
``controlDict``
++++++++++++++++


+++++++++++++++++++
``internalFields``
+++++++++++++++++++


++++++++++++++++
``boundaryDict``
++++++++++++++++


+++++++++++++++++
``collisionDict``
+++++++++++++++++


++++++++++++++++
``latticeDict``
++++++++++++++++


++++++++++++++++
``meshDict``
++++++++++++++++






