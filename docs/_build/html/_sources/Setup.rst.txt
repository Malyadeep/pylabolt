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
``controlDict`` has the following entries:

- ``startTime`` - denotes the starting timestep of the simulation. Usually set to zero.
  If set to any time other than zero, the solver searches for saved states at the specified
  time step. If a saved state is found, the solver resumes the simulation from the saved state.
- ``endTime`` - denotes the end timestep. 

.. note::
    The timesteps doesn't necessarily mean physical time in seconds. All the fields in lattice
    Boltzmann simulations are scaled appropriately to lattice units. For more information on
    scaling refer to `The Lattice Boltzmann Method - Timm Krüger, Halim Kusumaatmaja, Alexandr Kuzmin, Orest Shardt, Goncalo Silva, Erlend Magnus Viggen
    <https://doi.org/10.1007/978-3-319-44649-3>`_

- ``stdOutputInterval`` - denotes the time interval after which output is displayed to terminal/shell
- ``saveInterval`` - denotes the time interval after which the state of the simulation must be saved. Useful
  for large simulations which can be resumed from a saved state in between if something goes wrong. Default value 
  is ``None``. (Currently not implemented for usage with MPI. Will be added in future release)

All timestep and time interval values must be integers.

- ``relTolU`` - denotes the desired relative tolerance for x-component of velocity. 
- ``relTolV`` - denotes the desired relative tolerance for y-component of velocity.
- ``relTolRho`` - denotes the desired relative tolerance for density.
- ``precision`` - denotes the floating point precision to be used for simulation. For single precision floating point
  use the string entry ``single``. Similarly for double precision use ``double``. 

.. note::
    For GPU users, choose the floating point precision carefully because double precision floating point
    operations are considerably slower on a GPU compared to single precision. Consider the ``FP32/FP64`` 
    performance ratio and the accuracy needed in the simulation before setting the precision.



+++++++++++++++++++
``internalFields``
+++++++++++++++++++
This dictionary defines the initial values of the velocity and density fields.

- ``u`` - ``float`` entry that denotes initial value of x-component of velocity
- ``v`` - ``float`` entry that denotes initial value of y-component of velocity
- ``rho`` - ``float`` entry that denotes initial value of density

All the fields are initialized to the specified uniformly.

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






