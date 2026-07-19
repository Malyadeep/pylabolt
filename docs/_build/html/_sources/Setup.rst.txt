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
        'default': {
            'u': 0,
            'v': 0,
            'rho': 1
        }
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
This dictionary defines the initial values of the velocity and density fields. The mandatory keyword is ``default`` under 
which the fields are defined. Custom initialization of a certain region in the domain is possible using keys other than ``default``. 
Currently only ``line`` is supported for custom initialization. Other types of regions will be incorporated in future releases.
For more information on defining regions look at the ``velocity_diffusion`` tutorial.

- ``u`` - ``float`` entry that denotes initial value of x-component of velocity
- ``v`` - ``float`` entry that denotes initial value of y-component of velocity
- ``rho`` - ``float`` entry that denotes initial value of density

All the fields are initialized to the specified values uniformly.

++++++++++++++++
``boundaryDict``
++++++++++++++++
This dictionary defines the boundaries of the domain and the boundary conditions on them. 
You can define multiple dictionaries inside this dictionary which represents a particular 
boundary region. A sample entry defining a wall boundary is as follows::
  
  'walls': {
            'type': 'bounceBack',
            'points_0': [[1, 0], [1, 1]]
  }

Here, the key ``walls`` is a user-specified name. Inside the ``walls`` dictionary, there is 
a mandatory entry ``type``, which denotes the boundary condition to be applied. For example, 
here since the boundary is a wall, we apply ``bounceBack`` boundary condition. The next entries 
following the ``type`` keyword are the points that define the region of the boundary under 
consideration. Points must be a 2-Dimensional list entry that defines the starting and ending 
coordinates of the region on the boundary. 

.. note::
  * Note that the coordinates must be positive for now. This issue will be resolved in later releases.
  * Make sure the coordinates in the first row of the point lists are less than or equal to 
    the ones on the next row.


+++++++++++++++++
``collisionDict``
+++++++++++++++++
This dictionary defines the collision and equilibrium scheme to be used

- ``model`` - Keyword entry that defines the collision model to be used. Currently, 
  supports only Bhatnagar-Gross-Krook model (``BGK``). In future releases the MRT and
  TRT model will be added.
 
- ``tau`` - ``float`` entry denoting the relaxation time for the BGK model.
- ``equilibrium`` - Keyword entry denoting the type of equilbrium distribution functions
  to be used. Currently, provides two operations
  
  * ``stokesLinear`` - equilibrium distribution function which is first order in velocity.
    Useful in modelling Stokes' flow. For this type of equilibrium another keyword is required 
    called ``rho_ref`` which denotes the reference density for incompressible flows.
  * ``secondOrder`` - equilibrium distribution function which is second order in velocity.
    Used in most fluid flow simulations.
  * ``incompressible`` - equilibrium distribution function which is second order in velocity for 
    incompressible flow. For this model, ``rho_ref`` keyword is required. 

++++++++++++++++
``latticeDict``
++++++++++++++++
This dictionary lets you choose the type of lattices to use for simulation
- ``latticeType`` - Keyword entry which tells the lattice to use for simulation. Currently,
supports the following lattices
  
  * ``D1Q3`` - 1D lattice with 3 velcoity directions. Ideal for 1D simulations.
  * ``D2Q9`` - 2D lattice with 9 velcoity directions. Ideal for 2D fluid flow and
    heat transfer simulations.
  
In future release, ``D2Q5`` lattices will be added for 2D heat conduction simulations. A 3D extension
is also planned which shall see the inclusion of 3D lattices as well.

++++++++++++++++
``meshDict``
++++++++++++++++
This dictionary defines the computational domain. The entire domain is supposed to take a rectangular shape. 
Following keywords are required to define the dictionary

  * ``boundingBox`` - A 2-Dimensional list entry that defines two ends of the diagonal of the rectangular domain. 
    Note that the coordinates must be positive for now. This issue will be resolved in later releases. 
    Also, the coordinates must be defined in a way that the slope of the diagonal is positive.
  * ``grid`` - A list entry that defines the resolution of the computational domain, or in other words the no. of 
    lattice points. The lattices must be square; accordingly the no. of grid points must be set.








