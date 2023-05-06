---------
pylabolt
---------
|Documentation status|

pylabolt is a single phase, 2D, parallel lattice Boltzmann solver for fluid flow. It uses 
`Numba <https://numba.readthedocs.io/en/stable/>`_ accelerated `Python <https://www.python.org/>`_ code
to run lattice Boltzmann simulations on 2D lattices. Simulations can be run on CPU in parallel via 
Numba's own `OpenMP <https://www.openmp.org/>`_ parallelization and the `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ library.
For running on NVIDIA GPUs, pylabolt uses Numba's `CUDA <https://developer.nvidia.com/cuda-toolkit>`_ bindings.


.. |Documentation status| image:: https://readthedocs.org/projects/pylabolt/badge/?version=latest
    :target: https://pylabolt.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

=======================
Installation and Usage
=======================
pylabolt can be installed via ``pip`` via the command::

    $ pip install pylabolt

More details on the dependencies required to be installed and their configuration can be found
in the pylabolt documentation `here <https://pylabolt.readthedocs.io/en/latest/>`_.
Tutorial cases are provided in the ``tutorials`` folder. For example, consider the lid driven cavity
problem in ``tutorials/cavity/Re_100/``. The configuration file that defines the simulations is 
named as ``simulation.py``. After installation just run the following command from the ``tutorials/cavity/Re_100/``
folder to run the simulation::

    $ pylabolt --solver fluidLB

The output data is written in the ``output`` folder. By default the data is written into binary files with 
``.dat`` extension. To visualize the data in `Paraview <https://www.paraview.org/>`_ / 
`Mayavi <https://docs.enthought.com/mayavi/mayavi/>`_, the `VTK <https://vtk.org/>`_ library is used.
For example, to convert the last time-step data to a ``.vtk`` file, the following command should be run from
the working directory::

    $ pylabolt --toVTK last

The ``output_<time-step>.vtk`` files are stored in ``output/VTK`` directory which can be opened in
Paraview/Mayavi.

More details on setting up and running simulations can be found in the `documentation <https://pylabolt.readthedocs.io/en/latest/>`_.