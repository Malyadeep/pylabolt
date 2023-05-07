==============
Installation
==============
PyLaBolt relies on various Python libraries to run the simulations. The various dependencies of PyLaBolt are as follows:

- `Numpy <https://numba.readthedocs.io/en/stable/>`_ >= 1.23.5
- `Numba <https://numba.readthedocs.io/en/stable/>`_ >= 0.56.4
- `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ >= 3.1.4

Optional dependencies required to convert data to VTK formats include 

- `vtk <https://pypi.org/project/vtk/>`_ >= 9.2.6

The dependencies can be installed via pip as follows::

    $ pip install numpy numba mpi4py vtk

To run simulations on parallel using MPI, check whether gcc/mpicc is properly cofigured or not.
To check the gcc and mpicc version run the following command in terminal/shell::

    $ gcc --version
    $ mpicc --version

If not properly configured visit `OpenMPI's homepage <https://www.open-mpi.org/>`_.
Also, for visualization of ``.vtk`` files, it is recommended that Paraview/Mayavi be used.
For installing Paraview visit `Paraview installation <https://www.paraview.org/Wiki/ParaView:Build_And_Install>`_.
For installing Mayavi visit `Mayavi installation <https://docs.enthought.com/mayavi/mayavi/installation.html>`_.

----------------------------------------
Additional configurations for GPU users
----------------------------------------
PyLaBolt uses Numba's CUDA bindings to run simulations on NVIDIA GPU. In order to run simulations on GPU, make 
sure you have a compatible GPU. Currently Numba supports GPUs with compute capability >= 3.5. For more information
visit `Numba for CUDA users <https://numba.readthedocs.io/en/stable/cuda/overview.html>`_

To install CUDA toolkit and drivers, visit `NVIDIA developer's <https://developer.nvidia.com/cuda-toolkit>`_ page. Once CUDA
toolkit and Numba is installed, fire up a python shell and run the following commands to check the compute capability 
of the GPU::

    >> from numba import cuda
    >> cuda.detect()

.. warning::
    Though PyLaBolt supports execution on GPUs, it is fairly unoptimized compared to the CPU counterparts
    as of version 0.1.1.

    This issue is being catered to and hopefully will be fixed in a future release.

--------------------
Installing PyLaBolt
--------------------
Once the dependencies are installed, PyLaBolt can be installed via ``pip`` as::

    $ pip install pylabolt

To install an older version, one can use ``==`` and specify the version name explicitly. For example::

    $ pip install pylabolt==0.1.1

For the release history and version numbers, visit the official PyPi repository of `PyLaBolt <https://pypi.org/project/pylabolt/>`_.

To verify whether the installation was successfull, run the following command from the terminal::

    $ pylabolt -h 

For successfull installation, the output should look like::

    $ pylabolt -h
    usage: pylabolt [-h] [-s {fluidLB}] [-p] [-c] [-nt N_THREADS] [--reconstruct {last,all,time,None}] [-t TIME] [--toVTK {last,all,time,None}]

    A Lattice Boltzmann Python solver

    optional arguments:
    -h, --help            show this help message and exit
    -s {fluidLB}, --solver {fluidLB}
                            choice of solver to run
    -p, --parallel        set to run simulation in parallel using OpenMP
    -c, --cuda            set to run simulation in parallel using CUDA
    -nt N_THREADS, --n_threads N_THREADS
                            Number of threads for OpenMP/CUDA
    --reconstruct {last,all,time,None}
                            Domain reconstruction
    -t TIME, --time TIME  Specify time which is to be reconstructed
    --toVTK {last,all,time,None}
                            Convert output data to VTK format

Voila! PyLaBolt is successfully installed. Now you can run simulations.


