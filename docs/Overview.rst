==========
Overview
==========

pylabolt is a single phase, 2D, parallel lattice Boltzmann solver for fluid flow. It uses 
`Numba <https://numba.readthedocs.io/en/stable/>`_ accelerated `Python <https://www.python.org/>`_ code
to run lattice Boltzmann simulations on 2D lattices. Simulations can be run on CPU in parallel via 
Numba's own `OpenMP <https://www.openmp.org/>`_ parallelization and the `mpi4py <https://mpi4py.readthedocs.io/en/stable/>` library.
For running on NVIDIA GPUs, pylabolt uses Numba's `CUDA <https://developer.nvidia.com/cuda-toolkit>`_ bindings.
