---------
PyLaBolt
---------
|Documentation status|

PyLaBolt is a single phase, 2D, parallel lattice Boltzmann solver for fluid flow. It uses 
`Numba <https://numba.readthedocs.io/en/stable/>`_ accelerated `Python <https://www.python.org/>`_ code
to run lattice Boltzmann simulations on 2D lattices. Simulations can be run on CPU in parallel via 
Numba's own `OpenMP <https://www.openmp.org/>`_ parallelization and the `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ library.
For running on NVIDIA GPUs, PyLaBolt uses Numba's `CUDA <https://developer.nvidia.com/cuda-toolkit>`_ bindings.


.. |Documentation status| image:: https://readthedocs.org/projects/pylabolt/badge/?version=latest
    :target: https://PyLaBolt.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

=======================
Installation and Usage
=======================
PyLaBolt can be installed via ``pip`` via the command::

    $ pip install pylabolt

More details on the dependencies and their configuration can be found
in the PyLaBolt documentation `here <https://PyLaBolt.readthedocs.io/en/latest/>`_.

Tutorial cases are provided in the ``tutorials`` folder. For example, consider the lid driven cavity
problem in ``tutorials/cavity/Re_100/``. The configuration file that defines the simulation is 
called ``simulation.py``. After installation just run the following command from the ``tutorials/cavity/Re_100/``
folder::

    $ pylabolt --solver fluidLB

The output data is written in the ``output`` folder. By default the data is written into binary files with 
``.dat`` extension. To visualize the data in `Paraview <https://www.paraview.org/>`_ / 
`Mayavi <https://docs.enthought.com/mayavi/mayavi/>`_, the `VTK <https://vtk.org/>`_ library is used.
For example, to convert the last time-step data to a ``.vtk`` file, run the following command from
the working directory::

    $ pylabolt --toVTK last

The ``output_<time-step>.vtk`` files are stored in ``output/VTK`` directory which can be opened in
Paraview/Mayavi. Sample results for lid driven cavity, plane Poiseuille flow, and flow past a cylinder are shown below.

.. figure:: https://github.com/Malyadeep/pylabolt/blob/main/tutorials/cavity/Re_100/cavity_Vcontour.png
   :width: 550px
   :alt: lid driven cavity (Re = 100)

.. figure:: https://github.com/Malyadeep/pylabolt/blob/main/tutorials/poiseuille_flow/V_contour.png
   :width: 550px
   :alt: Plane Poiseuille flow (Re = 100)

.. figure:: https://github.com/Malyadeep/pylabolt/blob/main/tutorials/flow_past_cylinder/flowpastcylinder_Vcontour.png
   :width: 550px
   :alt: Flow past a cylinder (Re = 100)

More details on setting up and running simulations can be found in the `documentation <https://PyLaBolt.readthedocs.io/en/latest/>`_.


=======================
Features
=======================
PyLaBolt currently supports the following collision schemes

- Bhatnagar-Gross-Krook (BGK) scheme - `Physical Review, vol. 94, Issue 3, pp. 511-525 
  <https://ui.adsabs.harvard.edu/link_gateway/1954PhRv...94..511B/doi:10.1103/PhysRev.94.511>`_

The `MRT <https://doi.org/10.1098/rsta.2001.0955>`_ and `TRT <https://global-sci.org/intro/article_detail/cicp/7862.html>`_ 
collision schemes will be added in future releases. 

The boundary conditions available are

- No slip boundary via the halfway bounce back method - `Journal of Fluid Mechanics , Volume 271 , 25 July 1994 
  , pp. 285 - 309  <https://doi.org/10.1017/S0022112094001771>`_
- Moving wall boundary condition via the halfway bounce back method - `Journal of Fluid Mechanics , Volume 271 , 25 July 1994 
  , pp. 285 - 309  <https://doi.org/10.1017/S0022112094001771>`_ , 
  `Journal of Statistical Physics volume 104, pages 1191–1251 (2001)
  <https://doi.org/10.1023/A:1010414013942>`_ 
- The fixed pressure boundary condition via the anti-bounce back method - `Commun. Comput. Phys. 3, 427 (2008) 
  <https://www.researchgate.net/publication/281975403_Study_of_Simple_Hydrodynamic_Solutions_with_the_Two-Relaxation-Times_Lattice_Boltzmann_Scheme>`_
- Zero gradient boundary condition
- Periodic boundary condition - `The Lattice Boltzmann Method <https://doi.org/10.1007/978-3-319-44649-3>`_

For more information on the schemes and boundary conditions, we urge the reader the go through the following books

- `The Lattice Boltzmann Method - Timm Krüger, Halim Kusumaatmaja, Alexandr Kuzmin, Orest Shardt, Goncalo Silva, Erlend Magnus Viggen
  <https://doi.org/10.1007/978-3-319-44649-3>`_
- `The Lattice Boltzmann Equation: For Complex States of Flowing Matter - Sauro Succi 
  <https://global.oup.com/academic/product/the-lattice-boltzmann-equation-9780199592357?cc=us&lang=en&>`_

PyLaBolt leverages the performance advantages on multi-core CPUs, High Perfomance computing clusters, and
GPUs to run large simulations. Currently the parallel computing features supported by PyLaBolt are:

- `Numba <https://numba.readthedocs.io/en/stable/>`_ accelerated `Python <https://www.python.org/>`_ code can
  run in parallel on multi-core CPUs through `OpenMP <https://www.openmp.org/>`_.
- To run on multiple machines/clusters, PyLaBolt uses `OpenMPI <https://www.open-mpi.org/>`_ via the `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_
  library.
- PyLaBolt can also run simulations on NVIDIA GPUs through Numba's `CUDA <https://developer.nvidia.com/cuda-toolkit>`_ bindings.
- PyLaBolt provides support to convert output to `VTK <https://vtk.org/>`_ format, which can post-processed in Paraview/Mayavi.

For more details, refer to the `documentation <https://PyLaBolt.readthedocs.io/en/latest/>`_.

=======================
Acknowledgements
=======================
PyLaBolt grew from the course project of AE6102 course, of the `Department of Aerospace Engineering <https://www.aero.iitb.ac.in/home/>`_ 
at Indian Institute of Technology, Bombay. We are grateful to `Prof. Prabhu Ramachandran <https://www.aero.iitb.ac.in/~prabhu/>`_ 
of Department of Aerospace Engineering at Indian Institute of Technology, Bombay for exposing us to the necessary skills to write
high performance scientific codes.

We are also grateful to `Dr. Amol Subhedar <https://www.che.iitb.ac.in/faculty/amol-subhedar>`_ at
`Department of Chemical Engineering <https://www.che.iitb.ac.in/>`_, Indian Institute of Technology, Bombay for providing 
us critical suggestions on the theoretical foundations of the algorithms.

For contributors see the `Github contributors page <https://github.com/Malyadeep/pylabolt/graphs/contributors>`_.