====================
Boundary Conditions
====================

This section lists all the boundary conditions available in PyLaBolt, and how to 
set them in simulations.

++++++++++++++++++++++++++
Fluid boundary conditions
++++++++++++++++++++++++++
This section covers the boundary conditions that can applied to fluid flow simulations.

- ``bounceBack`` - Implements no-slip boundary conditions at walls via the `halfway bounce-back 
  <https://doi.org/10.1017/S0022112094001771>`_ method. ``entity`` type should be set to ``wall`` in order 
  to compute forces and torque acting on the boundary.
- ``fixedU`` - Provides a Dirichlet boundary condition for velocity via the `halfway bounce-back 
  <https://doi.org/10.1017/S0022112094001771>`_ method. When specified, a fixed velocity is
  assigned at the concerned boundary. ``entity`` type ``wall`` allows computation of forces and torque on the 
  boundary, whereas, for inlet or outlet type of boundaries, where force and torque computation isn't required, 
  the ``entity`` type can be set to ``patch``.
- ``fixedPressure`` - Provides a Dirichlet boundary condition for pressure via the `halfway bounce-back 
  <https://doi.org/10.1017/S0022112094001771>`_ method. For inlet or outlet, set ``entity`` type to ``patch``. 
  The ``value`` keyword takes a ``float`` value for pressure. For single phase fluid flow, the pressure can be 
  found from density as :math:`p = \rho c_s^2`, where :math:`c_s` is the speed of sound in lattice units. For more 
  information refer `here <https://doi.org/10.1007/978-3-319-44649-3>`_.
- Zero gradient boundary condition
- Periodic boundary condition - `The Lattice Boltzmann Method <https://doi.org/10.1007/978-3-319-44649-3>`_