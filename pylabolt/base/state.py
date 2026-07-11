import os
import numpy as np

import pylabolt.base.control as control
import pylabolt.base.mesh as mesh
import pylabolt.base.lattice as lattice
import pylabolt.base.transport as transport
import pylabolt.base.fields as fields
import pylabolt.base.boundary as boundary
import pylabolt.base.obstacle as obstacle
import pylabolt.parallel.domain as domain

from pylabolt.base.init_fields import init_fields
from pylabolt.utils.helpers import print_log


class State:
    def __init__(
        self,
        simulation,
        comm,
        mpi_rank,
        fluid=False,
        phase=False,
        scalar=False
    ):
        """
        Container for constants and evolving fields
        Attributes:

        """
        self.fluid = fluid
        self.phase = phase
        self.scalar = scalar
        try:
            self.control = control.Control(
                simulation,
                mpi_rank,
                verbose=True
            )

            self.mesh = mesh.Mesh(
                simulation,
                mpi_rank,
                verbose=True
            )

            self.lattice = lattice.Lattice(
                simulation,
                self.control,
                self.mesh,
                mpi_rank,
                verbose=True
            )

            self.domain = domain.Domain(
                simulation,
                self.mesh,
                comm,
                verbose=True
            )

            self.transport = transport.Transport(
                simulation,
                self.control,
                self.domain,
                fluid=self.fluid,
                phase=self.phase,
                scalar=self.scalar,
                verbose=True
            )

            self.fields = fields.Fields(
                self.control,
                self.lattice,
                self.domain,
                fluid=self.fluid,
                phase=self.phase,
                scalar=self.scalar
            )

            init_fields(
                simulation,
                self.control,
                self.domain,
                self.fields,
                fluid=self.fluid,
                phase=self.phase,
                scalar=self.scalar,
                verbose=True
            )

            self.boundary = boundary.Boundary(
                simulation,
                self.mesh,
                self.domain,
                self.control,
                self.fields,
                fluid=self.fluid,
                phase=self.phase,
                scalar=self.scalar,
                verbose=True
            )

            self.obstacle = obstacle.Obstacle(
                simulation,
                self.mesh,
                self.domain,
                self.control,
                self.fields,
                self.boundary,
                fluid=self.fluid,
                phase=self.phase,
                scalar=self.scalar,
                verbose=True
            )

        except Exception as e:
            print_log("-" * 80, mpi_rank, verbose=True)
            print_log("FATAL ERROR!", mpi_rank, verbose=True)
            print_log(str(e), mpi_rank, verbose=True)
            comm.Abort()
