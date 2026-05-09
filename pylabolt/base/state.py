import os
import sys
import numpy as np

import pylabolt.base.control as control
import pylabolt.base.mesh as mesh
import pylabolt.base.lattice as lattice
import pylabolt.base.fields as fields
import pylabolt.backend.domain as domain

from pylabolt.base.init_fields import init_fields
from pylabolt.utils.IO import print_log


class State:
    def __init__(self, comm, rank, fluid=False,
                 phase=False, scalar=False):
        try:
            try:
                working_dir = os.getcwd()
                sys.path.append(working_dir)
                import simulation
            except ImportError:
                raise ImportError(
                    "Missing simulation.py file in current working directory"
                )

            self.control = control.Control(
                simulation,
                rank,
                verbose=True
            )

            self.mesh = mesh.Mesh(
                simulation,
                rank,
                verbose=True)

            self.lattice = lattice.Lattice(
                simulation,
                self.control,
                self.mesh,
                rank,
                verbose=True
            )

            self.domain = domain.Domain(
                simulation,
                self.mesh,
                comm
            )

            self.fields = fields.Fields(
                self.control,
                self.lattice,
                self.domain,
                fluid=fluid,
                phase=phase,
                scalar=scalar
            )

            init_fields(
                simulation,
                self.control,
                self.domain,
                self.fields,
                fluid=fluid,
                phase=phase,
                scalar=scalar,
                verbose=True
            )
        except Exception as e:
            print_log("-" * 80, rank, verbose=True)
            print_log("FATAL ERROR!", rank, verbose=True)
            print_log(str(e), rank, verbose=True)
            comm.Abort()

        """ Output initial fields for testing """
        if not os.path.isdir("procs"):
            os.makedirs("procs")
        np.savez(
            "procs/proc_" + str(self.domain.mpi_rank) + ".npz",
            velocity=self.fields.velocity,
            density=self.fields.density,
            pressure=self.fields.pressure,
            phase_field=self.fields.phase_field,
            Nx=self.mesh.grid_global_shape[0],
            Ny=self.mesh.grid_global_shape[1],
            offset=self.domain.offset,
            Nx_rank=self.domain.Nx_rank,
            Ny_rank=self.domain.Ny_rank
        )
        """ """
        self.obstacle = None
        self.boundary = None
