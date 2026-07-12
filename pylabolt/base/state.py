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

            # self.set_backend(backend)
            # self.boundary.view_attrs(mpi_rank)
            # var_names = vars(self.fields)
            # for item in var_names:
            #     print(f"{item:<30}: {var_names[item]}")

        except Exception as e:
            print_log("-" * 80, mpi_rank, verbose=True)
            print_log("FATAL ERROR!", mpi_rank, verbose=True)
            print_log(str(e), mpi_rank, verbose=True)
            comm.Abort()

    def set_backend(
        self,
        backend,
        verbose=True
    ):
        """
        Configure backend attributes for state object
        Args:

        Returns:

        """
        self.control.set_backend(backend)
        self.mesh.set_backend(backend)
        self.lattice.set_backend(backend)
        self.domain.set_backend(backend)
        self.transport.set_backend(backend)
        self.fields.set_backend(backend)
        self.boundary.set_backend(backend)
        print_log("Backend set for state", self.domain.mpi_rank, verbose)
