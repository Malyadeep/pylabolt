import numpy as np
import numba


class Domain:
    def __init__(
        self,
        simulation,
        mesh,
        comm,
        verbose=True
    ):
        """
        Container for decomposed sub-domain description
        Attributes:
            mpi_rank: int
            mpi_size: int
            no_of_procs_x: int
            no_of_procs_y: int
            i_proc: int
            j_proc: int
            Nx_rank: int
            Ny_rank: int
            offset: (mesh.dimensions,) int array
            Nx_rank_padded: int
            Ny_rank_padded: int
            shape: (mesh.dimensions,) int array
            size: int
        """
        self.mpi_rank = comm.Get_rank()
        self.mpi_size = comm.Get_size()

        if not hasattr(simulation, "decompose_dict"):
            raise ValueError(
                "decompose_dict not found in simulation.py file"
            )
        decompose_dict = simulation.decompose_dict

        if ("nx" not in decompose_dict or
                "ny" not in decompose_dict):
            raise ValueError("nx or ny missing decompose_dict")
        self.no_of_procs_x = decompose_dict["nx"]
        self.no_of_procs_y = decompose_dict["ny"]

        if self.mpi_size != self.no_of_procs_x * self.no_of_procs_y:
            raise ValueError(
                "invalid domain decomposition. " +
                "nx * ny not equal to total no.of MPI processes"
            )

        self.i_proc = self.mpi_rank // self.no_of_procs_y
        self.j_proc = self.mpi_rank % self.no_of_procs_y
        offset_x, offset_y = 0, 0
        if self.i_proc != self.no_of_procs_x - 1:
            self.Nx_rank = int(
                np.ceil(mesh.grid_global_shape[0] / self.no_of_procs_x)
            )
            offset_x = self.i_proc * self.Nx_rank
        else:
            self.Nx_rank = int(
                mesh.grid_global_shape[0] - self.i_proc *
                np.ceil(mesh.grid_global_shape[0] / self.no_of_procs_x)
            )
            offset_x = self.i_proc * int(
                np.ceil(mesh.grid_global_shape[0] / self.no_of_procs_x)
            )
        if self.j_proc != self.no_of_procs_y - 1:
            self.Ny_rank = int(
                np.ceil(mesh.grid_global_shape[1] / self.no_of_procs_y)
            )
            offset_y = self.j_proc * self.Ny_rank
        else:
            self.Ny_rank = int(
                mesh.grid_global_shape[1] - self.j_proc *
                np.ceil(mesh.grid_global_shape[1] / self.no_of_procs_y)
            )
            offset_y = self.j_proc * int(
                np.ceil(mesh.grid_global_shape[1] / self.no_of_procs_y)
            )
        self.offset = np.array([offset_x, offset_y], dtype=np.int32)
        self.Nx_pad = self.Nx_rank + 2
        self.Ny_pad = self.Ny_rank + 2
        self.shape = np.array([self.Nx_pad, self.Ny_pad])
        self.size = np.prod(self.shape)
        self.inner_shape = np.array([self.Nx_rank, self.Ny_rank])
        self.inner_size = np.prod(self.inner_shape)

    def set_backend(
        self,
        backend
    ):
        """
        Configure backend attributes for domain object
        Args:

        Returns:

        """
        if backend.backend_type == "gpu":
            self._device_attrs = ["size", "shape"]
            for arg_name in self._device_attrs:
                arg_device = backend.allocate_to_device(
                    getattr(self, arg_name)
                )
                setattr(self, arg_name + "_device", arg_device)


@numba.njit(inline="always")
def local_to_global(i, j, offset):
    """
    convert local sub-domain index to global index
    Args:
        i: int
        j: int
        offset: (mesh.dimensions,) int array
    Returns:
        i_global: int
        j_global: int
    """
    return i + offset[0], j + offset[1]


@numba.njit(inline="always")
def global_to_local(i_global, j_global, offset):
    """
    convert global index to local sub-domain index
    Args:
        i_global: int
        j_global: int
        offset: (mesh.dimensions,) int array
    Returns:
        i: int
        j: int
    """
    return i_global - offset[0], j_global - offset[1]
