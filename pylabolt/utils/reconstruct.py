import numpy as np
import numba
from numba import prange
import json
import os
from types import SimpleNamespace

from pylabolt.utils.helpers import print_log
from pylabolt.parallel.domain import local_to_global


@numba.njit(parallel=True, nogil=True)
def copy_scalar(
    size,
    shape,
    offset,
    global_shape,
    field_global,
    field_local
):
    """
    Kernel to copy rank data to global data
    for scalar field
    Args:

    Returns:

    """
    Ny_local = shape[1]
    Ny_global = global_shape[1]
    for ind in prange(size):
        i = ind // Ny_local
        j = ind - i * Ny_local
        i_global, j_global = local_to_global(i, j, offset)
        ind_global = i_global * Ny_global + j_global
        field_global[ind_global] = field_local[ind]


@numba.njit(parallel=True, nogil=True)
def copy_vector(
    size,
    shape,
    offset,
    global_shape,
    field_global,
    field_local
):
    """
    Kernel to copy rank data to global data
    for scalar field
    Args:

    Returns:

    """
    Ny_local = shape[1]
    Ny_global = global_shape[1]
    for ind in prange(size):
        i = ind // Ny_local
        j = ind - i * Ny_local
        i_global, j_global = local_to_global(i, j, offset)
        ind_global = i_global * Ny_global + j_global
        field_global[ind_global, 0] = field_local[ind, 0]
        field_global[ind_global, 1] = field_local[ind, 1]


class ReconstructOperator:
    def __init__(
        self,
        metadata,
        option,
        time=0,
        verbose=True
    ):
        """
        Reconstructs saved data from parallel simulation
        Attributes:

        """
        self.metadata = metadata
        self.validate_metadata()
        self.validate_rank_metadata()

    def validate_metadata(
        self,
        verbose=True
    ):
        """
        Read and validate simulation metadata
        Args:

        Returns:

        """
        try:
            self.control = SimpleNamespace(
                end_time=self.metadata["control"]["end_time"],
                start_time=self.metadata["control"]["start_time"],
                save_interval=self.metadata["control"]["save_interval"],
                checkpoint_interval=self.metadata["control"]
                ["checkpoint_interval"]
            )
            self.control.no_of_snaps = (
                self.control.end_time - self.control.start_time
            ) // self.control.save_interval + 1
            self.control.save_times = np.linspace(
                0, self.control.end_time, self.control.no_of_snaps
            )
            self.mesh = SimpleNamespace(
                size=self.metadata["mesh"]["size"],
                shape=np.array(self.metadata["mesh"]["shape"], dtype=np.int64)
            )
            self.decomposition = SimpleNamespace(
                nx=self.metadata["decomposition"]["nx"],
                ny=self.metadata["decomposition"]["ny"],
            )
            self.decomposition.mpi_size = (
                self.decomposition.nx *
                self.decomposition.ny
            )
            self.fields_metadata = self.metadata["fields_saved"]
            self.fields = SimpleNamespace()
            print_log("\nFields to reconstruct:", 0, verbose)
            for field_name in self.fields_metadata:
                components = self.fields_metadata[field_name]["components"]
                dtype = self.fields_metadata[field_name]["dtype"]
                if components == 1:
                    field = np.zeros(self.mesh.size, dtype=dtype)
                else:
                    field = np.zeros(
                        (self.mesh.size, components),
                        dtype=dtype
                    )
                setattr(self.fields, field_name, field)
                print_log(
                    f"{field_name:<10}: {str(field.dtype):>5}", 0, verbose
                )
            print_log("\n", 0, verbose)
        except KeyError as e:
            print_log("Invalid metadata.json! missing key: " + str(e),
                      0, verbose)

    def validate_rank_metadata(
        self,
        verbose=True
    ):
        """
        Read and validate individual rank metadata
        Args:

        Returns:

        """
        self.domains = []
        for rank in range(self.decomposition.mpi_size):
            dir_path = "procs/proc_" + str(rank) + "/"
            if not os.path.isdir(dir_path):
                raise FileNotFoundError(
                    "processor directory not found in procs/ for rank: " +
                    str(rank)
                )
            if not os.path.isfile(dir_path + "rank_metadata.json"):
                raise FileNotFoundError(
                    "rank_metadata.json file not found for rank: " + str(rank)
                )
            with open(dir_path + "rank_metadata.json") as metadata_file:
                rank_metadata = json.load(metadata_file)
            try:
                domain_data = SimpleNamespace(
                    rank=rank_metadata["rank"],
                    processor_ij=np.array(
                        rank_metadata["processor_ij"], dtype=np.int64
                    ),
                    size=rank_metadata["domain_size"],
                    shape=np.array(
                        rank_metadata["domain_shape"], dtype=np.int64
                    ),
                    offset=np.array(
                        rank_metadata["offset"], dtype=np.int64
                    )
                )
                self.domains.append(domain_data)
            except KeyError as e:
                print_log("Invalid rank_metadata.json! missing key: " + str(e)
                          + " || rank: " + str(rank), 0, verbose=verbose)

    def reconstruct_time(
        self,
        time_step,
        verbose=True
    ):
        """
        Reconstruct single time step data
        Args:

        Returns:

        """
        for rank in range(self.decomposition.mpi_size):
            file_path = "procs/proc_" + str(rank) + "/t_" +\
                str(time_step) + ".npz"
            domain = self.domains[rank]
            if not os.path.isfile(file_path):
                raise FileNotFoundError(
                    "output file not found for time step: " + str(time_step) +
                    " || rank: " + str(rank)
                )
            fields_rank = np.load(file_path)
            try:
                for field_name in self.fields_metadata:
                    field_local = fields_rank[field_name]
                    components = self.fields_metadata[field_name]
                    if components == 1:
                        copy_scalar(
                            domain.size,
                            domain.shape,
                            domain.offset,
                            self.mesh.shape,
                            getattr(self.fields, field_name),
                            field_local
                        )
                    elif components == 2:
                        copy_vector(
                            domain.size,
                            domain.shape,
                            domain.offset,
                            self.mesh.shape,
                            getattr(self.fields, field_name),
                            field_local
                        )
            except KeyError as e:
                print_log(
                    f"{'missing field':<10}: {str(e):<20}"
                    f"{'time':<10}: {time_step:<20}"
                    f"{'rank':<10}: {rank:<20}"
                )

        print_log(
            f"{'Reconstruction done, time':<25}:"
            f"{time_step:>5}", 0, verbose
        )

    def compile(
        self,
        state,
        backend,
        verbose=True
    ):
        """
        JIT compile compute fields kernels
        Args:

        Returns:

        """
        for item in self.residues:
            args = (
                state.domain.size,
                state.fields.solid,
                state.fields.ghost_node,
                getattr(state.fields, item),
                getattr(self, item + "_old"),
            )
            compile_args = backend.make_compile_args(args)
            if len(self.residues[item]) == 1:
                self.compute_residues_kernel_scalar(*compile_args)
            elif len(self.residues[item]) == 2:
                self.compute_residues_kernel_vector(*compile_args)
        print_log("Compiled compute residues operator",
                  state.domain.mpi_rank, verbose)

    def compute_residues_cpu(
        self,
        state
    ):
        """
        Compute residuals on CPU kernels
        Args:

        Returns:

        """
        for item in self.residues:
            args = (
                state.domain.size,
                state.fields.solid,
                state.fields.ghost_node,
                getattr(state.fields, item),
                getattr(self, item + "_old")
            )
            if len(self.residues[item]) == 1:
                self.compute_residues_kernel_scalar(*args)
            elif len(self.residues[item]) == 2:
                self.compute_residues_kernel_vector(*args)

    def compute_residues_gpu(
        self,
        state
    ):
        """
        Compute residuals on GPU kernels
        Args:

        Returns:

        """
        pass


def reconstruct_data(
    option,
    time=0,
    verbose=True
):
    """
    Reconstructs saved data from parallel simulation
    Args:

    Returns:

    """
    print_log("-" * 80, 0, verbose)
    print_log("Reconstructing fields...\n", 0, verbose)
    supported_options = ["all", "time"]
    if option not in supported_options:
        raise ValueError(
            "Invalid reconstruction option!\n" +
            "Available options: " + str(supported_options)
        )
    print_log(f"{'Reconstruction option':<20}: {option:<20}", 0, verbose)
    if option == "time":
        if not isinstance(time, int):
            raise ValueError(
                "Reconstruction time must be an int!"
            )
        print_log(f"{'Reconstruction time':<20}: {time:<20}", 0, verbose)

    if not os.path.isfile("metadata.json"):
        raise FileNotFoundError(
            "metadata.json file not found in working directory"
        )
    with open("metadata.json") as metadata_file:
        print_log("Reading simulation metadata...", 0, verbose)
        metadata = json.load(metadata_file)
    if not os.path.isdir("procs"):
        raise FileNotFoundError(
            "procs/ not a directory in working directory"
        )

    reconstruct_operator = ReconstructOperator(
        metadata,
        option,
        time=time,
        verbose=verbose
    )

    reconstruct_operator.reconstruct_time(0)
