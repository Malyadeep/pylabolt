import numpy as np
import numba
from numba import cuda

from pylabolt.utils.helpers import print_log


class Backend:
    def __init__(
        self,
        backend,
        n_threads,
        mpi_rank,
        verbose=True
    ):
        """
        Container for hardware backend information
        Attributes:

        """
        self.backend_type = backend
        self.no_of_threads = n_threads
        print_log(f"{'Backend':<10}: {self.backend_type}",
                  mpi_rank, verbose=verbose)
        if self.backend_type == "cpu":
            numba.set_num_threads(self.no_of_threads)
            print_log(f"{'Threads-per-rank':<10}: {self.no_of_threads}",
                      mpi_rank, verbose=verbose)
        if self.backend_type == "gpu":
            print_log("GPU info:", mpi_rank, verbose=verbose)
            cuda.detect()

    def make_compile_args(
        self,
        args
    ):
        """
        Make disposable compilation args for kernels
        Args:

        Returns:

        """
        compile_args = []
        if self.backend_type == "cpu":
            for arg in args:
                if isinstance(arg, np.ndarray):
                    compile_args.append(np.copy(arg))
                else:
                    compile_args.append(arg)
        compile_args = tuple(compile_args)
        return compile_args

    def allocate_to_device(
        self,
        arg
    ):
        """
        Allocate device memory for given host argument
        Args:

        Returns:

        """
        if isinstance(arg, np.ndarray):
            return cuda.to_device(arg)
        else:
            return arg
