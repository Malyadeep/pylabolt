import numpy as np
import numba
from numba import cuda

from pylabolt.utils.helpers import print_log


REDUCE_BLOCK_SIZE = 256


class Backend:
    def __init__(
        self,
        state,
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
        print_log(f"{'Backend':<25}: {self.backend_type}",
                  mpi_rank, verbose=verbose)
        if self.backend_type == "cpu":
            numba.set_num_threads(self.no_of_threads)
            print_log(f"{'Threads-per-rank':<25}: {self.no_of_threads}",
                      mpi_rank, verbose=verbose)
        if self.backend_type == "gpu":
            print_log("GPU info:", mpi_rank, verbose=verbose)
            cuda.detect()
            if n_threads != 1:
                self.threads_per_block = n_threads
                self.reduce_threads_per_block = REDUCE_BLOCK_SIZE
            else:
                self.threads_per_block = 256
                self.reduce_threads_per_block = REDUCE_BLOCK_SIZE
            self.blocks = int(
                    np.ceil(state.domain.size / self.threads_per_block)
                )
            self.reduce_blocks = int(
                    np.ceil(state.domain.size / self.reduce_threads_per_block)
                )
            print_log(f"{'Threads-per-block':<25}: {self.threads_per_block}",
                      mpi_rank, verbose=verbose)
            print_log(f"{'No-of-blocks':<25}: {self.blocks}",
                      mpi_rank, verbose=verbose)

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
        if self.backend_type == "gpu":
            for arg in args:
                if isinstance(arg, cuda.cudadrv.devicearray.DeviceNDArray):
                    temp = cuda.device_array_like(arg)
                    compile_args.append(temp)
                elif isinstance(arg, np.ndarray):
                    raise RuntimeError(
                        "Developer Error! Unallocated device array" +
                        " encountered during compilation"
                    )
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
