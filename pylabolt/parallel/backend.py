import numpy as np
import numba


class Backend:
    def __init__(
        self,
        backend,
        n_threads,
        verbose=True
    ):
        """
        Container for hardware backend information
        Attributes:

        """
        self.backend_type = backend
        self.no_of_threads = n_threads
        if self.backend_type == "cpu":
            numba.set_num_threads(self.no_of_threads)

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
