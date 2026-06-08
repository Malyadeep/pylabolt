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
