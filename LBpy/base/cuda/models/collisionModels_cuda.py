<<<<<<< HEAD
from numba import cuda


@cuda.jit(device=True)
=======
import numba


@numba.njit
>>>>>>> 763b1fb88aebd7534ec6dac5ae28097d575b24a7
def BGK(f, f_new, f_eq, omega):
    for k in range(f.shape[0]):
        f[k] = (1 - omega) * f_new[k] + omega * f_eq[k]
