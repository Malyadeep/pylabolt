import numba


@numba.njit
def stream(elements, lattice, mesh):
    dtdx = lattice.dtdx
    Nx = mesh.Nx
    Ny = mesh.Ny
    for i in range(Nx):
        for j in range(Ny):
            ind = int(i * Nx + j)
            if elements[ind].nodeType != 'o':
                for k in range(1, lattice.c.shape[1]):
                    i_old = (i - int(lattice.c[k, 0]*dtdx)
                             + Nx) % Nx
                    j_old = (j - int(lattice.c[k, 1]*dtdx)
                             + Ny) % Ny
                    elements[ind].f_new[k] = elements[i_old * Nx + j_old].f[k]
