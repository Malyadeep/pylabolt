import matplotlib.pyplot as plt
import numpy as np


def analytical(Nx, Ny):
    u_an = np.zeros(Ny, dtype=np.float64)
    y_an = np.linspace(-1, 1, 41)
    for i in range(u_an.shape[0]):
        u_an[i] = (1 - y_an[i]**2)
    return y_an, u_an


def extract_data(Nx, Ny):
    cases = ['incompressible', 'compressible']
    u_all = []
    for case in cases:
        delY = 2/(Ny - 1)
        y_i, y_f = -1 + delY/2, 1 - delY/2
        y_sim = np.linspace(y_i, y_f, Ny)
        f = open('fields_' + case + '.dat', 'r')
        u = np.zeros((Nx, Ny, 2), dtype=np.float64)
        rho = np.zeros((Nx, Ny), dtype=np.float64)
        solid = np.zeros((Nx, Ny), dtype=np.int32)
        for line in f:
            data = line.split()
            ind = int(data[0])
            i, j = int(ind / Ny), int(ind % Ny)
            u[i, j, 0] = float(data[4])
            u[i, j, 1] = float(data[5])
            rho[i, j] = float(data[3])
            solid[i, j] = int(data[6])
        f.close()
        u_all.append(u)
    return y_sim, u_all


def plot(Nx, Ny, y_sim, u_all, y_an, u_an):
    cases = ['incompressible', 'compressible']
    markers = ['d', 's']
    colors = ['green', 'blue']
    plt.figure(1, figsize=(6, 6))
    for itr, case in enumerate(cases):
        plt.scatter(u_all[itr][int(Nx - 1), :, 0] /
                    np.max(u_all[itr][int(Nx - 1), :, 0]),
                    y_sim, edgecolors=colors[itr], facecolors='none',
                    marker=markers[itr], label='PyLaBolt - ' + case)
    plt.plot(u_an, y_an, c='r', label='Analytical')
    plt.legend(loc='center left')
    plt.xlabel('U')
    plt.ylabel('Y')
    plt.title('Plane Poiseulle Flow - Re = 100')
    plt.savefig('uLine.png')
    plt.close()


def main():
    Nx, Ny = 121, 41
    y_sim, u_all = extract_data(Nx, Ny)
    y_an, u_an = analytical(Nx, Ny)
    plot(Nx, Ny, y_sim, u_all, y_an, u_an)


if __name__ == '__main__':
    main()
