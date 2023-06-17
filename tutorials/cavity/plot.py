import matplotlib.pyplot as plt
import numpy as np


def ghia():
    u_g = []
    y_g = []
    ghia = open('Ghia-Ghia_U_Re100.dat', 'r')
    for line in ghia:
        data = line.split()
        u_g.append(float(data[0]))
        y_g.append(float(data[1]))

    v_g = []
    x_g = []
    ghia = open('Ghia-Ghia_V_Re100.dat', 'r')
    for line in ghia:
        data = line.split()
        v_g.append(float(data[1]))
        x_g.append(float(data[0]))
    return y_g, u_g, x_g, v_g


def extract_data_poiseuille(Nx, Ny):
    cases = ['']
    u_all = []
    for case in cases:
        delY = 2/(Ny - 1)
        y_i, y_f = 0 + delY/2, 1 - delY/2
        x_i, x_f = 0 + delY/2, 1 - delY/2
        y_sim = np.linspace(y_i, y_f, Ny)
        x_sim = np.linspace(x_i, x_f, Nx)
        f = open('fields' + case + '.dat', 'r')
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
    return x_sim, y_sim, u_all


def plot_cavity(Nx, Ny, x_sim, y_sim, u_all, y_g, u_g, x_g, v_g):
    cases = ['']
    markers = ['d', 's', '^']
    colors = ['green', 'blue', 'orange']
    plt.figure(1, figsize=(6, 6))
    for itr, case in enumerate(cases):
        plt.scatter(u_all[itr][int((Nx - 1)/2), :, 0] / 0.1,
                    y_sim, edgecolors=colors[itr], facecolors='none',
                    marker=markers[itr], label='PyLaBolt' + case)
    plt.plot(u_g, y_g, c='r', label='Ghia (1982)')
    plt.legend(loc='lower right')
    plt.xlabel(r'$U/U_{max}$')
    plt.ylabel(r'$Y$')
    plt.title('Lid driven cavity - U vs y (x = 0.5) - Re = 100')
    plt.savefig('Validation_UY.png')
    plt.close()

    plt.figure(2, figsize=(6, 6))
    for itr, case in enumerate(cases):
        plt.scatter(x_sim, u_all[itr][:, int((Ny - 1)/2), 1] / 0.1,
                    edgecolors=colors[itr], facecolors='none',
                    marker=markers[itr], label='PyLaBolt' + case)
    plt.plot(x_g, v_g, c='r', label='Ghia (1982)')
    plt.legend(loc='lower left')
    plt.xlabel(r'$X$')
    plt.ylabel(r'$V/U_{max}$')
    plt.title('Lid driven cavity - V vs x (y = 0.5) - Re = 100')
    plt.savefig('Validation_XV.png')
    plt.close()


def main():
    Nx, Ny = 101, 101
    x_sim, y_sim, u_all = extract_data_poiseuille(Nx, Ny)
    y_g, u_g, x_g, v_g = ghia()
    plot_cavity(Nx, Ny, x_sim, y_sim, u_all, y_g, u_g, x_g, v_g)


if __name__ == '__main__':
    main()
