import matplotlib.pyplot as plt
import numpy as np
import os
from argparse import ArgumentParser


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


def extract_data_cavity(Nx, Ny, time):
    delY = 2/(Ny - 1)
    y_i, y_f = 0 + delY/2, 1 - delY/2
    x_i, x_f = 0 + delY/2, 1 - delY/2
    y_sim = np.linspace(y_i, y_f, Ny)
    x_sim = np.linspace(x_i, x_f, Nx)
    f = open('output/' + str(time) + '/fields.dat', 'r')
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
    return x_sim, y_sim, u


def plot_cavity(Nx, Ny, x_sim, y_sim, u, y_g, u_g, x_g, v_g, U_lid):
    if not os.path.isdir('figures'):
        os.makedirs('figures')
    plt.figure(1, figsize=(6, 6))
    plt.scatter(u[int((Nx - 1)/2), :, 0] / U_lid,
                y_sim, edgecolors='blue', facecolors='none',
                marker='d', label='PyLaBolt')
    plt.plot(u_g, y_g, c='r', label='Ghia (1982)')
    plt.legend(loc='lower right')
    plt.xlabel(r'$U/U_{max}$')
    plt.ylabel(r'$Y$')
    plt.title('Lid driven cavity - U vs y (x = 0.5) - Re = 100')
    plt.savefig('figures/Validation_UY.png')
    plt.close()

    plt.figure(2, figsize=(6, 6))
    plt.scatter(x_sim, u[:, int((Ny - 1)/2), 1] / U_lid,
                edgecolors='blue', facecolors='none',
                marker='d', label='PyLaBolt')
    plt.plot(x_g, v_g, c='r', label='Ghia (1982)')
    plt.legend(loc='lower left')
    plt.xlabel(r'$X$')
    plt.ylabel(r'$V/U_{max}$')
    plt.title('Lid driven cavity - V vs x (y = 0.5) - Re = 100')
    plt.savefig('figures/Validation_XV.png')
    plt.close()


def main():
    parser = ArgumentParser(description='plot cavity')
    parser.add_argument('-t', '--time', type=int, default=0, help='timestep')
    parser.add_argument('-Nx', type=int, default=0, help='timestep')
    args = parser.parse_args()
    time = args.time
    Nx = args.Nx
    Ny = Nx
    U_lid = 0.1
    x_sim, y_sim, u = extract_data_cavity(Nx, Ny, time)
    y_g, u_g, x_g, v_g = ghia()
    plot_cavity(Nx, Ny, x_sim, y_sim, u, y_g, u_g, x_g, v_g, U_lid)


if __name__ == '__main__':
    main()
