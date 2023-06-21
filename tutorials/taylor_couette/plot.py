import matplotlib.pyplot as plt
import numpy as np
import os
from argparse import ArgumentParser


def analytical(Nx, Ny, r1, r2, omega1, omega2):
    delX = 1 / Nx
    r1 = r1 / delX
    r2 = r2 / delX
    r_an = np.linspace(0.2 / delX, 0.4 / delX, Nx)
    omega_an = np.zeros(101, dtype=np.float64)
    mu, eta = omega2/omega1, r1/r2
    A = omega1 * ((mu - eta**2) / (1 - eta**2))
    B = omega1 * r1 * r1 * ((1 - mu) / (1 - eta**2))
    for i in range(r_an.shape[0]):
        omega_an[i] = (A + B / (r_an[i] * r_an[i])) / omega1
    r_an = r_an / r1
    return r_an, omega_an


def extract_data_taylorCouette(Nx, time):
    Ny = Nx
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
    return u


def convertCylindrical(u, Nx):
    Ny = Nx
    center = (Nx - 1)/2
    r = np.zeros((Nx, Ny), dtype=np.float64)
    theta = np.zeros((Nx, Ny), dtype=np.float64)
    omega = np.zeros((Nx, Ny), dtype=np.float64)
    for i in range(Nx):
        for j in range(Ny):
            r[i, j] = np.sqrt((i - center)**2 + (j - center)**2)
            theta[i, j] = np.arctan2((j - center), (i - center))
            u_theta = np.multiply(u[i, j, 1], np.cos(theta[i, j])) \
                - np.multiply(u[i, j, 0], np.sin(theta[i, j]))
            omega[i, j] = u_theta / r[i, j]
    return r, omega


def plot_U_taylorCouette(r_an, omega_an, r, omega, omega1,
                         r1, Nx, time):
    if not os.path.isdir('figures'):
        os.makedirs('figures')
    plt.figure(1, figsize=(6, 6))
    delX = 1/(Nx - 1)
    plt.scatter(r[int((Nx - 1)/2) + 1:, int((Nx - 1)/2)] / (r1/delX),
                omega[int((Nx - 1)/2) + 1:, int((Nx - 1)/2)] / omega1,
                edgecolors='r', facecolors='none',
                marker='d', label='PyLaBolt')
    plt.plot(r_an, omega_an, c='b', label='Analytical')
    plt.legend(loc='center right')
    plt.ylabel(r'$\frac{\Omega(r)}{\Omega_1}$')
    plt.xlabel(r'$r/r_1$')
    plt.title('Taylor-Couette Flow (angular velocity vs r)')
    plt.savefig('figures/omega_vs_r_Nx_' + str(Nx) + '_time_' +
                str(time) + '.png')
    plt.close()


def main():
    parser = ArgumentParser()
    parser.add_argument('-t', '--time', type=int,
                        default=10000)
    parser.add_argument('-Nx', '--Nx', type=int,
                        default=17, help='grid')
    args = parser.parse_args()
    Nx = args.Nx
    time = args.time
    omega1, omega2 = 1e-5, 4e-5
    r1, r2 = 0.2, 0.4
    r_an, omega_an = analytical(101, 101, r1, r2, omega1, omega2)
    u = extract_data_taylorCouette(Nx, time)
    r, omega = convertCylindrical(u, Nx)
    plot_U_taylorCouette(r_an, omega_an, r, omega,
                         omega1, r1, Nx, time)


if __name__ == '__main__':
    main()
