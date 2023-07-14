import numpy as np
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser


def extractLBM(Nx, Ny, t):
    delY = 1/(Ny - 1)
    y_i, y_f = 0 + delY/2, 1 - delY/2
    y_sim = np.linspace(y_i, y_f, Ny)
    f = open('output/' + str(t) + '/fields.dat', 'r')
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
    return u, y_sim


def extractTorque(start, step, end):
    t = np.linspace(start, end, int((end - start)/step))
    T = []
    omega = []
    for i in range(start, end, step):
        readFile = open('postProcessing/' + str(i) +
                        '/obstacleProperties.dat', 'r')
        lineList = readFile.readlines()
        data = lineList[1].split()
        T.append(float(data[4]))
        omega.append(float(data[5]))
    T = np.array(T, dtype=np.float64)
    omega = np.array(omega, dtype=np.float64)
    return t, T, omega


def plot_all(u, y_sim, ul, Re, pos):
    if not os.path.isdir('figures'):
        os.makedirs('figures')
    plt.figure(1, figsize=(6, 6))
    ax = plt.gca()
    ax.plot(u[pos, :, 0]/ul, y_sim, label='PyLaBolt', c='red')
    ax.legend(loc='best')
    ax.set_ylabel(r'$y/H$')
    ax.set_xlabel(r'$U/U_{max}$')
    ax.set_title('u vs y (Re = ' + str(Re) + ')')
    plt.savefig('figures/u_vs_y_' + str(pos) + '.png')
    plt.close()


def plotTorque(t, T, omega):
    if not os.path.isdir('figures'):
        os.makedirs('figures')
    plt.figure(1, figsize=(6, 6))
    ax = plt.gca()
    ax.plot(t, T, label='PyLaBolt', c='blue')
    ax.legend(loc='best')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$T$')
    ax.set_title('Torque vs time')
    plt.savefig('figures/T_vs_t_initiallyMovingCylinder.png')
    plt.close()
    plt.figure(1, figsize=(6, 6))
    ax = plt.gca()
    ax.plot(t, omega, label='PyLaBolt', c='blue')
    ax.legend(loc='best')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\omega$')
    ax.set_title('Angular velocity vs time')
    plt.savefig('figures/omega_vs_t_initiallyMovingCylinder.png')
    plt.close()


def main():
    parser = ArgumentParser(description='compare shearing cylinder')
    parser.add_argument('-Re', type=float, default=0.1, help='Reynolds number')
    parser.add_argument('-Nx', type=int, default=161,
                        help='system size in lattice units')
    parser.add_argument('-pos', type=int, default=0,
                        help='x position')
    parser.add_argument('-t', type=int, default=0,
                        help='time')
    args = parser.parse_args()
    from simulation import boundaryDict
    ul = boundaryDict['topPlate']['value'][0]
    pos = args.pos
    Re = args.Re
    Nx, Ny = args.Nx, args.Nx
    t = args.t
    u, y_sim = extractLBM(Nx, Ny, t)
    plot_all(u, y_sim, ul, Re, pos)
    start = 0
    step = 10
    end = 40000
    t, T_all, omega_all = extractTorque(start, step, end)
    plotTorque(t, T_all, omega_all)


if __name__ == '__main__':
    main()
