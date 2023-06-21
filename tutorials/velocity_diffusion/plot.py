import numpy as np
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser

from simulation import internalFields, collisionDict


def extract(t, Nx, Ny, v_max):
    f = open('output/' + str(t) + '/fields.dat', 'r')
    v_grid = np.zeros((Nx, Ny), dtype=np.float64)
    for line in f:
        data = line.split()
        ind = int(data[0])
        i, j = int(ind / Ny), int(ind % Nx)
        v_grid[i, j] = float(data[5])/v_max
    f.close()
    v_all = v_grid[:, 50]
    x_all = np.linspace(0, 1, Nx)
    return v_all, x_all


def analytical(x_all, nu, t, Nx):
    x_0 = int(Nx/2)
    x = np.array(x_all) * (Nx - 1)
    preFactor = 1/(2 * np.sqrt(np.pi * nu * t))
    v_ana = preFactor * np.exp(- np.divide(np.power(x - x_0, 2), 4 * nu * t))
    return v_ana


def plot_all(x_all, v_all, v_ana, t, nu):
    if not os.path.isdir('figures'):
        os.makedirs('figures')
    plt.figure(1, figsize=(6, 6))
    ax = plt.gca()
    ax.scatter(x_all, v_all, label='PyLaBolt', marker='d',
               facecolors='none', edgecolors='red')
    ax.plot(x_all, v_ana, label='analytical')
    ax.legend(loc='best')
    ax.set_xlabel('x')
    ax.set_ylabel('v')
    ax.set_xlim([0, 1])
    ax.set_title('v vs x (time = ' + str(t) + ')')
    plt.savefig('figures/v_vs_x_' + str(t) + '.png')
    plt.close()


def main():
    parser = ArgumentParser(description='plot velocity diffusion cases')
    parser.add_argument('-t', '--time', type=int, default=0, help='timestep')
    args = parser.parse_args()
    t = args.time
    v_max = float(internalFields['region_0']['fields']['v'])
    nu = float(collisionDict['nu'])
    # Ma = np.round(v_max/cs, 2)
    Nx, Ny = 101, 101
    v_all, x_all = extract(t, Nx, Ny, v_max)
    v_ana = analytical(x_all, nu, t, Nx)
    plot_all(x_all, v_all, v_ana, t, nu)


if __name__ == '__main__':
    main()
