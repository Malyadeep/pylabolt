import matplotlib.pyplot as plt
import numpy as np

Nx, Ny = 1001, 101
x, y = np.mgrid[0:1000:1001j, 0:100:101j]


for t in range(49900, 49901):
    f = open('output/' + str(t) + '/fields.dat', 'r')
    u = np.zeros((Nx, Ny, 2), dtype=np.float64)
    rho = np.zeros((Nx, Ny), dtype=np.float64)
    solid = np.zeros((Nx, Ny), dtype=np.int32)
    for line in f:
        data = line.split()
        ind = int(data[0])
        i, j = int(ind / Ny), int(ind % Ny)
        # print(i, j)
        u[i, j, 0] = float(data[4])/0.1
        u[i, j, 1] = float(data[5])/0.1
        rho[i, j] = float(data[3])
        solid[i, j] = int(data[6])
    f.close()
    # print(u[:, :, 0])
    # print(u[:, :, 1])
    # print(rho[:, :])
    plt.figure(1, figsize=(7, 6))
    plt.contourf(x, y, u[:, :, 0])
    plt.colorbar()
    plt.savefig('images/u_' + str(t) + '.png')
    plt.close()

    plt.figure(2, figsize=(7, 6))
    plt.contourf(x, y, u[:, :, 1])
    plt.colorbar()
    plt.savefig('images/v_' + str(t) + '.png')
    plt.close()

    plt.figure(3, figsize=(7, 6))
    plt.contourf(x, y, rho)
    plt.colorbar()
    plt.savefig('images/rho_' + str(t) + '.png')
    plt.close()

    plt.figure(4, figsize=(6, 6))
    plt.plot(u[int(Nx/2), :, 0], np.arange(Ny)/Ny)
    # plt.scatter(u_g, y_g, c='r')
    plt.savefig('images/uLine_' + str(t) + '.png')
    plt.close()

    plt.figure(5, figsize=(6, 6))
    plt.plot(np.arange(Nx)/Nx, u[:, int(Ny/2), 1])
    # plt.scatter(x_g, v_g, c='r')
    plt.savefig('images/vLine_' + str(t) + '.png')
    plt.close()

    plt.figure(6, figsize=(6, 6))
    plt.contourf(x, y, solid)
    plt.colorbar()
    plt.savefig('images/solid_' + str(t) + '.png')
    plt.close()
