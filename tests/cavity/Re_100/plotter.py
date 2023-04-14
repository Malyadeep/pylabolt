import numpy as np
import matplotlib.pyplot as plt
file = open('fields.dat')
lst = []
for line in file:
    lst += [line.split()]

X = [x[1] for x in lst]
Y = [x[2] for x in lst]
U = [x[4] for x in lst]
V = [x[5] for x in lst]

indices = [i for i, x in enumerate(X) if x == "0.5"]

y_plt = np.array([Y[i] for i in indices])
y_plt=y_plt.astype(np.float64)
u_plt = np.array([U[i] for i in indices])
u_plt=u_plt.astype(np.float64)


plt.figure(1)
plt.plot(u_plt/0.1,y_plt)

file = open('ghia_data_YU_Re100.dat')
lst = []
for line in file:
    lst += [line.split()]

Y = np.array([x[0] for x in lst])
U = np.array([x[1] for x in lst])
Y = Y.astype(np.float64)
U = U.astype(np.float64)
plt.plot(U,Y,'*')
plt.xlabel('U')
plt.ylabel('Y')
plt.title('Velocity Profile at X=0.5')
plt.legend(['LBpy','Ghia(1982)'])
plt.savefig('Validation_UY.png')


plt.figure(2)
file = open('fields.dat')
lst = []
for line in file:
    lst += [line.split()]

X = [x[1] for x in lst]
Y = [x[2] for x in lst]
U = [x[4] for x in lst]
V = [x[5] for x in lst]

indices = [i for i, y in enumerate(Y) if y == "0.5"]
x_plt = np.array([Y[i] for i in indices])
x_plt=y_plt.astype(np.float64)
v_plt = np.array([V[i] for i in indices])
v_plt= v_plt.astype(np.float64)
plt.plot(x_plt,v_plt/0.1,)
file = open('ghia_data_XV_Re100.dat')
lst = []
for line in file:
    lst += [line.split()]

X = np.array([x[0] for x in lst])
V = np.array([x[1] for x in lst])
X = X.astype(np.float64)
V = V.astype(np.float64)
plt.plot(X,V,'*')
plt.xlabel('X')
plt.ylabel('V')
plt.title('Velocity Profile at Y=0.5')
plt.legend(['LBpy','Ghia(1982)'])
plt.savefig('Validation_XV.png')
plt.show()
