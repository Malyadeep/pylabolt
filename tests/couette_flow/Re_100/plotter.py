import numpy as np
import matplotlib.pyplot as plt
file = open('fields.dat')
lst = []
for line in file:
    lst += [line.split()]

X = [x[1] for x in lst]
Y = [x[2] for x in lst]
U = [x[4] for x in lst]

indices = [i for i, x in enumerate(X) if x == "0.5"]
y_plt = np.array([Y[i] for i in indices])
y_plt=y_plt.astype(np.float64)
u_plt = np.array([U[i] for i in indices])
u_plt=u_plt.astype(np.float64)
plt.plot(y_plt,u_plt/0.1)

y_an = np.linspace(0,1,20)
u_an = np.linspace(0,1,20)
plt.scatter(y_an,u_an,c='r')

plt.xlabel('Y')
plt.ylabel('U')
plt.title('Velocity Profile at X=0.5')
plt.legend(['LBpy','Analytical'])
plt.savefig('Validation_CouetteFlow.png')
plt.show()
