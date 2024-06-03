import matplotlib.pyplot as plt
import numpy as np

from Python_23_24.biblioteca import interpol_hermite as her
from Python_23_24.biblioteca import interpol_newton as new

x = np.array([1.5, 2.7, 3.1, -2.1, -6.6, 11.0])
y = np.array([0.0, 1.0, -0.5, 1.0, 0.5, 0.0])

coefs1 = new.coef_newton(x, y)
print(coefs1)
p1 = new.polinomio_newton(coefs1, x)
print(p1)

sample = np.linspace(-7, 11.5, 1001)

fig1, ax = plt.subplots()
ax.plot(sample, new.polyinterpolador_eval(coefs1, x, sample))
ax.plot(x, y, 'o')

a = [0, 1, 2, 3]
b = [[1, 2], [0, 1, 1], [3], [1, 1]]
r = [1, 0, 3, 1]

coefs2, nodos = her.coef_hermite(a, b)
p2 = new.polinomio_newton(coefs2, nodos)

print(p2)

sample2 = np.linspace(-1, 3.5, 1001)

fig2, ax2 = plt.subplots()
ax2.plot(sample2, new.polyinterpolador_eval(coefs2, nodos, sample2))
ax2.plot(a, r, 'o')

plt.show()
