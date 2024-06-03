import matplotlib.pyplot as plt
import numpy as np

from Python_23_24.biblioteca.splines import spline_natural, spline_eval, spline_sujeto

sample = np.linspace(-1, 5, 1001)  # Nos va a valer para casi todos los apartados

# Apartado a)

x = np.array([0, 1, 2, 5])
y = np.array([0, 2, 4, 10])
b, c, d = spline_natural(x, y)

f1, ax1 = plt.subplots()
ax1.plot(sample, spline_eval(x, y, b, c, d, sample))
ax1.scatter(x, y, c="red")

# Apartado b)

x = np.array([0, 1, 2, 3])
y = np.array([0, 1, 8, 27])
tan_ini = 0
tan_fin = 27
b, c, d = spline_sujeto(x, y, tan_ini, tan_fin)

f2, ax2 = plt.subplots()
ax2.plot(sample, spline_eval(x, y, b, c, d, sample))
ax2.scatter(x, y, c="red")

# Apartado c)

x = np.array([0, 1, 2, 3])
y = np.array([0, 4, 2, 0])
b, c, d = spline_natural(x, y)

f3, ax3 = plt.subplots()
ax3.plot(sample, spline_eval(x, y, b, c, d, sample))
ax3.scatter(x, y, c="red")

# Apartado d)

pendientes = np.array([[0, -1], [1, 5], [-2, -5], [-5, 1]])

f4, ax4 = plt.subplots()
ax4.scatter(x, y, c="red")
sample = np.linspace(0, 4, 1001)

for k in range(4):
    b, c, d = spline_sujeto(x, y, pendientes[k, 0], pendientes[k, 1])
    ax4.plot(sample, spline_eval(x, y, b, c, d, sample))

plt.show()
