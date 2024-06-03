import numpy as np
import matplotlib.pyplot as plt


# Evaluamos el polinomio p(x) = (x-1)^5 de tres formas diferentes
def p1(x):
    return (x - 1) ** 5


def p2(x):
    return x ** 5 - 5 * (x ** 4) + 10 * (x ** 2) + 5 * x - 1


def p3(x):
    return -1 + x * (5 + x * (-10 + x * (10 + x * (-5 + x))))


# Precisión doble
x = np.linspace(0.998, 1.002, 1000, dtype='float64')

fig, ax = plt.subplots(3, sharex=True)

ax[0].plot(x, p1(x))
ax[1].plot(x, p2(x))
ax[2].plot(x, p3(x))
plt.show()

del (x, fig, ax)

# Precisión simple
x = np.linspace(0.998, 1.002, 1000, dtype='float32')

fig, ax = plt.subplots(3, sharex=True)

ax[0].plot(x, p1(x))
ax[1].plot(x, p2(x))
ax[2].plot(x, p3(x))

plt.show()
