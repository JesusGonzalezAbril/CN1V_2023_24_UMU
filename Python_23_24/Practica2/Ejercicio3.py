import numpy as np
import matplotlib.pyplot as plt


def f(x, a):
    return np.exp(-x**2) * np.exp(-a**2)


def auxiliar(func, par, s):
    def f_a(x):
        return func(x, par)
    y = f_a(s)
    print(f'max={max(y)}')
    plt.cla()
    plt.plot(s, y)


x = np.linspace(-5, 5, 2000)

fig, ax = plt.subplots()

plt.ion()  # Activo el modo interactivo para que las figuras se actualizen
plt.show()

for a in np.arange(-2, 2, 0.1):
    auxiliar(f, a, x)
    plt.ylim(-0.1, 1.1)
    plt.pause(0.005)
