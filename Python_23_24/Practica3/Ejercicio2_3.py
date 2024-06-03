import matplotlib.pyplot as plt
import numpy as np

from Python_23_24.biblioteca import interpol_newton as newton

def f(a: float, x: np.array):
    return np.divide(1, 1 + a * x ** 2)


N = [1, 5, 10, 15]

sample = np.linspace(-5, 5, 1001)


fig1, ax = plt.subplots(4, 1, figsize=(5, 10))

for k in range(len(N)):
    ax[k].set_ylim(-0.2,1.1)
    valor_fun = f(N[k], sample)
    ax[k].plot(sample, valor_fun)
    nodos = np.linspace(-5, 5, 40)
    fa = f(N[k], nodos)
    ax[k].plot(sample, newton.interpol_newton(nodos, fa, sample))

plt.show()

"""
Observamos que conforme crece el valor de a, el intervalo donde los polinomios convergen se hace cada vez más pequeño
"""
