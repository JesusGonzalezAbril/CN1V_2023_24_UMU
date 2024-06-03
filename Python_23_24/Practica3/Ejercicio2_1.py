import matplotlib.pyplot as plt
import numpy as np

from Python_23_24.biblioteca import interpol_newton as newton
from Python_23_24.biblioteca.cheby_nodes import cheby_nodes


def f(x: np.array):
    return np.divide(1, 1 + x ** 2)


N = [5, 10, 15]

sample = np.linspace(-5, 5, 1001)

valor_fun = f(sample)

fig1, ax = plt.subplots(2, 2, figsize=(10, 10), sharey=True)
ax[0, 0].set_ylim(0, 1.1)
ax[0, 0].plot(sample, f(sample))
ax[1, 0].plot(sample, f(sample))
ax[0, 0].set_title('Nodos equiespaciados')
ax[0, 1].set_title('Error nodos equiespaciados')
ax[1, 0].set_title('Nodos Chebyshev')
ax[1, 1].set_title('Error nodos Chebyshev')

for k in N:
    nodos = np.linspace(-5, 5, k + 1)
    coefs = newton.coef_newton(nodos, f(nodos))
    pol = newton.polyinterpolador_eval(coefs, nodos, sample)
    ax[0, 0].plot(sample, pol)

    error = abs(pol - valor_fun)
    ax[0, 1].plot(sample, error)
    error_maximo = max(error)
    print('Error absoluto aproximado (%2u nodos equiespaciados): %16e \n' % (k, error_maximo))

for k in N:
    nodos = cheby_nodes(-5, 5, k)
    coefs = newton.coef_newton(nodos, f(nodos))
    pol = newton.polyinterpolador_eval(coefs, nodos, sample)
    ax[1, 0].plot(sample, pol)

    error = abs(pol - valor_fun)
    ax[1, 1].plot(sample, error)
    error_maximo = max(error)
    print('Error absoluto aproximado (%2u nodos chebyshev): %16e \n' % (k, error_maximo))

plt.show()
