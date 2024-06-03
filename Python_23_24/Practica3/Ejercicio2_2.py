import matplotlib.pyplot as plt
import numpy as np

from Python_23_24.biblioteca import interpol_hermite as hermite
from Python_23_24.biblioteca import interpol_newton as newton


def f(x: np.array):
    return np.divide(1, 1 + x ** 2)


def df(x: np.array):
    return np.divide(-2 * x, (1 + x ** 2) ** 2)


def d2f(x: np.array):
    return np.divide(6 * x ** 2 - 2, (1 + x ** 2) ** 3)


N = [5, 10, 15]

sample = np.linspace(-5, 5, 1001)

valor_fun = f(sample)

fig1, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].set_ylim(0, 1.1)
ax[1, 0].set_ylim(0, 1.1)
ax[0, 0].plot(sample, f(sample))
ax[1, 0].plot(sample, f(sample))
ax[0, 0].set_title('Derivadas extremos')
ax[0, 1].set_title('Error')
ax[1, 0].set_title('Derivadas extremos e interior')
ax[1, 1].set_title('Error')

for k in N:
    nodos = np.linspace(-5, 5, k + 1)
    val = [[elem] for elem in list(f(nodos))]
    val[0] = [f(nodos[0]), df(nodos[0]), d2f(nodos[0])]
    val[-1] = [f(nodos[-1]), df(nodos[-1]), d2f(nodos[-1])]
    coefs, nodos_rep = hermite.coef_hermite(nodos, val)
    pol = newton.polyinterpolador_eval(coefs, nodos_rep, sample)
    ax[0, 0].plot(sample, pol)

    error = abs(pol - valor_fun)
    ax[0, 1].plot(sample, error)
    error_maximo = max(error)
    print('Error absoluto aproximado (%2u nodos): %16e \n' % (k, error_maximo))

for k in N:
    nodos = np.linspace(-5, 5, k + 1)
    extra = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
    nodos = np.concatenate((nodos, extra))
    val = [[elem] for elem in list(f(nodos))]
    val[0] = [f(nodos[0]), df(nodos[0])]
    val[-3] = [f(nodos[-3]), df(nodos[-3])]
    val[-2] = ([f(-1 / np.sqrt(3)), df(-1 / np.sqrt(3))])
    val[-1] = ([f(1 / np.sqrt(3)), df(1 / np.sqrt(3))])
    coefs, nodos_rep = hermite.coef_hermite(nodos, val)
    pol = newton.polyinterpolador_eval(coefs, nodos_rep, sample)
    ax[1, 0].plot(sample, pol)

    error = abs(pol - valor_fun)
    ax[1, 1].plot(sample, error)
    error_maximo = max(error)
    print('Error absoluto aproximado (%2u con más derivadas): %16e \n' % (k, error_maximo))

plt.show()
