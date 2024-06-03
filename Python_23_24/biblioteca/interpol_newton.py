"""
Interpolacion de Newton
https://es.wikipedia.org/wiki/Interpolaci%C3%B3n_polin%C3%B3mica_de_Newton
"""
import numpy as np
from numpy import polynomial

def coef_newton(x: np.ndarray, y: np.ndarray):
    """
    Devuelve los coeficientes en la forma de Newton del polinomio que interpola los puntos de abscisas x, ordenadas y
    :param x: abscisas de los puntos a interpolar
    :param y: ordenadas de los puntos a interpolar
    :return: array de coeficientes
    """
    if x.ndim > 1 or y.ndim > 1:
        raise Exception('Las dimensiones de x e y son incorrectas')

    n = np.size(x)
    if np.size(y) != n:
        raise Exception('Las dimensiones de x e y no coinciden')

    coef_newton = y.copy()

    for i in range(n):
        div = (x[i + 1:n] - x[0:n - 1 - i])
        coef_newton[i + 1:n] = np.divide((coef_newton[i + 1:n] -
                                          coef_newton[i:n - 1]), div)

    return coef_newton


def polinomio_newton(coef: np.ndarray, nodos: np.ndarray):
    """
    Devuelve el polinomio con coeficientes de newton coef que interpola los nodos dados
    :param coef: coeficientes de newton, tal y como los devuelve la funcion coef_newton
    :param nodos: array de nodos que se interpolan
    :return: objeto del tipo polynomial del paquete numpy
    """
    n = np.size(coef) - 1
    a = np.zeros(n + 1)
    for i in range(n, -1, - 1):
        for k in range(n - i, 0, - 1):
            a[k] = a[k - 1] - a[k] * nodos[i]
        a[0] = coef[i] - a[0] * nodos[i]

    np.flip(a)
    return polynomial.polynomial.Polynomial(a).trim()


def interpol_newton(x: np.ndarray, y: np.ndarray, t: np.ndarray):
    """
    Calcula el polinomio interpolador que interpola los puntos [x,y] y evalúa ese polinomio en los puntos del array t
    :param x: abscisas
    :param y: ordenadas
    :param t: puntos donde se evalua el polinomio
    :return: np.array con los valores del polinomio en los puntos
    """
    coef = coef_newton(x, y)
    return polyinterpolador_eval(coef, x, t)


def polyinterpolador_eval(coef, x, t):
    n = np.size(coef) - 1
    m = np.size(t)
    pt = np.ones(m) * coef[n]
    for i in range(1, n + 1):
        pt = pt * (t - x[n - i]) + coef[n - i]
    return pt


if __name__ == '__main__':
    x = np.array([1, 2, 3, 4, 5, 6])
    y = np.array([1, 2, 3, 4])
    z = coef_newton(x, x)
    print(z)
    z = polinomio_newton(z, x)
    print(z)
    print(interpol_newton(x, x, [1, 2, 3, 123]))
    """
    fig1, ax = plt.subplots()
    ax.plot(np.linspace(0, 2, 100), interpol_newton(x, x, np.linspace(0, 2, 100)))
    plt.show()"""
    # print(interpolNewton(x, y)) esto da error (como debe ser)
