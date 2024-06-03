"""
Interpolacion de Hermite
https://es.wikipedia.org/wiki/Interpolaci%C3%B3n_polin%C3%B3mica_de_Hermite
"""
from math import factorial

import numpy as np

def coef_hermite(x: list, y: list):
    """
    Calcula los coeficientes en la forma de Newton interpolando a una funcion de la que se conocen algunos valores y algunas derivadas
    :param x: abscisas donde se interpola
    :param y: valores de la funcion en las abscisas, asi como, opcionalmente, de sus sucesivas derivadas. Debe ser una lista de listas, debe tener tantos elementos como x
    :return: coef = array de coeficientes, nodos_rep = array con los nodos repetidos
    """
    m = len(x)

    if len(y) != m:
        raise Exception('Las dimensiones de x e y no coinciden')

    l = [0]*m
    aux = [1] * (m+1)

    for i in range(m):
        l[i] = len(y[i])
        aux[i + 1] = aux[i] + l[i]
    n = aux[m] - 2
    nodos_rep = np.zeros(n + 1)
    coef = np.zeros(n + 1)

    for i in range(m):
        for j in range(aux[i], aux[i + 1]):
            nodos_rep[j - 1] = x[i]

    for k in range(n + 1):
        for i in range(m, -1, -1):
            for j in range(aux[i] - 2, max(aux[i - 1], k + 1) - 2, -1):
                if nodos_rep[j - k] == nodos_rep[j]:
                    coef[j] = y[i - 1][k] / factorial(k)
                else:
                    coef[j] = (coef[j] - coef[j - 1]) / (nodos_rep[j] - nodos_rep[j - k])

    return coef, nodos_rep


if __name__ == '__main__':
    from interpol_newton import polinomio_newton
    # p(x) = x^7 + 3x^5 -3x^2
    x = np.array([0, 1, -1])
    y = [[0, 0, -6, 0, 0, 360], [1], [-7]]
    c, nodos = coef_hermite(x, y)
    print(c)
    print(polinomio_newton(c, nodos))