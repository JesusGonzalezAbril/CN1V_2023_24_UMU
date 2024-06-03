"""
Los "nodos de Chebyshev" son los ceros de los polinomios de Chebyshev. Estos puntos están en el intervalo [-1,1], pero
se pueden trasladar a un intervalo arbitrario [a,b]
https://es.wikipedia.org/wiki/Polinomios_de_Chebyshov
"""
import numpy as np


def cheby_nodes(a, b, n):
    """
    Devuelve los ceros del polinomio de Chebyshev de grado n en el intervalo [a,b]
    :param a: extremo inferior del intervalo
    :param b: extremo superior del intervalo
    :param n: grado del polinomio de Chebyshev
    :return: np.array con los nodos
    """
    indic = (2 * np.arange(n, 0, -1) - 1) / (2 * n) * np.pi
    return (1 + np.cos(indic)) * (b - a) * 0.5 + a


if __name__ == '__main__':
    print(cheby_nodes(-1, 1, 5))
