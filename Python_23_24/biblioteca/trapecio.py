"""
Regla del trapecio
https://es.wikipedia.org/wiki/Regla_del_trapecio
"""
import numpy as np


def trapecio(f, a: float, b: float, num_divisiones: int):
    """
    Aproxima la integral de f entre a y b mediante la regla compuesta del trapecio con n divisiones
    :param f: funcion cuya integral se quiere aproximar
    :param a: extremo inferior del intervalo
    :param b: extremo superior del intervalo
    :param num_divisiones: número de divisiones del intervalo para la regla compuesta
    :return: aproximacion numerica de la integral
    """
    x = np.linspace(a, b, num_divisiones + 1)
    g = np.vectorize(f)
    im = g(x)
    return (b - a) * (im[0] + im[-1] + 2 * sum(im[1:-1])) / (2 * num_divisiones)


if __name__ == '__main__':
    from math import sin, pi

    def f(x): return sin(x)

    aprox = trapecio(f, 0, pi / 4, 100)

    print(aprox)
