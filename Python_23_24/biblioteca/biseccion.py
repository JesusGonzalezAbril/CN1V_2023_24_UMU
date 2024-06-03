"""
Método de bisección - https://es.wikipedia.org/wiki/M%C3%A9todo_de_bisecci%C3%B3n
"""
import numpy as np


def biseccion(f, a: float, b: float, tol: float, max_iter: int):
    """
    Busca un cero de una funcion continua en un intervalo [a,b], la funcion debe verificar f(a)f(b)<0
    :param f: funcion
    :param a: extremo inferior del intervalo
    :param b: extremo superior del intervalo
    :param tol: tolerancia
    :param max_iter: maximo de iteraciones permitidas
    :return: c = aproximacion del cero, k = numero de iteraciones
    """
    aa, bb, fa, fb = a, b, f(a), f(b)
    if fa * fb > 0:
        raise Exception('No hay cambio de signo')

    for k in range(max_iter):
        h = 0.5 * (bb - aa)
        c = aa + h
        fc = f(c)
        if abs(h) < tol and abs(fc) < tol:
            return c, k
        if fc * fa > 0:
            aa = c
            fa = fc
        else:
            bb = c
            fb = fc
    raise Exception('Numero maximo de iteraciones alcanzado')


if __name__ == '__main__':
    def f(x): return x ** 3 - 2007


    print('f(x) = x^3 - 2007')
    print('Aproximacion: %.16f | Iteraciones: %d' % biseccion(f, 12, 13, 10 ** -1, 100))
    print('Aproximacion: %.16f | Iteraciones: %d' % biseccion(f, 12, 13, 10 ** -12, 1000))
    print(biseccion(lambda x: np.cos(10 * x), 0, 1, 10 ** -5, 100))
