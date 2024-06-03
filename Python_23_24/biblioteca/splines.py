import numpy as np
from matplotlib import pyplot as plt

"""
Diversas funciones relacionada con la interpolación por splines cúbicos
"""


def spline_eval(x, a, b, c, d, t):
    """
    Función para evaluar un spline de coeficientes a,b,c,d en los puntos t.
    El spline es de la forma a_i+b_i*(x-x_i)+c_i*(x-x_i)^2+d_i*(x-x_i)^3
    :param x: vector que contiene las abscisas de los puntos a interpolar (nodos) ordenados de forma creciente.
    :param a: coeficientes del spline correspondientes a la potencia 0
    :param b: coeficientes del spline correspondientes a la potencia 1
    :param c: coeficientes del spline correspondientes a la potencia 2
    :param d: coeficientes del spline correspondientes a la potencia 3
    :param t: puntos a evaluar
    :return: evaluación del spline en t

    """
    n = len(x) - 1
    eval = np.zeros(len(t))
    test = (t < x[1])

    if len(t[test]) != 0:
        aux = t[test] - x[0]
        eval[test] = a[0] + aux * (b[0] + aux * (c[0] + aux * d[0]))

    test = (t >= x[-2])

    if len(t[test]) != 0:
        aux = t[test] - x[-2]
        eval[test] = a[-2] + aux * (b[-2] + aux * (c[-2] + aux * d[-2]))

    for k in range(1, n):
        test = (t >= x[k]) & (t < x[k + 1])

        if len(t[test]) != 0:
            aux = t[test] - x[k]
            eval[test] = a[k] + aux * (b[k] + aux * (c[k] + aux * d[k]))

    return eval


def spline_natural(x, y):
    """
    Calcula los coeficientes de un spline cúbico natural para los puntos dados.
    El spline es de la forma a_i+b_i*(x-x_i)+c_i*(x-x_i)^2+d_i*(x-x_i)^3 donde los a_i coinciden con y

    :param x: array de abscisas de los puntos a interpolar, ordenados de forma creciente
    :param y: array de ordenadas de los puntos a interpolar
    :return: tres arrays que representan los coeficientes b, c y d del spline cúbico natural:
             - b: coeficientes del spline correspondientes a la potencia 1
             - c: coeficientes del spline correspondientes a la potencia 2
             - d: coeficientes del spline correspondientes a la potencia 3
    """
    n = len(x)
    if len(y) != n:
        raise ValueError('Dimensiones incompatibles')
    n -= 1

    b = np.zeros(n + 1)
    c = np.zeros(n + 1)
    d = np.zeros(n + 1)
    h = np.zeros(n)
    ds = np.zeros(n)
    l = np.zeros(n + 1)
    u = np.zeros(n)
    z = np.zeros(n)

    for i in range(n):
        h[i] = x[i + 1] - x[i]
        ds[i] = y[i + 1] - y[i]

    l[0] = 1
    for i in range(1, n):
        l[i] = 2 * (h[i] + h[i - 1]) - h[i - 1] * u[i - 1]
        u[i] = h[i] / l[i]
        z[i] = (3 * ds[i] / h[i] - 3 * ds[i - 1] / h[i - 1] - h[i - 1] * z[i - 1]) / l[i]

    l[n] = 1
    c[n] = 0

    for i in range(n - 1, -1, -1):
        c[i] = z[i] - u[i] * c[i + 1]
        b[i] = ds[i] / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    return b, c, d


def spline_sujeto(x, y, tan_ini, tan_fin):
    """
    Calcula los coeficientes de un spline cúbico sujeto para los puntos dados.
    El spline es de la forma a_i+b_i*(x-x_i)+c_i*(x-x_i)^2+d_i*(x-x_i)^3 donde los a_i coinciden con y

    :param x: array de abscisas de los puntos a interpolar, ordenados de forma creciente
    :param y: array de ordenadas de los puntos a interpolar
    :param tan_ini: pendiente del spline en el primer punto
    :param tan_fin: pendiente del spline en el último punto
    :return: tres arrays que representan los coeficientes b, c y d del spline cúbico natural:
             - b: coeficientes del spline correspondientes a la potencia 1
             - c: coeficientes del spline correspondientes a la potencia 2
             - d: coeficientes del spline correspondientes a la potencia 3
    """
    n = len(x)
    if len(y) != n:
        raise ValueError('Dimensiones incompatibles')
    n -= 1

    b = np.zeros(n + 1)
    c = np.zeros(n + 1)
    d = np.zeros(n + 1)
    h = np.zeros(n)
    ds = np.zeros(n)
    l = np.zeros(n + 1)
    u = np.zeros(n)
    z = np.zeros(n + 1)

    for i in range(n):
        h[i] = x[i + 1] - x[i]
        ds[i] = y[i + 1] - y[i]

    l[0] = 2 * h[0]  # cambio con respecto al spline natural
    u[0] = 0.5  # cambio con respecto al spline natural
    z[0] = (3 * ds[0] / h[0] - 3 * tan_ini) / l[0]  # cambio con respecto al spline natural

    for i in range(1, n):
        l[i] = 2 * (h[i] + h[i - 1]) - h[i - 1] * u[i - 1]
        u[i] = h[i] / l[i]
        z[i] = (3 * (ds[i] / h[i] - ds[i - 1] / h[i - 1]) - h[i - 1] * z[i - 1]) / l[i]

    l[n] = h[n - 1] * (2 - u[n - 1])  # cambio con respecto al spline natural
    z[n] = (3 * tan_fin - 3 * ds[n - 1] / h[n - 1] - h[n - 1] * z[n - 1]) / l[n]  # cambio con respecto al spline natural
    c[n] = z[n]  # cambio con respecto al spline natural

    for i in range(n - 1, -1, -1):
        c[i] = z[i] - u[i] * c[i + 1]
        b[i] = ds[i] / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    return b, c, d


if __name__ == '__main__':
    x = [1, 2, 3, 4]
    y = [1, 8, 27, 64]
    b, c, d = spline_natural(x, y)

    sample = np.linspace(0, 5, 1000)

    plt.plot(sample, spline_eval(x, y, b, c, d, sample))

    plt.scatter(x, y, c="red")

    plt.show()
