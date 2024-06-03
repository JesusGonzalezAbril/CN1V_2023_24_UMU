import numpy as np
from matplotlib import pyplot as plt

# Defino f
def f(x):
    return np.pi + np.sqrt((1 + x ** 2))

# Creo mis abscisas
x = np.linspace(-3, 3, 179)

# Creo la figura sobre la que voy a dibujar
fig1, ax = plt.subplots()

# Dibujo f con colorines
ax.plot(x, f(x), color='r', linewidth='2', label='f(x)')

# Defino g y la dibujo
def g(x):
    return np.sin(x)

ax.plot(x, g(x), color='c', linewidth='0.5', linestyle='--', label='sin(x)')

# Defino h = f + g y la dibujo
def h(x):
    return f(x) + g(x)

ax.plot(x, h(x), color='g', linestyle='-.', label='f(x) + sen(x)')

ax.set_title('Mi primer gráfico')  # Cambio el titulo
ax.legend()  # Creo la leyenda
plt.show()  # Muestro el gráfico
