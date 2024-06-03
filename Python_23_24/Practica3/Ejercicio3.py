import matplotlib.pyplot as plt
import numpy as np

from Python_23_24.biblioteca import interpol_newton as newton

x = np.array([0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])
y = np.array([0, 0.5, np.sqrt(2) / 2, np.sqrt(3) / 2, 1])

coef = newton.coef_newton(y, x)

aprox = newton.polyinterpolador_eval(coef, y, 0.75)

real = np.arcsin(0.75)

print("Real: %16e, Aprox: %16e, Error: %16e, Error rel: %16e" % (
    real, aprox, abs(real - aprox), abs(real - aprox) / abs(real)))

a = x[:-1]
b = y[:-1]

coef2 = newton.coef_newton(b, a)

aprox2 = newton.polyinterpolador_eval(coef2, b, 0.75)

real = np.arcsin(0.75)
print('\nQuitando el ultimo punto')
print("Real: %16e, Aprox: %16e, Error: %16e, Error rel: %16e" % (
    real, aprox2, abs(real - aprox2), abs(real - aprox2) / abs(real)))

fig, ax = plt.subplots(1, 2, sharey=True,figsize=(10,6))

sample = np.linspace(0, 1, 1001)
ax[0].plot(sample, np.arcsin(sample), label='Arcoseno')
ax[1].plot(sample, np.arcsin(sample), label='Arcoseno')
ax[0].plot(sample, newton.polyinterpolador_eval(coef, y, sample), label='Interpolador')
ax[1].plot(sample, newton.polyinterpolador_eval(coef2, b, sample),label='Interpolador sin el ultimo punto')

ax[0].legend()
ax[1].legend()
plt.show()
