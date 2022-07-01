import numpy as np
import matplotlib.pyplot as plt
import math

def func(x):
    a = 0.1 * x ** 1.0 / 3
    b = np.sin(x / math.pi)
    y = a + (b + 0.1 * x) * x ** 2 + x
    return y


def integral():
    n = 20000000
    x_min, x_max = 0, 2.0
    y_min, y_max = 0, 6.0

    x = np.random.uniform(x_min, x_max, size=(n, 1))
    y = np.random.uniform(y_min, y_max, size=(n, 1))
    yy = func(x)
    c = np.sum(yy > y)
    ratio = c / float(n)
    res = ratio * (2.0 * 6.0)
    print(res)

x = np.linspace(0, 2, 1000)
x_hat = np.linspace(-0.1, 2.2, 1000)
y_hat = np.linspace(6, 6, 1000)
y = func(x)
plt.plot(x, y)
plt.fill_between(x, y, color='red', alpha=0.5)
plt.plot(x_hat, y_hat)
x_hhat = np.linspace(2.0, 2.0, 1000)
y_hhat = np.linspace(-1.0, 7.0, 1000)
plt.plot(x_hhat, y_hhat)
x_low_hat = np.linspace(-0.1, 2.2, 1000)
y_low_hat = np.linspace(0, 0, 1000)
plt.plot(x_low_hat, y_low_hat)
x_left_hat = np.linspace(0, 0, 1000)
y_left_hat = np.linspace(-1.0, 7.0, 1000)
plt.plot(x_left_hat, y_left_hat)

# plt.show()

integral()
