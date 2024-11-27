import numpy as np
import matplotlib.pyplot as plt

def euler_method(func, y0, t, h):
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i - 1] + h * func(t[i - 1], y[i - 1])
    return y

def f1(x, y):
    return 0.5 * y

def f2(x, y):
    return 2 * x + 3 * y

t = np.arange(0, 10, 0.025)

y1 = euler_method(f1, y0=1, t=t, h=0.025)
plt.figure(figsize=(10, 6))
plt.plot(t, y1, label="y' = 1/2 * y, y(0) = 1", color='blue')
plt.xlabel('x')
plt.ylabel('y(t)')
plt.title('Уравнение 1')
plt.grid(True)
plt.legend()
plt.show()

y2 = euler_method(f2, y0=-2, t=t, h=0.025)
plt.figure(figsize=(10, 6))
plt.plot(t, y2, label="y' = 2x + 3y, y(0) = -2", color='blue')
plt.xlabel('x')
plt.ylabel('y(t)')
plt.title('Уравнение 2')
plt.grid(True)
plt.legend()
plt.show()

def euler_system(func, y0, t, h):
    y = np.zeros((len(t), len(y0)))
    y[0, :] = y0
    for i in range(1, len(t)):
        y[i, :] = y[i - 1, :] + h * func(t[i - 1], y[i - 1, :])
    return y

def s1(t, y):
    x1, x2 = y
    return np.array([x2, -x1])

y0_3 = [1, 0]
solution1 = euler_system(s1, y0=y0_3, t=t, h=0.025)
plt.figure(figsize=(10, 6))
plt.plot(t, solution1[:, 0], label="x1(t)", color='blue')
plt.plot(t, solution1[:, 1], label="x2(t)", color='orange')
plt.xlabel('Время t')
plt.ylabel('x1, x2')
plt.title('Система 1')
plt.grid(True)
plt.legend()
plt.show()

def s2(t, y):
    x1, x2 = y
    return np.array([x2, 4 * x1])

y0_4 = [1, 1]
solution2 = euler_system(s2, y0=y0_4, t=t, h=0.025)
plt.figure(figsize=(10, 6))
plt.plot(t, solution2[:, 0], label="x1(t)", color='blue')
plt.plot(t, solution2[:, 1], label="x2(t)", color='orange')
plt.xlabel('Время t')
plt.ylabel('x1, x2')
plt.title('Система 2')
plt.grid(True)
plt.legend()
plt.show()