import numpy as np
import matplotlib.pyplot as plt
import math

# Define the functions
def f1(x):
    return np.exp(x/2)

def f2(x):
    return np.sin(3 * x)

def f3(x):
    return np.cos(5 * x) ** 2

def f4(x, terms=10):
    return sum([(x ** (2 * n)) / math.factorial(2 * n) for n in range(terms)])

ranges = [(0, np.pi/2), (2, 10), (-3, 3)]
h_values = [0.01, 0.005, 0.001]
functions = [f1, f2, f3, f4]
function_names = ['exp(-x/2)', 'sin(3x)', 'cos^2(5x)', 'вот этот ряд))']


for i, func in enumerate(functions):
    for interval in ranges:
        for h in h_values:
            x = np.arange(interval[0], interval[1], h)
            y = np.vectorize(func)(x)

            x_dense = np.linspace(interval[0], interval[1], 1000)
            y_dense = np.vectorize(func)(x_dense)

            #plots
            plt.figure(figsize=(10, 6))
            plt.plot(x_dense, y_dense, label='Исходн. функ.', color='blue')
            plt.plot(x, y, 'o', label=f'Расчёт с шагом h = {h}', markersize=2, color='red')
            plt.title(f'Функция: {function_names[i]} на отрезке {interval}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.grid(True)
            plt.show()