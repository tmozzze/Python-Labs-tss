import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return np.exp(-x ** 2 / 2)

def df1(x):
    return -x * np.exp((-x ** 2) / 2)

def f2(x):
    return np.sin((3 * x ** 4) / 5) ** 3

def df2(x):
    return 12 * x ** 3 * np.sin((3 * x ** 4) / 5) ** 2 * np.cos((3 * x ** 4) / 5) / 5

def f3(x):
    return np.cos(x / (x + 1)) ** 2

def df3(x):
    return -2 * np.cos(x / (x + 1)) * np.sin(x / (x + 1)) * (1 / (x + 1) ** 2)

def f4(x):
    return np.log(x + np.sqrt(4 + x**2))

def df4(x):
    return (1 + x / np.sqrt(4 + x**2)) / (x + np.sqrt(4 + x**2))

def f5(x):
    return (x * np.arctan(2 * x)) / (x ** 2 + 4)

def df5(x):
    numerator = np.arctan(2 * x) + (2 * x) / (1 + (2 * x) ** 2)
    denominator = x ** 2 + 4
    return (numerator * denominator - x * np.arctan(2 * x) * 2 * x) / (denominator ** 2)


ranges = [(0, 1), (2, 15), (-5, 5)]
h_values = [0.01, 0.005]
functions_names = ['exp(-x^2 / 2)', 'sin^3((3x^4) / 5)', 'cos^2(x / (x + 1))', 'ln(x + sqrt(4 + x^2))',
                  'x * arctan(2x) / (x^2 + 4)']

functions = [f1, f2, f3, f4, f5]
d_functions = [df1, df2, df3, df4, df5]

for i, (func, d_func) in enumerate(zip(functions, d_functions)):
    for range in ranges:
        for h in h_values:
            x = np.arange(range[0], range[1], h)
            y = func(x)

            y_prime_numeric = (y[1:] - y[:-1]) / h
            x_prime = x[:-1]

            y_prime_analytic = d_func(x)

            #plots
            plt.figure(figsize=(12, 6))
            plt.plot(x_prime, y_prime_numeric, label=f'Численная производная (h = {h})', linestyle='--')
            plt.plot(x, y_prime_analytic, label='Аналитическая производная', linestyle='-')
            plt.title(f'Функция: {functions_names[i]} на отрезке {range}')
            plt.xlabel('x')
            plt.ylabel("y'")
            plt.legend()
            plt.grid(True)
            plt.show()