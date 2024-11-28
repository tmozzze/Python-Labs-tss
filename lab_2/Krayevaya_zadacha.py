import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

def ode_system(x, y):
    y1, y2 = y
    dy1dx = y2
    dy2dx = x**2 * y2 + (2 / x**2) * y1 + 1 + (4 / x**2)
    return np.vstack((dy1dx, dy2dx))

def boundary_conditions(ya, yb):
    bc1 = 2 * ya[0] - ya[1] - 6
    bc2 = yb[0] + 3 * yb[1] + 1
    return np.array([bc1, bc2])

x = np.linspace(0.5, 1, 100)

y_init = np.zeros((2, x.size))

solution = solve_bvp(ode_system, boundary_conditions, x, y_init)

if solution.status != 0:
    print("Numerical solution could not be obtained")
else:
    print("The numerical solution was obtained successfully.")

x_sol = solution.x
y_sol = solution.y[0]
y_prime_sol = solution.y[1]
y_double_prime_sol = ode_system(x_sol, solution.y)[1]

plt.figure(figsize=(8, 6))
plt.plot(x_sol, y_sol, label='y(x)')
plt.title('Graph of function y(x)')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(x_sol, y_prime_sol, label="y'(x)", color='orange')
plt.title("Graph of the first derivative y'(x)")
plt.xlabel('x')
plt.ylabel("y'(x)")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(x_sol, y_double_prime_sol, label="y''(x)", color='green')
plt.title("Graph of the second derivative y''(x)")
plt.xlabel('x')
plt.ylabel("y''(x)")
plt.grid(True)
plt.legend()
plt.show()

x_table = np.linspace(0.5, 1, 10)
y_table = solution.sol(x_table)[0]
y_prime_table = solution.sol(x_table)[1]
y_double_prime_table = ode_system(x_table, solution.sol(x_table))[1]

print("Values of y(x) and its derivatives at selected points:")
print(f"{'x':>10} {'y(x)':>15} {'y\'(x)':>15} {'y\'\'(x)':>15}")
for xi, yi, ypi, ydpi in zip(x_table, y_table, y_prime_table, y_double_prime_table):
    print(f"{xi:>10.3f} {yi:>15.6f} {ypi:>15.6f} {ydpi:>15.6f}")
