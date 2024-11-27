import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import symbols, Function, dsolve, Eq, exp, sin, lambdify, simplify

m = 1.0
k = 1.0
v0 = 1.0
x0 = 0.0
t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 400)

h_under = 1.0
h_over = 4.0

forces = [
    {'name': 'f(t) = 0', 'f_expr': 0, 'f_func': lambda t: 0},
    {'name': 'f(t) = t - 1', 'f_expr': symbols('t') - 1, 'f_func': lambda t: t - 1},
    {'name': 'f(t) = e^{-t}', 'f_expr': exp(-symbols('t')), 'f_func': lambda t: np.exp(-t)},
    {'name': 'f(t) = sin(t)', 'f_expr': sin(symbols('t')), 'f_func': lambda t: np.sin(t)}
]


def analytical_solution(h, f_expr):
    t = symbols('t', real=True)
    x = Function('x')(t)
    f = f_expr
    eq = Eq(m * x.diff(t, t) + h * x.diff(t) + k * x, f)
    ics = {x.subs(t, 0): x0, x.diff(t).subs(t, 0): v0}
    sol = dsolve(eq, x, ics=ics)
    x_t = simplify(sol.rhs)
    x_func = lambdify(t, x_t, modules=['numpy'])
    return x_func


def numerical_solution(h, f_func):
    def ode_system(t, y):
        x1, x2 = y
        dx1dt = x2
        dx2dt = (f_func(t) - h * x2 - k * x1) / m
        return [dx1dt, dx2dt]

    y0 = [x0, v0]
    sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval)
    return sol.t, sol.y[0]


def plot_results(t_eval, x_analytical_under, x_num_under, x_analytical_over, x_num_over, force_name):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t_eval, x_analytical_under(t_eval), label='Analytical')
    plt.plot(t_eval, x_num_under, '--', label='Numerical')
    plt.title(f'h^2 < 4km Case\n{force_name}')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t_eval, x_analytical_over(t_eval), label='Analytical')
    plt.plot(t_eval, x_num_over, '--', label='Numerical')
    plt.title(f'h^2 > 4km Case\n{force_name}')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.legend()

    plt.tight_layout()
    plt.show()

for force in forces:
    force_name = force['name']
    f_expr = force['f_expr']
    f_func = force['f_func']

    x_analytical_under = analytical_solution(h_under, f_expr)
    _, x_num_under = numerical_solution(h_under, f_func)

    x_analytical_over = analytical_solution(h_over, f_expr)
    _, x_num_over = numerical_solution(h_over, f_func)

    plot_results(t_eval, x_analytical_under, x_num_under, x_analytical_over, x_num_over, force_name)