import numpy as np
import draw
import lib
from lib import Tracker

# # точность нахождения минимума
# TOLERANCE = 1e-4
# # базовый щаг поиска
# STEP = 0.3
# # Количество итераций
# MAX_ITERATIONS = 10_000
# # Начальная точка для градиентного спуска
# START_POINT = np.array([-3., -4.])


# точность нахождения минимума
TOLERANCE = 1e-4
# базовый щаг поиска
STEP = 0.0001
# Количество итераций
MAX_ITERATIONS = 10_0000
# Начальная точка для градиентного спуска
START_POINT = np.array([-3., -4.])

# Fixed:  [0.6706809  0.44823252]
# Decreasing:  [-0.65937767 -3.35801013]
# Armijo:  [0.67055025 0.44805657]
# Wolfe:  [0.86112903 0.74093977]
# Golden:  [ 4.98105457 24.81334131]
# Dichotomy:  [1.00553843 1.01111477]


# Функция для которой ищем минимум (1)
# f(x, y) = x^2 + y^2
def f_quadratic_standart(point):
    x, y = point
    return x**2 + y**2

# Функция для которой ищем минимум (2) - Розенброк
# f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
def rozenbrok(point):
    x, y = point
    return (1 - x)**2 + 100 * (y - x**2) ** 2 


def draw_all(f, filename, coordinates):
    # draw.animate_2d_gradient_descent(f, coordinates, x_min_fixed, filename + '.gif', filename)
    draw.draw(f, filename + '.png', coordinates)
    draw.draw_interactive(f, filename + '.html', coordinates)

if __name__ == "__main__":

    tracker = Tracker()
    x_min_fixed = lib.gradient_descent_fixed(rozenbrok, START_POINT.copy(), STEP, TOLERANCE, MAX_ITERATIONS, tracker)
    draw_all(rozenbrok, "fixed", tracker.coordinates)
    print("Fixed: ", x_min_fixed)

    tracker = Tracker()
    x_min_fixed = lib.gradient_descent_decreasing(rozenbrok, START_POINT.copy(), STEP, TOLERANCE, MAX_ITERATIONS // 100, tracker)
    draw_all(rozenbrok, "decreasing", tracker.coordinates)
    print("Decreasing: ", x_min_fixed)

    tracker = Tracker()
    x_min_fixed = lib.gradient_descent_armijo(rozenbrok, START_POINT.copy(), STEP, TOLERANCE, MAX_ITERATIONS, 0.7, tracker)
    draw_all(rozenbrok, "armijo", tracker.coordinates)
    print("Armijo: ", x_min_fixed)

    tracker = Tracker()
    x_min_fixed = lib.gradient_descent_wolfe(rozenbrok, START_POINT.copy(), STEP, TOLERANCE, MAX_ITERATIONS, 0.5, 0.5, 0.7, tracker)
    draw_all(rozenbrok, "wolfe", tracker.coordinates)
    print("Wolfe: ", x_min_fixed)

    tracker = Tracker()
    x_min_fixed = lib.gradient_descent_golden(rozenbrok, START_POINT.copy(), TOLERANCE, MAX_ITERATIONS, tracker)
    draw_all(rozenbrok, "golden", tracker.coordinates)
    print("Golden: ", x_min_fixed)

    tracker = Tracker()
    x_min_fixed = lib.gradient_descent_dichotomy(rozenbrok, START_POINT.copy(), TOLERANCE, MAX_ITERATIONS, tracker)
    draw_all(rozenbrok, "dichotomy", tracker.coordinates)
    print("Dichotomy: ", x_min_fixed)
    