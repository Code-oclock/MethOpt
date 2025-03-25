import numpy as np
import draw
import lib
from lib import Tracker

# точность нахождения минимума
TOLERANCE = 1e-4
# базовый щаг поиска
STEP = 0.3
# Количество итераций
MAX_ITERATIONS = 10_000
# Начальная точка для градиентного спуска
START_POINT = np.array([-3., -4.])

def draw_all(filename, coordinates):
    draw.animate_2d_gradient_descent(coordinates, x_min_fixed, filename + '.gif', filename)
    draw.draw(filename + '.png', coordinates)
    draw.draw_interactive(filename + '.html', coordinates)

if __name__ == "__main__":

    tracker = Tracker()
    x_min_fixed = lib.gradient_descent_fixed(lib.f, START_POINT.copy(), STEP, TOLERANCE, MAX_ITERATIONS, tracker)
    print(x_min_fixed)
    