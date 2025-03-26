import numpy as np
import draw
import lib
from lib import Tracker

import quadrartic as config
# import rozenbrok as config

def draw_all(f, filename, coordinates):
    # draw.animate_2d_gradient_descent(f, coordinates, x_min_fixed, filename + '.gif', filename)
    draw.draw(f, "drawings/" + filename + '.png', coordinates)
    draw.draw_interactive(f, "drawings/" + filename + '.html', coordinates)

if __name__ == "__main__":

    tracker = Tracker()
    x_min_fixed = lib.gradient_descent_fixed(config.f, config.START_POINT.copy(), config.STEP, config.TOLERANCE, config.MAX_ITERATIONS, tracker)
    draw_all(config.f, "fixed", tracker.coordinates)
    print("Fixed: ", x_min_fixed)

    tracker = Tracker()
    x_min_fixed = lib.gradient_descent_decreasing(config.f, config.START_POINT.copy(), config.STEP, config.TOLERANCE, config.MAX_ITERATIONS // 100, tracker)
    draw_all(config.f, "decreasing", tracker.coordinates)
    print("Decreasing: ", x_min_fixed)

    tracker = Tracker()
    x_min_fixed = lib.gradient_descent_armijo(config.f, config.START_POINT.copy(), config.STEP, config.TOLERANCE, config.MAX_ITERATIONS, 0.7, tracker)
    draw_all(config.f, "armijo", tracker.coordinates)
    print("Armijo: ", x_min_fixed)

    tracker = Tracker()
    x_min_fixed = lib.gradient_descent_wolfe(config.f, config.START_POINT.copy(), config.STEP, config.TOLERANCE, config.MAX_ITERATIONS, 0.5, 0.5, 0.7, tracker)
    draw_all(config.f, "wolfe", tracker.coordinates)
    print("Wolfe: ", x_min_fixed)

    tracker = Tracker()
    x_min_fixed = lib.gradient_descent_golden(config.f, config.START_POINT.copy(), config.TOLERANCE, config.MAX_ITERATIONS, tracker)
    draw_all(config.f, "golden", tracker.coordinates)
    print("Golden: ", x_min_fixed)

    tracker = Tracker()
    x_min_fixed = lib.gradient_descent_dichotomy(config.f, config.START_POINT.copy(), config.TOLERANCE, config.MAX_ITERATIONS, tracker)
    draw_all(config.f, "dichotomy", tracker.coordinates)
    print("Dichotomy: ", x_min_fixed)
    