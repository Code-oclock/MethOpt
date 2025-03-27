import numpy as np
import draw
import lib
from lib import Tracker
import my_scipy as scp

import quadrartic_demonstration as config
# import rozenbrok_demonstration as config

def draw_all(f, filename, coordinates, x_min_fixed):
    draw.animate_2d_gradient_descent(f, coordinates, x_min_fixed, 'drawings/' + filename + '.gif', filename)
    draw.draw(f, "drawings/" + filename + '.png', coordinates)
    draw.draw_interactive(f, "drawings/" + filename + '.html', coordinates)

def scipy_optimize(f):
    x_min, iterations = scp.minimize_BFGS(f, config.START_POINT.copy(), config.TOLERANCE)
    print("BFGS:", x_min, "Iteartions:", iterations)

    x_min, iterations = scp.minimize_golden(f, config.START_POINT.copy(), config.TOLERANCE, config.MAX_ITERATIONS, Tracker())
    print("Golden:", x_min, "Iteartions:", iterations)
    
    x_min, iterations = scp.minimize_dichotomy(f, config.START_POINT.copy(), config.TOLERANCE, config.MAX_ITERATIONS, Tracker())
    print("Dichotomy:", x_min, "Iteartions:", iterations)
    

if __name__ == "__main__":
    tracker = Tracker()
    x_min_fixed = lib.gradient_descent_fixed(config.f, config.START_POINT.copy(), config.STEP, config.TOLERANCE, config.MAX_ITERATIONS, tracker)
    draw_all(config.f, "fixed", tracker.coordinates, x_min_fixed)
    print("Fixed: ", x_min_fixed, "Iteartions:", tracker.iterations)

    tracker = Tracker()
    x_min_fixed = lib.gradient_descent_decreasing(config.f, config.START_POINT.copy(), config.STEP, config.TOLERANCE, config.MAX_ITERATIONS // 100, tracker)
    draw_all(config.f, "decreasing", tracker.coordinates, x_min_fixed)
    print("Decreasing: ", x_min_fixed, "Iteartions:", tracker.iterations)

    tracker = Tracker()
    x_min_fixed = lib.gradient_descent_armijo(config.f, config.START_POINT.copy(), config.STEP, config.TOLERANCE, config.MAX_ITERATIONS, 0.7, tracker)
    draw_all(config.f, "armijo", tracker.coordinates, x_min_fixed)
    print("Armijo: ", x_min_fixed, "Iteartions:", tracker.iterations)

    tracker = Tracker()
    x_min_fixed = lib.gradient_descent_wolfe(config.f, config.START_POINT.copy(), config.STEP, config.TOLERANCE, config.MAX_ITERATIONS, 0.5, 0.5, 0.7, tracker)
    draw_all(config.f, "wolfe", tracker.coordinates, x_min_fixed)
    print("Wolfe: ", x_min_fixed, "Iteartions:", tracker.iterations)

    tracker = Tracker()
    x_min_fixed = lib.gradient_descent_golden(config.f, config.START_POINT.copy(), config.TOLERANCE, config.MAX_ITERATIONS, tracker)
    draw_all(config.f, "golden", tracker.coordinates, x_min_fixed)
    print("Golden: ", x_min_fixed, "Iteartions:", tracker.iterations)

    tracker = Tracker()
    x_min_fixed = lib.gradient_descent_dichotomy(config.f, config.START_POINT.copy(), config.TOLERANCE, config.MAX_ITERATIONS, tracker)
    draw_all(config.f, "dichotomy", tracker.coordinates, x_min_fixed)
    print("Dichotomy: ", x_min_fixed, "Iteartions:", tracker.iterations)

    print("-----------------------SCIPY-----------------------")

    scipy_optimize(config.f)
    