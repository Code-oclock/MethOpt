import draw as dr
import lib
import config
import numpy as np

np.set_printoptions(formatter={'float': '{:.9f}'.format}, suppress=True)

def draw(f, filename, coordinates):
    dr.draw(f, config.RESULT_FOLDER + filename + '.html', coordinates)

if __name__ == "__main__":
    tracker = lib.Tracker()
    result = lib.gradient_descent_fixed(config.f, config.START_POINT.copy(), config.STEP_FIXED, config.TOLERANCE, config.MAX_ITERATIONS, tracker)
    draw(config.f, "fixed", tracker.coordinates)
    print("Fixed:      ", result, "Iteartions:", tracker.iterations)

    tracker = lib.Tracker()
    result = lib.gradient_descent_decreasing(config.f, config.START_POINT.copy(), config.STEP_DECREASING, config.TOLERANCE, config.MAX_ITERATIONS, tracker)
    draw(config.f, "decreasing", tracker.coordinates)
    print("Decreasing: ", result, "Iteartions:", tracker.iterations)

    tracker = lib.Tracker()
    result = lib.gradient_descent_armijo(config.f, config.START_POINT.copy(), config.STEP_ARMIJO, config.TOLERANCE, config.MAX_ITERATIONS, config.TAU, tracker)
    draw(config.f, "armijo", tracker.coordinates)
    print("Armijo:     ", result, "Iteartions:", tracker.iterations)

    tracker = lib.Tracker()
    result = lib.gradient_descent_wolfe(config.f, config.START_POINT.copy(), config.STEP_WOLFE, config.TOLERANCE, config.MAX_ITERATIONS, config.C1, config.C2, config.TAU, tracker)
    draw(config.f, "wolfe", tracker.coordinates)
    print("Wolfe:      ", result, "Iteartions:", tracker.iterations)

    tracker = lib.Tracker()
    result = lib.gradient_descent_golden(config.f, config.START_POINT.copy(), config.TOLERANCE, config.MAX_ITERATIONS, tracker)
    draw(config.f, "golden", tracker.coordinates)
    print("Golden:     ", result, "Iteartions:", tracker.iterations)

    tracker = lib.Tracker()
    result = lib.gradient_descent_dichotomy(config.f, config.START_POINT.copy(), config.TOLERANCE, config.MAX_ITERATIONS, tracker)
    draw(config.f, "dichotomy", tracker.coordinates)
    print("Dichotomy:  ", result, "Iteartions:", tracker.iterations)