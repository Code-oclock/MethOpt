import draw as dr
import lib
import config
import numpy as np
import my_scipy
from opt_hyper import find_best_params
import warnings
warnings.filterwarnings("ignore") 
np.set_printoptions(formatter={'float': '{:.9f}'.format}, suppress=True)


def draw(f, filename, coordinates):
    dr.draw(f, config.RESULT_FOLDER + filename + '.html', coordinates)

def our_methods():
    tracker = lib.Tracker()
    result = lib.newton_method_with_armijo(config.f, config.START_POINT.copy(), config.STEP_ARMIJO, config.TOLERANCE, config.MAX_ITERATIONS, config.C1, config.TAU, tracker)
    draw(config.f, "Newton Method with Armijo", tracker.coordinates)
    print("Newton Method with Armijo:  ", result)
    tracker.print_stats()
    print()

    tracker = lib.Tracker()
    result = lib.newton_method_with_wolfe(config.f, config.START_POINT.copy(), config.STEP_WOLFE, config.TOLERANCE, config.MAX_ITERATIONS, config.C1, config.C2, config.TAU, tracker)
    draw(config.f, "Newton Method with Wolfe", tracker.coordinates)
    print("Newton Method with Wolfe:  ", result)
    tracker.print_stats()
    print()

    tracker = lib.Tracker()
    result = lib.newton_method_with_golden(config.f, config.START_POINT.copy(), config.TOLERANCE, config.MAX_ITERATIONS, tracker)
    draw(config.f, "Newton Method with Golden", tracker.coordinates)
    print("Newton Method with Golden:  ", result)
    tracker.print_stats()
    print()

    tracker = lib.Tracker()
    result = lib.bfgs_section_search(config.f, config.START_POINT.copy(), config.TOLERANCE, config.MAX_ITERATIONS, tracker)
    draw(config.f, "BFGS Method realization", tracker.coordinates)
    print("BFGS Method realization:  ", result)
    tracker.print_stats()
    print()

def our_scipy():
    tracker = lib.Tracker()
    result, tracker.iterations = my_scipy.minimize_newton_CG(lib.to_tracked_f(config.f, tracker), config.START_POINT.copy(), config.TOLERANCE)
    print("Newton_CG:       ", result)
    tracker.print_stats()
    print()

    tracker = lib.Tracker()
    result, tracker.iterations = my_scipy.minimize_BFGS(lib.to_tracked_f(config.f, tracker), config.START_POINT.copy(), config.TOLERANCE)
    print("BFGS:            ", result)
    tracker.print_stats()
    print()

    tracker = lib.Tracker()
    result, tracker.iterations = my_scipy.minimize_LBFGS(lib.to_tracked_f(config.f, tracker), config.START_POINT.copy(), config.TOLERANCE)
    print("L-BFGS:            ", result)
    tracker.print_stats()
    print()


if __name__ == "__main__":
    # our_methods()
    our_scipy()
    find_best_params()
