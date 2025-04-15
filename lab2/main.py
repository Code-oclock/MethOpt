import draw as dr
import lib
import config
import numpy as np
import my_scipy

np.set_printoptions(formatter={'float': '{:.9f}'.format}, suppress=True)

def draw(f, filename, coordinates):
    dr.draw(f, config.RESULT_FOLDER + filename + '.html', coordinates)

def our_methods():
    # tracker = lib.Tracker()
    # result = lib.newton_method_with_armijo(config.f, config.START_POINT.copy(), config.TOLERANCE, config.MAX_ITERATIONS, tracker)
    # draw(config.f, "Newton Method", tracker.coordinates)
    # print("Newton Method with Armijo:  ", result, "Iteartions:", tracker.iterations)

    # tracker = lib.Tracker()
    # result = lib.newton_method_with_wolfe(config.f, config.START_POINT.copy(), config.TOLERANCE, config.MAX_ITERATIONS, tracker)
    # draw(config.f, "Newton Method", tracker.coordinates)
    # print("Newton Method with Wolfe:  ", result, "Iteartions:", tracker.iterations)

    # tracker = lib.Tracker()
    # result = lib.newton_method_with_golden(config.f, config.START_POINT.copy(), config.TOLERANCE, config.MAX_ITERATIONS, tracker)
    # draw(config.f, "Newton Method", tracker.coordinates)
    # print("Newton Method with Golden:  ", result, "Iteartions:", tracker.iterations)

    tracker = lib.Tracker()
    result = lib.bfgs_section_search(config.f, config.START_POINT.copy(), config.TOLERANCE, config.MAX_ITERATIONS, tracker)
    draw(config.f, "Newton Method", tracker.coordinates)
    print("Newton Method with Golden:  ", result, "Iteartions:", tracker.iterations)

def our_scipy():
    result, iterations = my_scipy.minimize_newton_CG(config.f, config.START_POINT.copy(), config.TOLERANCE)
    print("Newton_CG:       ", result, "Iteartions:", iterations)

    result, iterations = my_scipy.minimize_BFGS(config.f, config.START_POINT.copy(), config.TOLERANCE)
    print("BFGS:            ", result, "Iteartions:", iterations)

    result, iterations = my_scipy.minimize_LBFGS(config.f, config.START_POINT.copy(), config.TOLERANCE)
    print("L-BFGS:            ", result, "Iteartions:", iterations)

if __name__ == "__main__":
    our_methods()
    # our_scipy()
