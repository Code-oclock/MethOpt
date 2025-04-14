import draw as dr
import lib
import config
import numpy as np

np.set_printoptions(formatter={'float': '{:.9f}'.format}, suppress=True)

def draw(f, filename, coordinates):
    dr.draw(f, config.RESULT_FOLDER + filename + '.html', coordinates)

def our_methods():
    tracker = lib.Tracker()
    result = lib.newton_method(config.f, config.START_POINT.copy(), config.TOLERANCE, config.MAX_ITERATIONS, tracker)
    draw(config.f, "Newton Method", tracker.coordinates)
    print("Newton Method:  ", result, "Iteartions:", tracker.iterations)


if __name__ == "__main__":
    our_methods()
