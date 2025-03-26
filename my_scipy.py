from scipy.optimize import minimize, minimize_scalar
import numpy as np
import lib

def minimize_BFGS(f, point: np.array, tolerancy: float):
    result_quadratic = minimize(f, 
                    x0=point, 
                    method='BFGS', 
                    tol=tolerancy)
    if result_quadratic.success:
        return result_quadratic.x, result_quadratic.nit
    raise Exception('Минимум не найден')


def minimize_golden(f, point: np.array, tolerance: float, max_iterations: int, tracker: lib.Tracker):
    tracker.track(point)
    for _ in range(max_iterations):
        current_gradient = lib.gradient(f, point)
        if np.linalg.norm(current_gradient) < tolerance:
            break
        def g(alpha): return f(point - alpha * current_gradient)
        result = minimize_scalar(g, method='golden', tol=tolerance)
        step = result.x
        point -= step * current_gradient
        tracker.track(point)
    return np.array(point), tracker.iterations

def minimize_dichotomy(f, point: np.array, tolerance: float, max_iterations: int, tracker: lib.Tracker):
    tracker.track(point)
    for _ in range(max_iterations):
        current_gradient = lib.gradient(f, point)
        if np.linalg.norm(current_gradient) < tolerance:
            break
        def g(alpha): return f(point - alpha * current_gradient)
        result = minimize_scalar(g, bounds=(0, 1), method='bounded')
        step = result.x
        point -= step * current_gradient

        tracker.track(point)
    return np.array(point), tracker.iterations
