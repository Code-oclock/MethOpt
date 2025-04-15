from functools import partial
from scipy.optimize import minimize, minimize_scalar, linesearch
import numpy as np
import lib

np.set_printoptions(formatter={'float': '{:.9f}'.format}, suppress=True)

def minimize_BFGS(f, point: np.array, tolerancy: float):
    result_quadratic = minimize(f, 
                    x0=point, 
                    method='BFGS', 
                    tol=tolerancy)
    if result_quadratic.success:
        return result_quadratic.x, result_quadratic.nit
    raise result_quadratic.message

def minimize_LBFGS(f, point: np.array, tolerancy: float):
    result_quadratic = minimize(f, 
                    x0=point, 
                    method='L-BFGS-B', 
                    tol=tolerancy)
    if result_quadratic.success:
        return result_quadratic.x, result_quadratic.nit
    raise result_quadratic.message

def minimize_linesearch(f, point: np.array, tolerance: float, max_iterations: int, tracker: lib.Tracker):
    tracker.track(point)
    for _ in range(max_iterations):
        current_gradient = lib.gradient(f, point)
        if np.linalg.norm(current_gradient) < tolerance:
            break
        pk = -current_gradient
        alpha, fc, gc, new_fval, old_fval, new_slope = linesearch.line_search(
            f,
            lambda x: lib.gradient(f, x),
            point,
            pk,
            current_gradient,
            f(point)
        )
        point = point + alpha * pk
        tracker.track(point)
    return np.array(point)


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
    return np.array(point)

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
    return np.array(point)


def minimize_newton_CG(f, point: np.array, tolerance: float):
    # Используем partial для фиксирования аргумента point в функции градиента и гессиана
    grad_f = partial(lib.gradient, f)  # фиксируем f, оставляем point
    hess_f = partial(lib.hessian, f)   # аналогично для гессиана
    result_newton = minimize(
        f, x0=point, method='Newton-CG', jac=grad_f, hess=hess_f, tol=tolerance)
    if result_newton.success:
        return result_newton.x, result_newton.nit


