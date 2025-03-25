from typing import List
import numpy as np
from autograd import grad

# Функция для которой ищем минимум
# f(x, y) = x^2 + y^2
def f_quadratic_standart(point):
    x, y = point
    return x**2 + y**2

# Функция для которой ищем минимум (2) - Розенброк
# f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
def rozenbrok(point):
    x, y = point
    return (1 - x)**2 + 100 * (y - x**2) ** 2 


# Градиент функции f в точке
def gradient(f, point: np.array):
    x, y = point
    grad_f = grad(f)
    return grad_f(point)


class Tracker:
    def __init__(self) -> None:
        self.__path = [[], []]
        self.__iterations = 0

    def track(self, point: np.array) -> None:
        self.__path[0].append(point[0])
        self.__path[1].append(point[1])

        self.__iterations += 1

    @property
    def coordinates(self) -> np.array:
        return self.__path
    
#----------------------
#       ЧАСТЬ 1       |
#----------------------

# Градиентный спуск с константным шагом
def gradient_descent_fixed(f: function, point: np.array, h: float, tolerance: float, max_iterations: int, tracker: Tracker) -> np.array:
    tracker.track(point)

    for _ in range(max_iterations):
        current_gradient = gradient(f, point)
        # Если норма градиента меньше заданной точности, то завершаем поиск
        if np.linalg.norm(current_gradient) < tolerance:
            break
        # Обновляем координаты x_k = x_k-1 - h * grad(f)
        point -= h * current_gradient
        
        tracker.track(point)
    return np.array(point)

def gradient_descent_decreasing(f: function, point: np.array, h: float, tolerance: float, max_iterations: int, tracker: Tracker) -> np.array:
    tracker.track(point)

    for i in range(max_iterations):
        current_gradient = gradient(f, point)
        # Если норма градиента меньше заданной точности, то завершаем поиск
        if np.linalg.norm(current_gradient) < tolerance:
            break
        # Обновляем координаты x_k = x_k-1 - (h / k) * grad(f)
        point -= h / (1 + i) * current_gradient
        
        tracker.track(point)
    return np.array(point)

def backtracking_armijo(f, grad_f, point, alpha=1.0, c=np.random.uniform(0, 1), tau=0.7):
    grad = grad_f(point)
    # Условие Армихо: f(x - alpha*grad) <= f(x) - c*alpha*(||grad||^2)
    while f(point - alpha * grad) > f(point) - c * alpha * np.dot(grad, grad):
        alpha *= tau  # уменьшаем шаг
    return alpha

def gradient_descent_armijo(f, grad_f, point, h, tol=1e-6, max_iter=100_000, c=1e-4, tau=0.1):
    path = [[point[0]], [point[1]]]
    for i in range(max_iter):
        grad = grad_f(point)
        if np.linalg.norm(grad) < tol:
            break
        alpha = backtracking_armijo(f, grad_f, point, h, np.random.uniform(0, 1), tau)
        point -= alpha * grad
        path[0].append(point[0])
        path[1].append(point[1])
    print("Кол-во итераций:", i)
    return np.array(point), np.array(path)

# 1.4 Градиентный спуск с подбором шага по условиям Вульфа
# Здесь проверяются два условия: условие Армихо и условие кривизны.
def backtracking_wolfe(f, grad_f, point, alpha, c1, c2, tau, max_iter=50):
    grad_x = grad_f(point)
    phi0 = f(point)
    grad_norm0 = np.dot(grad_x, -grad_x)  # должно быть отрицательным
    for i in range(max_iter):
        point -= alpha * grad_x
        phi = f(point)
        grad_new = grad_f(point)
        phi_prime = np.dot(grad_new, -grad_x)
        # Условие Армихо
        if phi > phi0 + c1 * alpha * grad_norm0:
            alpha *= tau
        # Условие кривизны (Wolfe): phi_prime >= c2 * phi_prime0
        elif phi_prime < c2 * grad_norm0:
            alpha *= 1.1  # увеличиваем шаг, если наклон слишком крутой
        else:
            break
    return alpha
