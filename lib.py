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
def gradient_descent_fixed(
    f, 
    point: np.array,
    step: float, 
    tolerance: float, 
    max_iterations: int, 
    tracker: Tracker
) -> np.array:
    tracker.track(point)

    for _ in range(max_iterations):
        current_gradient = gradient(f, point)
        # Если норма градиента меньше заданной точности, то завершаем поиск
        if np.linalg.norm(current_gradient) < tolerance: break

        # Обновляем координаты x_k = x_k-1 - h * grad(f)
        point -= step * current_gradient
        
        tracker.track(point)
    return np.array(point)

# Градиентный спуск с уменьшающимся шагом
def gradient_descent_decreasing(
    f, 
    point: np.array, 
    step: float, 
    tolerance: float, 
    max_iterations: int, 
    tracker: Tracker
) -> np.array:
    tracker.track(point)

    for i in range(1, max_iterations):
        current_gradient = gradient(f, point)
        # Если норма градиента меньше заданной точности, то завершаем поиск
        if np.linalg.norm(current_gradient) < tolerance: break

        # Обновляем координаты x_k = x_k-1 - (h / k) * grad(f)
        point -= step / i * current_gradient
        
        tracker.track(point)
    return np.array(point)

# Стратегия выбора шага по условию Армихо
def backtracking_armijo(
    f, 
    point: np.array, 
    alpha: float, 
    c: float, 
    tau
):
    # Условие Армихо: 
    #     f(x - alpha * grad) <= f(x) - c * alpha * (||grad||^2)
    #
    #   f(x - alpha * grad) 
    #     - текущее значение после шага спуска
    #   f(x) - c * alpha * (||grad||^2) 
    #     - минимально допустимое уменьшение функции

    current_gradient = gradient(f, point)
    while f(point - alpha * current_gradient) > f(point) - c * alpha * np.dot(current_gradient, current_gradient):
        alpha *= tau
    return alpha

#def gradient_descent_armijo(f, point: np.array, h, tol=1e-6, max_iter=100_000, c=1e-4, tau=0.1, tracker: Tracker):

# Градиентный спуск с выбором шага по условию Армихо
def gradient_descent_armijo(
    f, 
    point: np.array, 
    step: float, 
    tolerance: float, 
    max_iterations: int, 
    tau: float, 
    tracker: Tracker
) -> np.array:
    tracker.track(point)
    for _ in range(max_iterations):
        current_gradient = gradient(f, point)
        # Если норма градиента меньше заданной точности, то завершаем поиск
        if np.linalg.norm(current_gradient) < tolerance: break

        # Находим шаг удовлетворяющий условию Армихо
        alpha = backtracking_armijo(f, current_gradient, point, step, np.random.uniform(0, 1), tau)
        point -= alpha * current_gradient
        tracker.track(point)
        
    return np.array(point)

# Стратегия выбора шага по условию Вульфа
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

# def gradient_descent_wolfe(f, point: np.array, step: float, tolerance: float, max_iterations, c1=0.1, c2=0.9, tau=0.7):
# Градиентный спуск с выбором шага по условию Вульфа
def gradient_descent_wolfe(f, point: np.array, step: float, tolerance: float, max_iterations, c1: float, c2: float, tau: float, tracker: Tracker):
    tracker.track(point)

    for i in range(max_iterations):
        current_gradient = gradient(point)
        if np.linalg.norm(current_gradient) < tolerance:
            break
        alpha = backtracking_wolfe(f, grad_f, point, step, c1, c2, tau)
        point -= alpha * grad
        tracker.track(point)

    return np.array(point)

#############################
# Часть 2. Одномерный поиск минимума
#############################

# Здесь предполагается, что функция f(point) является одномерной и имеет один минимум на отрезке [start, end].
def golden_section_search(f, start: int, end: int, tolerance: float, max_iterations: int) -> float:
    phi = (np.sqrt(5) - 1) / 2  # Золотое сечение ~0.618
    # Находим две точки c и d, которые делят отрезок [start, end] в пропорции золотого сечения: [start, c, d, end]
    c = end - (end - start) * phi
    d = start + (end - start) * phi

    for _ in range(max_iterations):
        # Если разница между концами отрезка меньше заданной точности, то завершаем поиск
        if abs(end - start) < tolerance:
            break
        # Если значение функции в точке c меньше, чем в точке d, то сдвигаем правую границу (иначе левую)
        if f(c) < f(d):
            end = d # сдвигаем правую границу
            d = c # сдвигаем правую границу разделения отрзка
            c = end - (end - start) * phi # новая пропорция (левая граница) разделения отрзка
        else:
            start = c # сдвигаем левую границу
            c = d # сдвигаем правую леву. границу разделения отрзка
            d = start + (end - start) * phi # новая пропорция (правая граница) разделения отрзка
    middle = (start + end) / 2
    return middle

def bisection_search(f, start: int, end: int, tolerance: float, max_iterations: int) -> float:
    delta = tolerance
    for _ in range(max_iterations):
        # Если разница между концами отрезка меньше заданной точности, то завершаем поиск
        if abs(end - start) < tolerance:
            break
        # Находим середину отрезка
        mid = (start + end) / 2
        # Отступаем от середины на delta
        left = mid - delta
        right = mid + delta

        # Если значение функции в левой точке меньше, чем в правой, то сдвигаем правую границу (иначе левую)
        if f(left) < f(right):
            end = right
        else:
            start = left 
    # Возвращаем середину отрезка
    middle = (start + end) / 2
    return middle

def gradient_descent_golden(f, point: np.array, tolerance: float, max_iterations: int, tracker: Tracker) -> np.array:
    tracker.track(point)

    for _ in range(max_iterations):
        current_gradient = gradient(f, point)
        # Если норма градиента меньше заданной точности, то завершаем поиск
        if np.linalg.norm(current_gradient) < tolerance:
            break
        # g - функция одной переменной (сечение фукнции f плоскостью) - для подбора шага
        def g(alpha): return f(point - alpha * current_gradient)
        # Поиск шага методом золотого сечения
        step = golden_section_search(g, 0, 1, 1e-6, 100)
        # Обновляем координаты x_k = x_k-1 - h * grad(f)
        point -= step * current_gradient

        tracker.track(point)
    return np.array(point)

def gradient_descent_dichotomy(f, point: np.array, tolerance: float, max_iterations: int, tracker: Tracker) -> np.array:
    tracker.track(point)

    for _ in range(max_iterations):
        current_gradient = gradient(f, point)
        # Если норма градиента меньше заданной точности, то завершаем поиск
        if np.linalg.norm(current_gradient) < tolerance:
            break
        # g - функция одной переменной (сечение фукнции f плоскостью) - для подбора шага
        def g(alpha): return f(point - alpha * current_gradient)
        # Поиск шага методом дихотомии
        step = bisection_search(g, 0, 1, 1e-6, 100)
        # Обновляем координаты x_k = x_k-1 - h * grad(f)
        point -= step * current_gradient

        tracker.track(point)
    return np.array(point)
