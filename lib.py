from typing import List
import numpy as np
from autograd import grad

# Градиент функции f в точке
def gradient(f, point: np.array):
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
    
    @property
    def iterations(self) -> int:
        return self.__iterations
    
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
) -> float:
    # Условие Армихо: 
    #     f(x - alpha * grad) <= f(x) - c * alpha * (||grad||^2)
    #
    #   f(x - alpha * grad) 
    #     - текущее значение после шага спуска
    #   f(x) - c * alpha * (||grad||^2) 
    #     - минимально допустимое уменьшение функции

    current_gradient = gradient(f, point)
    while f(point - alpha * current_gradient) > f(point) - c * alpha * np.dot(current_gradient, current_gradient):
        alpha = alpha * tau
    return alpha

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
        alpha = backtracking_armijo(f, point, step, np.random.uniform(0, 1), tau)
        point -= alpha * current_gradient
        tracker.track(point)
        
    return np.array(point)

# Стратегия выбора шага по условию Вульфа
def backtracking_wolfe(
    f, 
    point: np.array, 
    alpha: float, 
    c1: float, 
    c2: float, 
    tau: float, 
    max_iterations: int
) -> float:
    current_gradient = gradient(f, point)
    for _ in range(max_iterations):
        # проверяем условие Армихо
        if f(point - alpha * current_gradient) > f(point) - c1 * alpha * np.dot(current_gradient, current_gradient):
            alpha = alpha * tau
        elif f(point - alpha * current_gradient) < c2 * np.dot(current_gradient, current_gradient):
            alpha = alpha + alpha * tau
        else: 
            break

    return alpha

# Градиентный спуск с выбором шага по условию Вульфа
def gradient_descent_wolfe(
    f, 
    point: np.array, 
    step: float, 
    tolerance: float, 
    max_iterations: int, 
    c1: float, 
    c2: float, 
    tau: float, 
    tracker: Tracker
) -> np.array:
    tracker.track(point)

    for _ in range(max_iterations):
        current_gradient = gradient(f, point)
        # Если норма градиента меньше заданной точности, то завершаем поиск
        if np.linalg.norm(current_gradient) < tolerance: break

        alpha = backtracking_wolfe(f, point, step, c1, c2, tau, max_iterations // 100)
        point -= alpha * current_gradient
        tracker.track(point)

    return np.array(point)

#----------------------
#       ЧАСТЬ 2       |
#----------------------

# Здесь предполагается, что функция f(point) является одномерной и имеет один минимум на отрезке [start, end].


def golden_section_search(f, start: int, end: int, tolerance: float, max_iterations: int) -> float:
    phi = (np.sqrt(5) - 1) / 2  # Золотое сечение ~0.618
    # Находим две точки c и d, которые делят отрезок [start, end] в пропорции золотого сечения: [start, c, d, end]
    c = end - (end - start) * phi
    d = start + (end - start) * phi

    for _ in range(max_iterations):
        # Если разница между концами отрезка меньше заданной точности, то завершаем поиск
        if abs(end - start) < tolerance: break
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

def gradient_descent_golden(f, point: np.array, tolerance: float, max_iterations: int, tracker: Tracker) -> np.array:
    tracker.track(point)

    for _ in range(max_iterations):
        current_gradient = gradient(f, point)
        # Если норма градиента меньше заданной точности, то завершаем поиск
        if np.linalg.norm(current_gradient) < tolerance: break

        # g - функция одной переменной (сечение фукнции f плоскостью) - для подбора шага
        def g(alpha): return f(point - alpha * current_gradient)
        # Поиск шага методом золотого сечения
        step = golden_section_search(g, 0, 1, tolerance / 100, max_iterations // 1000)
        # Обновляем координаты x_k = x_k-1 - h * grad(f)
        point -= step * current_gradient

        tracker.track(point)
    return np.array(point)

def bisection_search(f, start: int, end: int, tolerance: float, max_iterations: int) -> float:
    delta = tolerance
    for _ in range(max_iterations):
        # Если разница между концами отрезка меньше заданной точности, то завершаем поиск
        if abs(end - start) < tolerance: break

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

def gradient_descent_dichotomy(f, point: np.array, tolerance: float, max_iterations: int, tracker: Tracker) -> np.array:
    tracker.track(point)

    for _ in range(max_iterations):
        current_gradient = gradient(f, point)
        # Если норма градиента меньше заданной точности, то завершаем поиск
        if np.linalg.norm(current_gradient) < tolerance: break
        
        # g - функция одной переменной (сечение фукнции f плоскостью) - для подбора шага
        def g(alpha): return f(point - alpha * current_gradient)
        # Поиск шага методом дихотомии
        step = bisection_search(g, 0, 1, tolerance / 100, max_iterations // 1000)
        # Обновляем координаты x_k = x_k-1 - h * grad(f)
        point -= step * current_gradient

        tracker.track(point)
    return np.array(point)
