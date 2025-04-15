import numpy as np
from autograd import grad
from autograd import hessian as autograd_hessian


# Градиент функции f в точке
def gradient(f, point: np.array):
    grad_f = grad(f)
    return grad_f(point)

def hessian(f, point: np.array):
    hessian_f = autograd_hessian(f)
    return hessian_f(point)

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


def golden_section_search(
    f, 
    start: int, 
    end: int, 
    tolerance: float, 
    max_iterations: int
) -> float:
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

# Стратегия выбора шага по условию Армихо
def backtracking_armijo(
    f, 
    point: np.array, 
    alpha: float, 
    c: float, 
    tau: float
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

def newton_method(
        f,
        finding_step, 
        point: np.array, 
        tolerance: float, 
        max_iterations: int, 
        tracker: Tracker = None
) -> np.array:
    if tracker is not None: tracker.track(point)

    for _ in range(max_iterations):
        current_gradient = gradient(f, point)
        if np.linalg.norm(current_gradient) < tolerance:
            break
        current_hessian = hessian(f, point)

        # Вычисляем направление: d = H^(-1) * grad, если вычислить невозможно то переходим на шаг по градиенту
        try:
            d = np.linalg.solve(current_hessian, current_gradient)
        except np.linalg.LinAlgError:
            d = current_gradient

        # Выбор шага методом backtracking line search
        # g - функция одной переменной (сечение фукнции f плоскостью) - для подбора шага
        def g(alpha): return f(point - alpha * d)
        # Поиск шага методом золотого сечения
        step = finding_step(g, 0, 1, tolerance, max_iterations // 100)
        point -= step * d
        if tracker is not None: tracker.track(point)
        
    return point

def newton_method_with_golden(
        f,
        point: np.array, 
        tolerance: float, 
        max_iterations: int, 
        tracker: Tracker = None
) -> np.array:
    return newton_method(f, golden_section_search, point, tolerance, max_iterations, tracker)

def newton_method_with_wolfe(
        f,
        point: np.array, 
        tolerance: float, 
        max_iterations: int, 
        tracker: Tracker = None
) -> np.array:
    def backtracking_wolfe_search(
            f, start: int, end: int, tolerance: float, max_iterations: int):
        return backtracking_wolfe( f, point, 1, 0.5, 0.5, 0.8, max_iterations // 100)
    return newton_method(f, backtracking_wolfe_search, point, tolerance, max_iterations, tracker)


def newton_method_with_armijo(
        f,
        point: np.array, 
        tolerance: float, 
        max_iterations: int, 
        tracker: Tracker = None
) -> np.array:
    def backtracking_armijo_search(
            f, start: int, end: int, tolerance: float, max_iterations: int):
        return backtracking_armijo(f, point, 1, tolerance, 0.8)
    return newton_method(f, backtracking_armijo_search, point, tolerance, max_iterations, tracker)


def bfgs_section_search(
        f,
        point: np.array,
        tolerance: float,
        max_iterations: int,
        tracker: Tracker = None
) -> np.array:
    # размерность пространства
    dimension = len(point)
    # начальная матрица Гессе
    B = np.eye(dimension)
    # начальный градиент
    current_gradient = gradient(f, point)
    prev_f = f(point)

    if tracker is not None: tracker.track(point)
    
    for _ in range(max_iterations):
        if np.linalg.norm(current_gradient) < tolerance:
            break
        # направление спуска
        pk = -B @ current_gradient
        # шаг спуска
        alpha = backtracking_armijo(f, point, 1, 0.5, 0.05)
        # обновляем точку
        point_new = point + alpha * pk
        # новый градиент
        new_gradient = gradient(f, point_new)
        # разница градиента
        s = point_new - point
        y = new_gradient - current_gradient
        # переписываем точку и градиент
        point = point_new
        current_gradient = new_gradient
        # проверяем условие остановки
        if y @ s <= 1e-5:
            continue 
        # определяем скалярное произведение
        # y @ s = (grad_new - grad) @ (x_new - x)
        rho = 1.0 / (y @ s)
        # обновляем матрицу Гессе
        B = (np.eye(dimension) - rho * y @ s) @ B @ (np.eye(dimension) - rho * y @ s) + rho * s @ s
        min_eigenval = np.min(np.linalg.eigvalsh(B))
        if min_eigenval < 1e-6:
            B += (1e-6 - min_eigenval) * np.eye(dimension)


        new_f = f(point)
        if abs(new_f - prev_f) < 1e-12 * (abs(new_f) + 1):
            break
        prev_f = new_f

        if tracker is not None: tracker.track(point)

    return point
