import numpy as np
from autograd import grad as autograd_gradient
from autograd import hessian as autograd_hessian

class Tracker:
    def __init__(self) -> None:
        self.__path = [[], []]
        self.__iterations = 0
        self.__f_calls = 0
        self.__gradient_calls = 0
        self.__hessian_calls = 0

    def track_point(self, point: np.array) -> None:
        self.__path[0].append(point[0])
        self.__path[1].append(point[1])

        self.__iterations += 1

    def f_called(self) -> None:
        self.__f_calls += 1

    def gradient_called(self) -> None:
        self.__gradient_calls += 1

    def hessian_called(self) -> None:
        self.__hessian_calls += 1

    def print_stats(self) -> None:
        print(f"#-------STATS---------")
        print(f"|  Iterations: {self.__iterations}")
        print(f"|  Func calls: {self.__f_calls}")
        print(f"|  Grad calls: {self.__gradient_calls}")
        print(f"|  Hess calls: {self.__hessian_calls}")
        print("#---------------------")

    @property
    def coordinates(self) -> np.array:
        return self.__path
    
    @property
    def iterations(self) -> int:
        return self.__iterations
    
    @iterations.setter
    def iterations(self, value: int) -> None:
        self.__iterations = value
    
# Градиент функции f в точке
def gradient(f, point: np.array, tracker: Tracker = None):
    if tracker is not None: tracker.gradient_called()
    grad_f = autograd_gradient(f)
    return grad_f(point)

# Гессиан функции f в точке
def hessian(f, point: np.array, tracker: Tracker = None):
    if tracker is not None: tracker.hessian_called()
    hessian_f = autograd_hessian(f)
    return hessian_f(point)

def to_tracked_f(f, tracker: Tracker):
    def tracked_f(*args, **kwargs):
        tracker.f_called()
        return f(*args, **kwargs)
    return tracked_f

# Градиентный спуск с константным шагом
def gradient_descent_fixed(
    f, 
    point: np.array,
    step: float, 
    tolerance: float, 
    max_iterations: int, 
    tracker: Tracker = None
) -> np.array:
    if tracker is not None: 
        tracker.track_point(point)
        f = to_tracked_f(f, tracker)

    for _ in range(max_iterations):
        current_gradient = gradient(f, point, tracker)
        # Если норма градиента меньше заданной точности, то завершаем поиск
        if np.linalg.norm(current_gradient) < tolerance: break

        # Обновляем координаты x_k = x_k-1 - h * grad(f)
        point -= step * current_gradient
        
        if tracker is not None: tracker.track_point(point)
    return np.array(point)

# Градиентный спуск с уменьшающимся шагом
def gradient_descent_decreasing(
    f, 
    point: np.array, 
    step: float, 
    tolerance: float, 
    max_iterations: int, 
    tracker: Tracker = None
) -> np.array:
    if tracker is not None: 
        tracker.track_point(point)
        f = to_tracked_f(f, tracker)

    for i in range(1, max_iterations):
        current_gradient = gradient(f, point, tracker)
        # Если норма градиента меньше заданной точности, то завершаем поиск
        if np.linalg.norm(current_gradient) < tolerance: break

        # Обновляем координаты x_k = x_k-1 - (h / k) * grad(f)
        point -= step / i * current_gradient
        
        if tracker is not None: tracker.track_point(point)
    return np.array(point)

# Стратегия выбора шага по условию Армихо
def backtracking_armijo(
    f, 
    point: np.array, 
    alpha: float, 
    c: float, 
    tau: float, 
    tracker: Tracker = None
) -> float:
    # Условие Армихо: 
    #     f(x - alpha * grad) <= f(x) - c * alpha * (||grad||^2)
    #
    #   f(x - alpha * grad) 
    #     - текущее значение после шага спуска
    #   f(x) - c * alpha * (||grad||^2) 
    #     - минимально допустимое уменьшение функции

    current_gradient = gradient(f, point, tracker)
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
    tracker: Tracker = None
) -> np.array:
    if tracker is not None: 
        tracker.track_point(point)
        f = to_tracked_f(f, tracker)

    for _ in range(max_iterations):
        current_gradient = gradient(f, point, tracker)
        # Если норма градиента меньше заданной точности, то завершаем поиск
        if np.linalg.norm(current_gradient) < tolerance: break

        # Находим шаг удовлетворяющий условию Армихо
        alpha = backtracking_armijo(f, point, step, np.random.uniform(0, 1), tau, tracker)
        point -= alpha * current_gradient
        if tracker is not None: tracker.track_point(point)
        
    return np.array(point)

# Стратегия выбора шага по условию Вульфа
def backtracking_wolfe(
    f, 
    point: np.array, 
    alpha: float, 
    c1: float, 
    c2: float, 
    tau: float, 
    max_iterations: int,
    tracker: Tracker = None
) -> float:
    current_gradient = gradient(f, point, tracker)
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
    tracker: Tracker = None
) -> np.array:
    if tracker is not None: 
        tracker.track_point(point)
        f = to_tracked_f(f, tracker)

    for _ in range(max_iterations):
        current_gradient = gradient(f, point, tracker)
        # Если норма градиента меньше заданной точности, то завершаем поиск
        if np.linalg.norm(current_gradient) < tolerance: break

        alpha = backtracking_wolfe(f, point, step, c1, c2, tau, max_iterations // 100, tracker)
        point -= alpha * current_gradient
        if tracker is not None: tracker.track_point(point)

    return np.array(point)

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

def gradient_descent_golden(
    f, 
    point: np.array, 
    tolerance: float, 
    max_iterations: int, 
    tracker: Tracker = None
) -> np.array:
    if tracker is not None: 
        tracker.track_point(point)
        f = to_tracked_f(f, tracker)

    for _ in range(max_iterations):
        current_gradient = gradient(f, point, tracker)
        # Если норма градиента меньше заданной точности, то завершаем поиск
        if np.linalg.norm(current_gradient) < tolerance: break

        # g - функция одной переменной (сечение фукнции f плоскостью) - для подбора шага
        def g(alpha): return f(point - alpha * current_gradient)
        # Поиск шага методом золотого сечения
        step = golden_section_search(g, 0, 1, tolerance / 10, max_iterations // 100)
        # Обновляем координаты x_k = x_k-1 - h * grad(f)
        point -= step * current_gradient

        if tracker is not None: tracker.track_point(point)
    return np.array(point)

def bisection_search(
    f, 
    start: int, 
    end: int, 
    tolerance: float, 
    max_iterations: int,
    tracker: Tracker = None
) -> float:
    if tracker is not None: f = to_tracked_f(f, tracker)

    for _ in range(max_iterations):
        # Если разница между концами отрезка меньше заданной точности, то завершаем поиск
        if abs(end - start) < tolerance: break

        # Находим середину отрезка
        mid = (start + end) / 2
        # Отступаем от середины на tolerance
        left = mid - tolerance
        right = mid + tolerance

        # Если значение функции в левой точке меньше, чем в правой, то сдвигаем правую границу (иначе левую)
        if f(left) < f(right): end = right
        else:                  start = left 
    # Возвращаем середину отрезка
    middle = (start + end) / 2
    return middle

def gradient_descent_dichotomy(
    f, 
    point: np.array, 
    tolerance: float, 
    max_iterations: int, 
    tracker: Tracker = None
) -> np.array:
    if tracker is not None: 
        tracker.track_point(point)
        f = to_tracked_f(f, tracker)

    for _ in range(max_iterations):
        current_gradient = gradient(f, point, tracker)
        # Если норма градиента меньше заданной точности, то завершаем поиск
        if np.linalg.norm(current_gradient) < tolerance: break
        
        # g - функция одной переменной (сечение фукнции f плоскостью) - для подбора шага
        def g(alpha): return f(point - alpha * current_gradient)
        # Поиск шага методом дихотомии
        step = bisection_search(g, 0, 1, tolerance / 10, max_iterations // 100, tracker)
        # Обновляем координаты x_k = x_k-1 - h * grad(f)
        point -= step * current_gradient

        if tracker is not None: tracker.track_point(point)
    return np.array(point)

def newton_method(
    f,
    finding_step, 
    point: np.array, 
    tolerance: float, 
    max_iterations: int, 
    tracker: Tracker = None
) -> np.array:
    if tracker is not None: 
        tracker.track_point(point)
        f = to_tracked_f(f, tracker)

    for _ in range(max_iterations):
        current_gradient = gradient(f, point, tracker)
        if np.linalg.norm(current_gradient) < tolerance:
            break
        current_hessian = hessian(f, point, tracker)

        # исправляем кривизну
        min_eigval = np.linalg.eigvals(current_hessian).min()
        if min_eigval <= 0:
            current_hessian += (abs(min_eigval) + tolerance) * np.eye(len(point))

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
        if tracker is not None: tracker.track_point(point)
        
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
        return backtracking_wolfe( f, point, 1, 0.5, 0.5, 0.8, max_iterations // 100, tracker)
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
        return backtracking_armijo(f, point, 0.3, np.random.uniform(0, 1), 0.8, tracker)
    return newton_method(f, backtracking_armijo_search, point, tolerance, max_iterations, tracker)


def bfgs_section_search(
    f,
    point: np.array,
    tolerance: float,
    max_iterations: int,
    tracker: Tracker = None
) -> np.array:
    if tracker is not None: 
        tracker.track_point(point)
        f = to_tracked_f(f, tracker)
    # размерность пространства
    dimension = len(point)
    # начальная матрица Гессе
    B = np.eye(dimension)
    # начальный градиент
    current_gradient = gradient(f, point, tracker)
    
    for _ in range(max_iterations):
        if np.linalg.norm(current_gradient) < tolerance:
            break
        # направление спуска
        direct = -B @ current_gradient
        # g - функция одной переменной (сечение фукнции f плоскостью) - для подбора шага
        def g(alpha): return f(point + alpha * direct)
        # шаг спуска
        alpha = golden_section_search(g, 0, 1, tolerance, max_iterations // 100)
        # обновляем точку
        point_new = point + alpha * direct
        # новый градиент
        new_gradient = gradient(f, point_new, tracker)
        # разница градиента
        s = point_new - point
        y = new_gradient - current_gradient
        # переписываем точку и градиент
        point = point_new
        current_gradient = new_gradient
        # проверяем условие остановки
        if y @ s <= 0:
            break 
        # y @ s = (grad_new - grad) @ (x_new - x)
        rho = 1.0 / (y @ s)
        # обновляем матрицу Гессе
        I = np.eye(dimension)
        V = (I - rho * y @ s)
        B = V @ B @ V + rho * s @ s
        if tracker is not None: tracker.track_point(point)

    return point
