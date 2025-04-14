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

def newton_method(
        f, 
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

        # Вычисляем направление: d = H^(-1) * grad
        d = np.linalg.solve(current_hessian, current_gradient)

        # Выбор шага методом backtracking line search
        # g - функция одной переменной (сечение фукнции f плоскостью) - для подбора шага
        def g(alpha): return f(point - alpha * d)
        # Поиск шага методом золотого сечения
        step = golden_section_search(g, 0, 1, tolerance / 10, max_iterations // 100)

        point -= step * d
        if tracker is not None: tracker.track(point)
        
    return point
