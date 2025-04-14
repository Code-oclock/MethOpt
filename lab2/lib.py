import numpy as np
from autograd import grad

# Градиент функции f в точке
def gradient(f, point: np.array):
    grad_f = grad(f)
    return grad_f(point)

def hessian(f, point: np.array):
    grad_f = grad(f)
    hessian_f = grad(grad_f)
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
