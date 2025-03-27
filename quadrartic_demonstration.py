import numpy as np

# Функция для которой ищем минимум (1)
# f(x, y) = x^2 + y^2
def f(point):
    x, y = point
    return x**2 + y**2

# точность нахождения минимума
TOLERANCE = 1e-4
# базовый щаг поиска
STEP = 0.3
# Количество итераций
MAX_ITERATIONS = 10_000
# Начальная точка для градиентного спуска
START_POINT = np.array([-3., -4.])
# Коэф с1
C1 = 0.5
# Коэф с2
C2 = 0.5
# Тау
TAU = 0.7
