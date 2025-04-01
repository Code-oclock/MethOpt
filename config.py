import autograd.numpy as np


# def f(point):
#     x, y = point
#     return x**2 + y**2

# QUADRATIC GOOD
# точность нахождения минимума
# TOLERANCE = 1e-4
# # базовый щаг поиска
# STEP_FIXED = 0.3
# STEP_DECREASING = 0.9
# STEP_ARMIJO = 0.3
# STEP_WOLFE = 0.3
# # Количество итераций
# MAX_ITERATIONS = 1000
# # Начальная точка для градиентного спуска
# START_POINT = np.array([2., 4.])
# # Коэф с1
# C1 = 0.5
# # Коэф с2
# C2 = 0.5
# # Тау
# TAU = 0.2

# RESULT_FOLDER = "quadratic_good/"

# QUADRATIC BAD
# TOLERANCE = 1e-4
# # базовый щаг поиска
# STEP_FIXED = 1
# STEP_DECREASING = 0.01
# STEP_ARMIJO = 0.001
# STEP_WOLFE = 0.001
# # Количество итераций
# MAX_ITERATIONS = 1000
# # Начальная точка для градиентного спуска
# START_POINT = np.array([3.5, 3.5])
# # Коэф с1
# C1 = 0.5
# # Коэф с2
# C2 = 0.5
# # Тау
# TAU = 0.1

# RESULT_FOLDER = "quadratic_bad/"

# def f(point):
#     x, y = point
#     return 0.26 * (x**2 + y**2) - 0.48 * x * y

# # MATYAS GOOD
# TOLERANCE = 1e-4
# # базовый щаг поиска
# STEP_FIXED = 0.3
# STEP_DECREASING = 20
# STEP_ARMIJO = 0.3
# STEP_WOLFE = 0.3
# # Количество итераций
# MAX_ITERATIONS = 1000
# # Начальная точка для градиентного спуска
# START_POINT = np.array([-1., 4.])
# # Коэф с1
# C1 = 0.5
# # Коэф с2
# C2 = 0.5
# # Тау
# TAU = 0.9

# RESULT_FOLDER = "matyas_good/"

# MATYAS BAD
# TOLERANCE = 1e-4
# # базовый щаг поиска
# STEP_FIXED = 2
# STEP_DECREASING = 1
# STEP_ARMIJO = 0.3
# STEP_WOLFE = 0.3
# # Количество итераций
# MAX_ITERATIONS = 100
# # Начальная точка для градиентного спуска
# START_POINT = np.array([-1., 4.])
# # Коэф с1
# C1 = 0.5
# # Коэф с2
# C2 = 0.5
# # Тау
# TAU = 0.1

# RESULT_FOLDER = "matyas_bad/"

# def f(point):
#     x, y = point
#     return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# TOLERANCE = 1e-4
# # базовый щаг поиска
# STEP_FIXED = 0.001
# STEP_DECREASING = 0.01
# STEP_ARMIJO = 0.3
# STEP_WOLFE = 0.3
# # Количество итераций
# MAX_ITERATIONS = 10000
# # Начальная точка для градиентного спуска
# START_POINT = np.array([5., 5.])
# # Коэф с1
# C1 = 0.5
# # Коэф с2
# C2 = 0.5
# # Тау
# TAU = 0.8

# RESULT_FOLDER = "himmelblau_good_55/"

# def f(point):
#     x, y = point
#     return 0.26 * (x**2 + y**2) - 0.48 * x * y + np.random.normal(loc=0, scale=NOISE)

# TOLERANCE = 1e-7
# # базовый щаг поиска
# STEP_FIXED = 0.3
# STEP_DECREASING = 0.01
# STEP_ARMIJO = 0.3
# STEP_WOLFE = 0.3
# # Количество итераций
# MAX_ITERATIONS = 10000
# # Начальная точка для градиентного спуска
# START_POINT = np.array([-1., 3.5])
# # Коэф с1
# C1 = 0.5
# # Коэф с2
# C2 = 0.5
# # Тау
# TAU = 0.8
# NOISE = 1

# RESULT_FOLDER = "matyas_with_noise_0.01/"

def f(point):
    x, y = point
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2 + np.random.normal(loc=0, scale=NOISE)

TOLERANCE = 1e-4
# базовый щаг поиска
STEP_FIXED = 0.01
STEP_DECREASING = 0.01
STEP_ARMIJO = 0.3
STEP_WOLFE = 0.03
# Количество итераций
MAX_ITERATIONS = 1000
# Начальная точка для градиентного спуска
# START_POINT = np.array([5., 5.]) # for fixed min is [2.999998518 2.000003579]
# START_POINT = np.array([5., -5.]) # for fixed min is [3.584428654 -1.848129961]
# START_POINT = np.array([-5., 5.]) # for fixed min is [3.584428654 -1.848129961]
START_POINT = np.array([-0.1, -0.5]) # for fixed min is [3.584428654 -1.848129961]
# Коэф с1
C1 = 0.5
# Коэф с2
C2 = 0.5
# Тау
TAU = 0.8
NOISE = 10

RESULT_FOLDER = "himmelblau_good_with_noise_0.01/"