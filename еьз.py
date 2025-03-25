import autograd.numpy as np
from autograd import grad

def f(point):
    x, y = point
    return (1 - x)**2 + 100 * (y - x**2) ** 2

grad_f = grad(f)
point = np.array([1.0, 1.0])
print(grad_f(point))  # Выведет: [2.0, 4.0]