import numpy as np

class Tracker:
    def __init__(self, name="") -> None:
        self.__history = {'errors':[], 'steps':[]}
        self.__total_flops = 0
        self.__eras = 0
        self.__name = name

    def track(self, error: float, step: float, flops_epoch: int) -> None:
        self.__history['errors'].append(error)
        self.__history['steps'].append(step)

        self.__total_flops += flops_epoch
        self.__eras += 1

    @property
    def name(self) -> str:
        return self.__name    

    @property
    def history_errors(self) -> list[float]:
        return self.__history["errors"]
    
    @property
    def history_steps(self) -> list[float]:
        return self.__history["steps"]
    @property
    def total_flops(self) -> list[float]:
        return self.__total_flops
    @property
    def eras(self) -> int:
        return self.__eras


def rosenbrock(x):
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rastrigin(x):
    A = 10
    return A * x.size + np.sum(x**2 - A * np.cos(2 * np.pi * x))


def temperature_schedule(name: str):
    if name == 'linear':
        return lambda t: 1.0 - 0.001 * t
    elif name == 'exponential':
        return lambda t: 0.99 ** t
    elif name == 'logarithmic':
        return lambda t: 1.0 / np.log(t + 1)
    elif name == 'constant_step':
        return lambda t: 0.5 ** (t // 100)
    else:
        raise ValueError("Unknown temperature schedule type.")

def simulated_annealing(obj_func, x0, T0, t_sheduling, n_iter, bounds, step_size):
    x = x0.copy()
    f_x = obj_func(x)
    best_x, best_f = x.copy(), f_x
    T = T0
    history = []

    for i in range(n_iter):
        x_new = x + np.random.normal(scale=step_size, size=x.shape)
        if bounds is not None:
            x_new = np.clip(x_new, bounds[:,0], bounds[:,1])
        f_new = obj_func(x_new)
        delta = f_new - f_x

        if (delta < 0) or (np.random.rand() < np.exp(-delta / T)):
            x, f_x = x_new, f_new
            if f_x < best_f:
                best_x, best_f = x.copy(), f_x

        T = t_sheduling(T)
        history.append(best_f)

    return best_x, best_f, history
