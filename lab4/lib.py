import numpy as np

def simulated_annealing(obj_func, x0, T0=1.0, alpha=0.99, n_iter=1000, bounds=None, step_size=0.1):
    """
    step_size — стандартное отклонение нормальных шагов (раньше было жёстко 0.1).
    """
    x = x0.copy()
    f_x = obj_func(x)
    best_x, best_f = x.copy(), f_x
    T = T0
    history = []

    for i in range(n_iter):
        # теперь используем step_size
        x_new = x + np.random.normal(scale=step_size, size=x.shape)
        if bounds is not None:
            x_new = np.clip(x_new, bounds[:,0], bounds[:,1])
        f_new = obj_func(x_new)
        delta = f_new - f_x

        if (delta < 0) or (np.random.rand() < np.exp(-delta / T)):
            x, f_x = x_new, f_new
            if f_x < best_f:
                best_x, best_f = x.copy(), f_x

        T *= alpha
        history.append(best_f)

    return best_x, best_f, history

def genetic_algorithm(obj_func, bounds, pop_size=50, n_gen=100, cross_rate=0.8, mut_rate=0.1):
    """
    Возвращает: best_x, best_f, history
    """
    # TODO: реализовать
    pass