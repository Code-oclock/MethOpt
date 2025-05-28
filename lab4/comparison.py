from lib3 import sgd, sgd_momentum, sgd_nesterov
from lib2 import bfgs_section_search

import numpy as np
import matplotlib.pyplot as plt

from lib import (
    simulated_annealing, temperature_schedule
)

# -----------------------
# 1. Синтетический датасет
# y = θ0 + θ1·x1 + θ2·x2 + шум
np.random.seed(42)
n_samples, dim = 500, 2
X = np.random.uniform(-5, 5, size=(n_samples, dim))
θ_true = np.array([1.5, -2.0, 0.5])
y = θ_true[0] + X.dot(θ_true[1:]) + np.random.normal(scale=0.5, size=n_samples)

def run_sgd(trainer_fn):
    tr = Tracker()
    w = trainer_fn(tr)          # тренируем, передав трекер
    return w, tr.mse_history

# MSE-функция
def mse_loss(w):
    y_pred = X.dot(w[1:]) + w[0]
    return np.mean((y - y_pred)**2)

# начальное приближение и границы для SA / BFGS
x0     = np.zeros(dim+1)
bounds = np.array([[-10,10]]*(dim+1))

# лёгкий трекер для SGD-фамилии
class Tracker:
    def __init__(self):
        self.mse_history = []
    def track(self, mse, *args):
        self.mse_history.append(mse)
    def track_point(self, *args):
        pass
    def gradient_called(self):
        # этот метод требуется lib2.bfgs_section_search, 
        # но нам пока не нужно в нём ничего делать
        pass
    def f_called(self):
        # этот метод требуется lib2.bfgs_section_search, 
        # но нам пока не нужно в нём ничего делать
        pass

# обёртка для BFGS, чтобы вернуть историю MSE
def run_bfgs(f, x0):
    tr = Tracker()
    x_opt = bfgs_section_search(
        f, point=x0.copy(),
        tolerance=1e-6, max_iterations=200,
        tracker=tr
    )
    # bfgs_section_search не трекает MSE, но в tr.mse_history остаются точки
    # если нет – пересчитаем:
    if not tr.mse_history:
        tr.mse_history = [f(x_opt)]
    return x_opt, tr.mse_history


def run_sa(f, x0):
    tr = Tracker()
    x_opt = simulated_annealing(
        f, x0,
        T0=5.0,
        t_sheduling=temperature_schedule('exponential'),
        n_iter=500,
        bounds=bounds,
        step_size=0.5,
        tracker=tr              # ← record MSE at each step
    )
    # if SA never called the tracker, at least record the final point
    if not tr.mse_history:
        tr.mse_history = [f(x_opt)]
    return x_opt, tr.mse_history

# словарь методов и их “хорошие” гиперпараметры
methods = {
    'SGD':   lambda: run_sgd(lambda tr: sgd(
                    tr, X, y,
                    eras=100, batch_size=32,
                    step_name='constant', step_initial=0.01, decay_rate=0,
                    reg_type='l2', reg_lambda=1e-3, l1_ratio=0.5,
                    eps=1e-8
                )),
    'SGD+M': lambda: run_sgd(lambda tr: sgd_momentum(
                    tr, X, y,
                    eras=100, batch_size=32,
                    step_name='linear', step_initial=0.01, decay_rate=1e-3,
                    beta=0.9
                )),
    'SGD+N': lambda: run_sgd(lambda tr: sgd_nesterov(
                    tr, X, y,
                    eras=100, batch_size=32,
                    step_name='linear', step_initial=0.01, decay_rate=1e-3,
                    beta=0.9
                )),
    'BFGS':  lambda: run_bfgs(mse_loss, x0),
    'SA':    lambda: run_sa(mse_loss, x0.copy())
}

final_results = {}
histories     = {}

for name, runner in methods.items():
    out = runner()
    if name.startswith('SGD'):
        w = out
        w, history = out  
        mse_final = history[-1]
    else:
        w, history = out
        mse_final   = history[-1] if len(history)>1 else mse_loss(w)

    final_results[name] = mse_final
    histories    [name] = history
    print(f"{name:6s} → final MSE = {mse_final:.3e}")

plt.figure(figsize=(8,5))
for name, hist in histories.items():
    plt.plot(hist, label=name)
plt.yscale('log')
plt.xlabel('Итерация / Эра')
plt.ylabel('MSE')
plt.title('Сходимость методов на синтетической регрессии')
plt.legend()
plt.savefig("comparison_sa.png")
