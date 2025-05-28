import matplotlib.ticker as ticker
import lib3
import lib2
import lib
import numpy as np
import matplotlib.pyplot as plt


from time import perf_counter
import tracemalloc
from sklearn.metrics import mean_squared_error


X, y = lib3.make_anisotropic_regression()
n, d = X.shape

def mse_loss(w):
    return mean_squared_error(y, X @ w)

x0 = np.zeros(d)
bounds = np.array([[-10,10]]*d)
ERAS = 50
def run_sgd():
    tr = lib3.Tracker("SGD")
    w = lib3.sgd(
        tracker=tr, x=X, y=y,
        eras=ERAS, batch_size=32,
        step_name='constant', step_initial=0.01, decay_rate=0,
        reg_type='l2', reg_lambda=1e-3, l1_ratio=0.5,
        eps=1e-8
    )
    return tr.history_errors[-1]

def run_sgd_mom():
    tr = lib3.Tracker("SGD+M")
    w = lib3.sgd_momentum(
        tracker=tr, x=X, y=y,
        eras=ERAS, batch_size=32,
        step_name='linear', step_initial=0.01, decay_rate=1e-3,
        beta=0.9
    )
    return tr.history_errors[-1]

def run_sgd_nest():
    tr = lib3.Tracker("SGD+N")
    w = lib3.sgd_nesterov(
        tracker=tr, x=X, y=y,
        eras=ERAS, batch_size=32,
        step_name='linear', step_initial=0.01, decay_rate=1e-3,
        beta=0.9
    )
    return tr.history_errors[-1]

def run_bfgs():
    tr = lib2.Tracker()
    w_opt = lib2.bfgs_section_search(
        f=lambda w: mse_loss(w),
        point=x0.copy(),
        tolerance=1e-6,
        max_iterations=ERAS,
        tracker=None 
    )
    return mse_loss(w_opt)

def run_sa():
    _, best_f, _ = lib.simulated_annealing(
        obj_func=lambda w: mse_loss(w),
        x0=x0.copy(),
        T0=1.0,
        t_sheduling=lib.temperature_schedule('exponential'),
        n_iter=ERAS,
        bounds=bounds,
        step_size=0.5
    )
    return best_f

methods = {
    "SGD": run_sgd,
    "SGD+Momentum": run_sgd_mom,
    "SGD+Nesterov": run_sgd_nest,
    # "BFGS": run_bfgs,
    "SA": run_sa
}
# … ваш код до этого момента …

# 3) Запускаем всё и собираем истории ошибок
results   = {}
histories = {}

for name, fn in methods.items():
    if name in ("SGD", "SGD+Momentum", "SGD+Nesterov"):
        # для SGD-семейства трекеры уже внутри, так же достаём историю
        tr = lib3.Tracker(name)
        # вызываем нужную функцию напрямую, передавая наш трекер
        if name == "SGD":
            lib3.sgd(tr, x=X, y=y, eras=ERAS, batch_size=32,
                     step_name='constant', step_initial=0.01, decay_rate=0,
                     reg_type='l2', reg_lambda=1e-3, l1_ratio=0.5, eps=1e-8)
        elif name == "SGD+Momentum":
            lib3.sgd_momentum(tr, x=X, y=y, eras=ERAS, batch_size=32,
                              step_name='linear', step_initial=0.01, decay_rate=1e-3, beta=0.9)
        else:  # "SGD+Nesterov"
            lib3.sgd_nesterov(tr, x=X, y=y, eras=ERAS, batch_size=32,
                              step_name='linear', step_initial=0.01, decay_rate=1e-3, beta=0.9)

        history = tr.history_errors
        mse_final = history[-1]

    elif name == "SA":
        _, mse_final, history = lib.simulated_annealing(
            obj_func=lambda w: mean_squared_error(y, X @ w),
            x0=x0.copy(),
            T0=1.0,
            t_sheduling=lib.temperature_schedule('exponential'),
            n_iter=ERAS,
            bounds=bounds,
            step_size=0.5
        )

    else:
        # BFGS
        w_opt = lib2.bfgs_section_search(
            f=lambda w: mean_squared_error(y, X @ w),
            point=x0.copy(),
            tolerance=1e-6,
            max_iterations=ERAS,
            tracker=None
        )
        mse_final = mean_squared_error(y, X @ w_opt)
        # у BFGS’а нет помесячной истории ошибок в вашем коде, можно построить flat-line:
        history = [mse_final] * ERAS

    results[name]   = mse_final
    histories[name] = history
    print(f"{name:12s} → final MSE = {mse_final:.3e}")

plt.figure(figsize=(8,5))
for name, hist in histories.items():
    plt.plot(range(1, len(hist)+1), hist, label=name)

plt.xlabel("Эра (Epoch)")
plt.ylabel("MSE")
plt.yscale("log")
plt.title("Сходимость методов по эпохам")
plt.legend()

ax = plt.gca()               # получаем текущую ось
# ax.set_yscale('log')         # остаёмся в лог-шкале
# Задаём ScalarFormatter без scientific notation
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

plt.show()