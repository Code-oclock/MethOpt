import optuna
import numpy as np
from sklearn.metrics import mean_squared_error
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
import lib



optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---- ваши реализации ----
# load_dataset, Tracker, sgd, step_schedule, regularization_gradient, count_flops

# 1) Загружаем данные
X, y = lib.load_dataset(id=186)  # замените id на нужный

# 2) Настройки
step_names = ['constant', 'linear', 'exponential']
reg_types  = ['none', 'l2', 'l1', 'elastic']
REG_LAMBDA = 1e-3
L1_RATIO   = 0.5
ERAS       = 50
EPS        = 1e-8

results = {}

for step_name in step_names:
    for reg_type in reg_types:
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        def objective(trial: optuna.Trial) -> float:
            batch_size   = trial.suggest_categorical('batch_size', [16,32,64,128])
            step_initial = trial.suggest_float('step_initial', 1e-4, 1e-1)
            decay_rate   = trial.suggest_float('decay_rate',   1e-4, 1e-1)

            tracker = lib.Tracker()
            w = lib.sgd(
                tracker=tracker,
                x=X, y=y,
                eras=ERAS,
                batch_size=batch_size,
                step_name=step_name,
                step_initial=step_initial,
                decay_rate=decay_rate,
                reg_type=reg_type,
                reg_lambda=REG_LAMBDA,
                l1_ratio=L1_RATIO,
                eps=EPS
            )
            return mean_squared_error(y, X.dot(w))

        # запускаем оптимизацию (например, 30 испытаний)
        study.optimize(objective, n_trials=30)

        # сохраняем результаты
        results[(step_name, reg_type)] = {
            'mse':    study.best_value,
            'params': study.best_params
        }

        print(f">>> step={step_name}, reg={reg_type}  →  "
              f"best MSE={study.best_value:.4f},  params={study.best_params}")

# По окончании в словаре results будет всё, что нужно