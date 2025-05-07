import numpy as np
import optuna
from config import f as test_function, START_POINT
from lib import *

optuna.logging.set_verbosity(optuna.logging.CRITICAL)

# 1. Градиентный спуск с фиксированным шагом

def objective_fixed(trial: optuna.trial.Trial) -> float:
    step = trial.suggest_float("step_fixed", 1e-4, 1.0, log=True)
    tolerance = trial.suggest_float("tol_fixed", 1e-6, 1e-2, log=True)
    max_iter = trial.suggest_int("max_iter_fixed", 100, 5000)
    point = gradient_descent_fixed(
        test_function,
        START_POINT.copy(),
        step,
        tolerance,
        max_iter
    )
    trial.set_user_attr("opt_point", point.tolist())
    return test_function(point)

# 2. Градиентный спуск с уменьшающимся шагом

def objective_decreasing(trial: optuna.trial.Trial) -> float:
    step = trial.suggest_float("step_decr", 1e-4, 1.0, log=True)
    tolerance = trial.suggest_float("tol_decr", 1e-6, 1e-2, log=True)
    max_iter = trial.suggest_int("max_iter_decr", 100, 5000)
    point = gradient_descent_decreasing(
        test_function,
        START_POINT.copy(),
        step,
        tolerance,
        max_iter
    )
    trial.set_user_attr("opt_point", point.tolist())
    return test_function(point)

# 3. Градиентный спуск с условием Армихо

def objective_armijo(trial: optuna.trial.Trial) -> float:
    step = trial.suggest_float("step_armijo", 1e-3, 1.0, log=True)
    tolerance = trial.suggest_float("tol_armijo", 1e-6, 1e-2, log=True)
    max_iter = trial.suggest_int("max_iter_armijo", 100, 5000)
    tau = trial.suggest_float("tau_armijo", 0.1, 0.9)
    c = trial.suggest_float("c_armijo", 1e-4, 1e-1, log=True)
    # Заметим, что функция backtracking_armijo принимает c, но наша gradient_descent_armijo
    # не использует c напрямую — при необходимости передайте его
    point = gradient_descent_armijo(
        test_function,
        START_POINT.copy(),
        step,
        tolerance,
        max_iter,
        tau
    )
    trial.set_user_attr("opt_point", point.tolist())
    return test_function(point)

# 4. Градиентный спуск с условием Вульфа

def objective_wolfe(trial: optuna.trial.Trial) -> float:
    step = trial.suggest_float("step_wolfe", 1e-3, 1.0, log=True)
    tolerance = trial.suggest_float("tol_wolfe", 1e-6, 1e-2, log=True)
    max_iter = trial.suggest_int("max_iter_wolfe", 100, 5000)
    c1 = trial.suggest_float("c1_wolfe", 1e-4, 0.5)
    c2 = trial.suggest_float("c2_wolfe", c1 + 1e-4, 0.9)
    tau = trial.suggest_float("tau_wolfe", 0.1, 0.9)
    point = gradient_descent_wolfe(
        test_function,
        START_POINT.copy(),
        step,
        tolerance,
        max_iter,
        c1,
        c2,
        tau
    )
    trial.set_user_attr("opt_point", point.tolist())
    return test_function(point)

# 5. Градиентный спуск с золотым сечением

def objective_golden(trial: optuna.trial.Trial) -> float:
    tolerance = trial.suggest_float("tol_golden", 1e-6, 1e-2)
    max_iter = trial.suggest_int("max_iter_golden", 100, 5000)
    point = gradient_descent_golden(
        test_function,
        START_POINT.copy(),
        tolerance,
        max_iter
    )
    trial.set_user_attr("opt_point", point.tolist())
    return test_function(point)

# 6. Градиентный спуск с методом дихотомии

def objective_dichotomy(trial: optuna.trial.Trial) -> float:
    tolerance = trial.suggest_float("tol_dichotomy", 1e-6, 1e-2)
    max_iter = trial.suggest_int("max_iter_dichotomy", 100, 5000)
    point = gradient_descent_dichotomy(
        test_function,
        START_POINT.copy(),
        tolerance,
        max_iter
    )
    trial.set_user_attr("opt_point", point.tolist())
    return test_function(point)

def objective_newton_method_with_golden(trial: optuna.trial.Trial) -> float:
    tolerance = trial.suggest_float("tol_newton_golden", 1e-6, 1e-2)
    max_iter = trial.suggest_int("max_iter_newton_golden", 100, 5000)
    point = newton_method_with_golden(
        test_function,
        START_POINT.copy(),
        tolerance,
        max_iter
    )
    trial.set_user_attr("opt_point", point.tolist())
    return test_function(point)

def objective_newton_method_with_armijo(trial: optuna.trial.Trial) -> float:
    step = trial.suggest_float("step_newton_armijo", 1e-3, 1.0, log=True)
    tolerance = trial.suggest_float("tol_newton_armijo", 1e-6, 1e-2, log=True)
    max_iter = trial.suggest_int("max_iter_newton_armijo", 100, 5000)
    tau = trial.suggest_float("tau_newton_armijo", 0.1, 0.9)
    c = trial.suggest_float("c_newton_armijo", 1e-4, 1e-1, log=True)
    point = newton_method_with_armijo(
        test_function,
        START_POINT.copy(),
        step,
        tolerance,
        max_iter,
        c,
        tau,
    )
    trial.set_user_attr("opt_point", point.tolist())
    return test_function(point)

def objective_newton_method_with_wolfe(trial: optuna.trial.Trial) -> float:
    step = trial.suggest_float("step_newton_wolfe", 1e-3, 1.0, log=True)
    tolerance = trial.suggest_float("tol_newton_wolfe", 1e-6, 1e-2, log=True)
    max_iter = trial.suggest_int("max_iter_newton_wolfe", 100, 5000)
    c1 = trial.suggest_float("c1_newton_wolfe", 1e-4, 0.5)
    c2 = trial.suggest_float("c2_newton_wolfe", c1 + 1e-4, 0.9)
    tau = trial.suggest_float("tau_newton_wolfe", 0.1, 0.9)
    point = newton_method_with_wolfe(
        test_function,
        START_POINT.copy(),
        step,
        tolerance,
        max_iter,
        c1,
        c2,
        tau
    )
    trial.set_user_attr("opt_point", point.tolist())
    return test_function(point)


def find_best_params():
    methods = {
        "fixed": objective_fixed,
        "decreasing": objective_decreasing,
        "armijo": objective_armijo,
        "wolfe": objective_wolfe,
        "golden": objective_golden,
        "dichotomy": objective_dichotomy,
        "newton_golden": objective_newton_method_with_golden,
        "newton_armijo": objective_newton_method_with_armijo,
        "newton_wolfe": objective_newton_method_with_wolfe
    }

    for name, objective in methods.items():
        study = optuna.create_study(direction="minimize", study_name=name)
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        print(f"=== {name.upper()} ===")
        print("Best params:", study.best_params)
        print("Best objective value:", study.best_value)
        print("Best point:", study.best_trial.user_attrs["opt_point"])
        print()
