import time
import tracemalloc
import numpy as np
from sklearn.metrics import mean_squared_error
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler

import ssl
import os
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''

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

def load_dataset(id: int) -> tuple[np.ndarray, np.ndarray]:
    wine = fetch_ucirepo(id=id)
    X_raw = wine.data.features
    y_raw = wine.data.targets

    X = X_raw.values
    y = y_raw.values.ravel()

    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    X = np.c_[np.ones((len(X),1)), X]

    return X, y


def make_anisotropic_regression(
    n_samples: int = 1000,
    noise: float = 5.0,
    random_state: int = 42
):
    """
    Генерирует X ~ N(0, diag([1, 100])), w_true = [3, -1], y = Xw + noise.
    Это даст сильно «растянутую» по одной оси поверхность потерь.
    """
    rng = np.random.default_rng(random_state)
    # Две фичи с разными дисперсиями
    cov = np.diag([1.0, 100.0])
    X = rng.multivariate_normal(mean=[0,0], cov=cov, size=n_samples)
    
    # Истинные веса
    w_true = np.array([3.0, -1.0])
    y = X @ w_true + noise * rng.standard_normal(n_samples)
    
    # Добавим столбец единиц для bias и стандартизируем признаки
    X = StandardScaler().fit_transform(X)
    X = np.c_[np.ones((n_samples,1)), X]
    
    return X, y


def step_schedule(name: str):
  match name:
    case 'constant':
        s = lambda epoch, decay_rate, step: step
    case 'linear':
        s = lambda epoch, decay_rate, step: step * (0.5 ** (epoch // 10))
    case 'exponential':
        s = lambda epoch, decay_rate, step: step * np.exp(-decay_rate * epoch)
    case 'inverse_time':
        s = lambda epoch, decay_rate, step: step / (1 + decay_rate * epoch)
  return s


def regularization_gradient(
        w: np.ndarray, 
        reg_type: str, 
        reg_lambda: float, 
        l1_ratio: float) -> np.ndarray:
    if reg_type == 'l2':
        return 2 * reg_lambda * w
    elif reg_type == 'l1':
       return reg_lambda * np.sign(w)
    elif reg_type == 'elastic':
        return reg_lambda * ((1 - l1_ratio) * w + l1_ratio * np.sign(w))
    else:
        return 0


def sgd_nesterov(
        tracker: Tracker,
        x: np.ndarray,
        y: np.ndarray,
        eras: int,
        batch_size: int,
        step_name: str,
        step_initial: float,
        decay_rate: float,
        beta: float,
) -> np.ndarray:
    w = np.random.rand(len(x[0]))
    v = np.zeros_like(w)
    step_update = step_schedule(step_name)
    step = step_initial

    for era in range(eras):
        indices = np.random.permutation(len(x))
        X_shuffled = x[indices]
        Y_shuffled = y[indices]

        flops = 0
        for i in range(0, len(x), batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            Y_batch = Y_shuffled[i:i + batch_size]

            predictions = np.dot(X_batch, w - beta * v)
            gradient_average = (2 / batch_size) * np.dot(X_batch.T, (predictions - Y_batch))

            v = beta * v - step * gradient_average
            w += v

            flops += count_flops(batch_size, len(x[0]))

        step = step_update(era, decay_rate, step_initial)
        mse = mean_squared_error(y, np.dot(x, w))
        tracker.track(mse, step, flops)

    return w    

def sgd_momentum(
        tracker: Tracker,
        x: np.ndarray,
        y: np.ndarray,
        eras: int,
        batch_size: int,
        step_name: str,
        step_initial: float,
        decay_rate: float,
        beta: float,
) -> np.ndarray:
    w = np.random.rand(len(x[0]))
    v = np.zeros_like(w)
    step_update = step_schedule(step_name)
    step = step_initial

    for era in range(eras):
        indices = np.random.permutation(len(x))
        X_shuffled = x[indices]
        Y_shuffled = y[indices]

        flops = 0
        for i in range(0, len(x), batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            Y_batch = Y_shuffled[i:i + batch_size]

            predictions = np.dot(X_batch, w)
            gradient_average = (2 / batch_size) * np.dot(X_batch.T, (predictions - Y_batch))

            v = beta * v + gradient_average
            w -= step * v

            flops += count_flops(batch_size, len(x[0]))

        step = step_update(era, decay_rate, step_initial)
        mse = mean_squared_error(y, np.dot(x, w))
        tracker.track(mse, step, flops)

    return w



def sgd(
        tracker: Tracker, 
        x: np.ndarray, 
        y: np.ndarray, 
        eras: int, 
        batch_size: int, 
        step_name: str, 
        step_initial: float, 
        decay_rate: float, 
        reg_type: str,
        reg_lambda: float,
        l1_ratio: float,
        eps: float) -> np.ndarray:
    
    w = np.random.rand(len(x[0]))
    step_update = step_schedule(step_name)
    step = step_initial

    for era in range(eras):
        indices = np.random.permutation(len(x))
        X_shuffled = x[indices]
        Y_shuffled = y[indices]

        flops = 0
        for i in range(0, len(x), batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            Y_batch = Y_shuffled[i:i + batch_size]

            predictions = np.dot(X_batch, w)
            gradient_average = (2 / batch_size) * np.dot(X_batch.T, (predictions - Y_batch))

            gradient_average += regularization_gradient(w, reg_type, reg_lambda, l1_ratio)

            w -= step * gradient_average

            flops += count_flops(batch_size, len(x[0]))

        step = step_update(era, decay_rate, step_initial)
        mse = mean_squared_error(y, np.dot(x, w))
        tracker.track(mse, step, flops)

    return w


def count_flops(batch_size: int, n_features: int) -> int:
        """
        Считает число арифметических операций (умножения + сложения)
        для одного мини-батча в линейной регрессии:
        B - размер батча, D - число признаков
        Xb - батч признаков, yb - батч ответов, w - веса модели
        Xb - размера (B, D), yb - (B,), w - (D,)
          1) Число умножений: B*D + число сложений: B*(D-1)
             Xb @ w      → B*D mult + B*(D-1) add
          2) preds - yb   → B sub
          3) Xb.T @ res  → D*B mult + D*(B-1) add
          4) scale grad  → D mult
          5) lr * grad   → D mult
          6) w -= grad   → D sub
        """
        B, D = batch_size, n_features
        mults = B*D           \
              + D*B           \
              + D             \
              + D
        adds  = B*(D-1)       \
              + D*(B-1)
        subs  = B             \
              + D
        return mults + adds + subs

def run_experiment(
        tracker: Tracker, 
        x: np.ndarray, 
        y: np.ndarray, 
        eras: int, 
        batch_size: int, 
        step_name: str, 
        step_initial: float, 
        decay_rate: float, 
        reg_type: str,
        reg_lambda: float,
        l1_ratio: float,
        eps: float):

    tracemalloc.start()
    t0 = time.perf_counter()

    w = sgd(
        tracker=tracker,
        x=x, y=y,
        eras=eras,
        batch_size=batch_size,
        step_name=step_name,
        step_initial=step_initial,
        decay_rate=decay_rate,
        eps=eps,
        reg_type=reg_type,
        reg_lambda=reg_lambda,
        l1_ratio=l1_ratio
    )

    duration = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mse_test = mean_squared_error(y, x @ w)

    return f"""
        'batch_size': {batch_size},
        'mse_test': {mse_test},
        'time_s': {duration},
        'peak_mem_mb': {peak / 1e6},
        'flops_est': {tracker.total_flops},
    """
