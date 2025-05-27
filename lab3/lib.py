import time
import tracemalloc
import numpy as np
from sklearn.metrics import mean_squared_error
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler


class Tracker:
    def __init__(self) -> None:
        self.__history = {'errors':[], 'steps':[]}
        self.__total_flops = 0
        self.__eras = 0

    def track(self, error: float, step: float, flops_epoch: int) -> None:
        self.__history['errors'].append(error)
        self.__history['steps'].append(step)

        self.__total_flops += flops_epoch
        self.__eras += 1

    @property
    def history_errors(self) -> list[float]:
        return self.__history["errors"]
    
    @property
    def history_steps(self) -> list[float]:
        return self.__history["steps"]
    
    @property
    def eras(self) -> int:
        return self.__eras


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

        for i in range(0, len(x), batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            Y_batch = Y_shuffled[i:i + batch_size]

            predictions = np.dot(X_batch, w)
            gradient_average = (2 / batch_size) * np.dot(X_batch.T, (predictions - Y_batch))

            gradient_average += regularization_gradient(w, reg_type, reg_lambda, l1_ratio)

            w -= step * gradient_average

        step = step_update(era, decay_rate, step_initial)
        mse = mean_squared_error(y, np.dot(x, w))
        tracker.track(mse, step)

    return w


def count_flops(self, batch_size: int, n_features: int) -> int:
        """
        Считает число арифметических операций (умножения + сложения)
        для одного мини-батча в линейной регрессии:
        B - размер батча, D - число признаков
        Xb - батч признаков, yb - батч ответов, w - веса модели
        Xb - размера (B, D), yb - (B,), w - (D,)
          1) Xb @ w      → B*D mult + B*(D-1) add
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

    # 6) Оценка FLOPs: примерно 2·D·N·E  
    D = x.shape[1]
    N = x.shape[0]
    E = eras
    flops_est = 2 * D * N * E

    return f"""
        'batch_size': {batch_size},
        'mse_test': {mse_test},
        'time_s': {duration},
        'peak_mem_mb': {peak / 1e6},
        'flops_est': {flops_est}
    """


def sgd1(X, Y, ERAS, BATCH_SIZE, STEP_SIZE, EPS):
    w = np.zeros(len(X[0]))
    for era in range(ERAS):

        # gradient = np.zeros_like(w)
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]

        for i in range(0, len(X), BATCH_SIZE):
            X_batch = X_shuffled[i:i + BATCH_SIZE]
            Y_batch = Y_shuffled[i:i + BATCH_SIZE]


            # Y_i = Y_batch[i]
            # X_i = X_batch[i]
            # predictions = np.dot(X_i, w)
            # errors = (predictions - Y_i) ** 2     #   (x_i * W - y_i) ** 2
            # gradient = (2 / BATCH_SIZE) * (predictions - Y_i) * X_i   # 2 * (x_i * W - y_i) * x_i
            # w -= STEP_SIZE * gradient


            predictions = np.dot(X_batch, w)
            # errors = (predictions - Y_batch) ** 2
            gradient = (2 / BATCH_SIZE) * np.dot(X_batch.T, (predictions - Y_batch))  # 2 * (x_i * W - y_i) * x_i
            w -= STEP_SIZE * gradient

        # if np.linalg.norm(gradient) < EPS:
        #     print(f"Early stopping at era {era}, gradient norm is too small.")
        #     break

        # mse = np.mean((X.dot(w) - Y) ** 2)
        # if mse < EPS:
        #     print(f"Early stopping by MSE at epoch {era}, MSE={mse:.2e}")
        #     break
    return w

def test_sgd(w, X, Y):
    print(f"Got: {np.dot(X, w)}, Expected: {Y}") 


def manual_sgd(X, y,
               lr=0.01,
               batch_size=32,
               n_epochs=100,
               reg=None,       # {'type':'L2','alpha':0.1} и т.п.
               schedule=None   # function(epoch, lr0) -> lr_epoch
              ):
    w = np.zeros(X.shape[1])
    history = {'loss':[], 'lr':[]}

    for epoch in range(n_epochs):
        # 1) обновить lr (если есть расписание)
        lr_epoch = schedule(epoch, lr) if schedule else lr

        # 2) перемешать данные
        idx = np.random.permutation(len(y))
        X_shuf, y_shuf = X[idx], y[idx]

        # 3) пройти по батчам
        for start in range(0, len(y), batch_size):
            Xb = X_shuf[start:start+batch_size]
            yb = y_shuf[start:start+batch_size]

            # 4) вычислить градиент MSE
            preds = Xb.dot(w)
            error = preds - yb
            grad = (2 / len(yb)) * Xb.T.dot(error)

            # 5) добавить регуляризацию, если нужна
            if reg:
                if reg['type']=='L2':
                    grad += 2 * reg['alpha'] * w
                elif reg['type']=='L1':
                    grad += reg['alpha'] * np.sign(w)
                elif reg['type']=='Elastic':
                    α = reg['alpha']
                    λ = reg['l1_ratio']
                    grad += α * (λ * np.sign(w) + (1-λ)*2*w)

            # 6) шаг
            w -= lr_epoch * grad

        # 7) записать MSE на всём датасете для графиков
        loss = mean_squared_error(y, X.dot(w))
        history['loss'].append(loss)
        history['lr'].append(lr_epoch)

    return w, history
