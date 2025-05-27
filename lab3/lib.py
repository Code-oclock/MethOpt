import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler


def step_schedule(name: str):
  match name:
    case 'constant':
        s = lambda epoch, decay_rate, step: step
    case 'linear':
        s = lambda epoch, decay_rate, step: step * (1 - decay_rate * epoch)
    case 'exponential':
        s = lambda epoch, decay_rate, step: step * (decay_rate ** epoch)
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


def sgd(x, y, eras, batch_size, step_name, step, decay_rate, EPS) -> np.ndarray:
    w = np.zeros(len(x[0]))
    for era in range(eras):
        indices = np.random.permutation(len(x))
        X_shuffled = x[indices]
        Y_shuffled = y[indices]

        for i in range(0, len(x), batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            Y_batch = Y_shuffled[i:i + batch_size]

            predictions = np.dot(X_batch, w)
            gradient_avarage = (2 / batch_size) * np.dot(X_batch.T, (predictions - Y_batch))
            w -= step_schedule(step_name)(era, decay_rate, step) * gradient_avarage

    return w



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
