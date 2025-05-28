import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader
import config, lib, draw
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
optuna.logging.set_verbosity(optuna.logging.WARNING)


ERAS     = 20
FIXED_LR = 0.01
BATCH    = 32

def get_data():
    # Загрузка реального датасета
    X, y = lib.load_dataset(id=config.DATASET_ID)
    y = y.reshape(-1, 1)  # Убедимся, что y двумерный

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=BATCH, shuffle=True), X_t, y_t


def train_sgd(loader, in_features, lr, n_epochs):
    """Тренируем простым SGD (momentum=0)."""
    model = nn.Sequential(nn.Linear(in_features, 1))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0)

    tracker = lib.Tracker("sgd")
    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(loader.dataset)
        tracker.track(epoch_loss, 0, 0)
        # print(f"[SGD] Epoch {epoch:2d}/{n_epochs}, Loss = {epoch_loss:.4f}")

    # draw.draw(tracker, "lib_sgd.png")
    return model, tracker

def train_sgd_momentum(loader, in_features, lr, momentum, n_epochs):
    """Тренируем SGD с momentum"""
    model = nn.Sequential(nn.Linear(in_features, 1))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        nesterov=False
    )

    tracker = lib.Tracker("momentum")
    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(loader.dataset)
        tracker.track(epoch_loss, 0, 0)
        # print(f"[Momentum] Epoch {epoch:2d}/{n_epochs}, Loss = {epoch_loss:.4f}")

    # draw.draw(tracker, "lib_sgd_momentum.png")
    return model, tracker

def train_sgd_nesterov(loader, in_features, lr, momentum, n_epochs):
    """Тренируем SGD с momentum и Nesterov."""
    model = nn.Sequential(nn.Linear(in_features, 1))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        nesterov=True
    )

    tracker = lib.Tracker("nesterov")
    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(loader.dataset)
        tracker.track(epoch_loss, 0, 0)
        # print(f"[Nesterov] Epoch {epoch:2d}/{n_epochs}, Loss = {epoch_loss:.4f}")

    # draw.draw(tracker, "lib_sgd_nesterov.png")
    return model, tracker

def train_adagrad(loader, in_features, lr, lr_decay, weight_decay, init_accumulator, n_epochs):
    """Тренировка с AdaGrad."""
    model = nn.Sequential(nn.Linear(in_features, 1))
    criterion = nn.MSELoss()
    optimizer = optim.Adagrad(
        model.parameters(),
        lr=lr,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        initial_accumulator_value=init_accumulator
    )

    tracker = lib.Tracker("adagrad")
    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            epoch_loss = running_loss/len(loader.dataset)

        epoch_loss = running_loss/len(loader.dataset)
        tracker.track(epoch_loss, 0, 0)
        # print(f"[AdaGrad]   Epoch {epoch:2d}/{n_epochs}, Loss = {epoch_loss:.4f}")

    # draw.draw(tracker, "lib_sgd_adagrad.png")
    return model, tracker

def train_rmsprop(loader, in_features, lr, alpha, eps, momentum, weight_decay, n_epochs):
    model = nn.Sequential(nn.Linear(in_features, 1))
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=lr,
        alpha=alpha,         
        eps=eps,
        momentum=momentum,   
        weight_decay=weight_decay
    )

    tracker = lib.Tracker("rmsprop")
    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        epoch_loss = running_loss / len(loader.dataset)
        tracker.track(epoch_loss, 0, 0)
        # print(f"[RMSProp]   Epoch {epoch:2d}/{n_epochs}, Loss = {epoch_loss:.4f}")

    # draw.draw(tracker, "lib_sgd_rmsprop.png")
    return model, tracker

def train_adam(loader, in_features, lr, beta1, beta2, eps, weight_decay, n_epochs):
    """Тренировка с Adam."""
    model = nn.Sequential(nn.Linear(in_features, 1))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay
    )

    tracker = lib.Tracker("adam")
    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(loader.dataset)
        tracker.track(epoch_loss, 0, 0)
        # print(f"[Adam]      Epoch {epoch:2d}/{n_epochs}, Loss = {epoch_loss:.4f}")
    
    # draw.draw(tracker, "lib_sgd_adam.png")
    return model, tracker



# Словарь: для каждого метода — его функция в lib и пространство поиска
METHODS = {
    "SGD":        {
        "fn":  train_sgd,
        "space": lambda t: {}
    },
    "Momentum":   {
        "fn":  train_sgd_momentum,
        "space": lambda t: {
            "momentum": t.suggest_float("momentum", 0.5, 0.99)
        }
    },
    "Nesterov":   {
        "fn":  train_sgd_nesterov,
        "space": lambda t: {
            "momentum": t.suggest_float("momentum", 0.5, 0.99)
        }
    },
    "AdaGrad":    {
        "fn":  train_adagrad,
        "space": lambda t: {
            "lr_decay":           t.suggest_float("lr_decay", 1e-4, 1e-1),
            "weight_decay":       t.suggest_float("weight_decay", 1e-6, 1e-2),
            "init_accumulator":   t.suggest_float("init_accumulator", 1e-8, 1e-1)
        }
    },
    "RMSProp":    {
        "fn":  train_rmsprop,
        "space": lambda t: {
            "alpha":        t.suggest_float("alpha", 0.7, 0.99),
            "eps":          t.suggest_float("eps", 1e-8, 1e-4),
            "momentum":     t.suggest_float("momentum", 0.0, 0.9),
            "weight_decay": t.suggest_float("weight_decay", 1e-6, 1e-2)
        }
    },
    "Adam":       {
        "fn":  train_adam,
        "space": lambda t: {
            "beta1":          t.suggest_float("beta1", 0.8, 0.99),
            "beta2":          t.suggest_float("beta2", 0.9, 0.999),
            "eps":            t.suggest_float("eps", 1e-8, 1e-4),
            "weight_decay":   t.suggest_float("weight_decay", 1e-6, 1e-2)
        }
    },
}

def optimize_method(name, fn, space, loader, in_features, y_all):
    """Запускаем Optuna для одного метода и возвращаем best_params и best_value."""
    def objective(trial):
        # собираем все параметры: lr, batch_size, rest from space(trial)
        params = space(trial)
        # для единообразия передаем batch_size, lr, n_epochs
        model, tracker = fn(
            loader, in_features,
            lr=FIXED_LR,
            n_epochs=ERAS,
            **params
        )
        # конечный MSE на всём датасете
        mse = nn.MSELoss()(model(loader.dataset.tensors[0]), loader.dataset.tensors[1]).item()
        return mse

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    print(f">>> {name} best MSE: {study.best_value:.4f}, params: {study.best_params}")
    return study.best_params

def main():
    loader, X_all, y_all = get_data()
    in_features = X_all.shape[1]

    trackers = []
    finals   = {}

    # 1) Подбираем для каждого метода
    best_params = {}
    for name, meta in METHODS.items():
        print(f"\n=== Tuning {name} ===")
        bp = optimize_method(name, meta["fn"], meta["space"], loader, in_features, y_all)
        best_params[name] = bp

    # 2) Финальный запуск всех с best_params
    for name, meta in METHODS.items():
        print(f"\n=== Final training {name} ===")
        params = best_params[name]
        # обращаем внимание: train_* принимают разные аргументы,
        # но у всех есть lr, n_epochs, batch_size и свои спецы
        model, tracker = meta["fn"](
            loader, in_features,
            **{"lr": FIXED_LR, "n_epochs": ERAS, **params}
        )
        trackers.append(tracker)
        finals[name] = nn.MSELoss()(model(X_all), y_all).item()
        print(f"{name} final MSE = {finals[name]:.4f}")

    # 3) Рисуем все вместе
    draw.draw_more(trackers)

    # 4) Таблица итогов
    print("\n=== Summary ===")
    print(f"{'Method':>10s} | {'Final MSE':>10s}")
    print("-"*25)
    for name in METHODS:
        print(f"{name:>10s} | {finals[name]:10.4f}")

if __name__ == "__main__":
    main()