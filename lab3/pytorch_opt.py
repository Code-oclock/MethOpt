import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

import config
import lib
import draw

ERAS = 30

def get_data(batch_size=32, noise=3.0, n_samples=1000, n_features=10):
    """Генерируем и подготавливаем DataLoader."""
    # X, y = make_regression(
    #     n_samples=n_samples,
    #     n_features=n_features,
    #     noise=noise,
    #     random_state=42
    # )
    # y = y.reshape(-1, 1)
    # X = StandardScaler().fit_transform(X)
    # Загрузка реального датасета
    X, y = lib.load_dataset(id=config.DATASET_ID)
    y = y.reshape(-1, 1)  # Убедимся, что y двумерный

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), X_t, y_t


def train_sgd(loader, in_features, lr, n_epochs):
    """Тренируем простым SGD (momentum=0)."""
    model = nn.Sequential(nn.Linear(in_features, 1))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0)

    tracker = lib.Tracker()
    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(loader.dataset)
        tracker.track(epoch_loss, 0)
        print(f"[SGD] Epoch {epoch:2d}/{n_epochs}, Loss = {epoch_loss:.4f}")

    draw.draw(tracker, "lib_sgd.png")
    return model

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

    tracker = lib.Tracker()
    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(loader.dataset)
        tracker.track(epoch_loss, 0)
        print(f"[Momentum] Epoch {epoch:2d}/{n_epochs}, Loss = {epoch_loss:.4f}")

    draw.draw(tracker, "lib_sgd_momentum.png")
    return model

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

    tracker = lib.Tracker()
    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(loader.dataset)
        tracker.track(epoch_loss, 0)
        print(f"[Nesterov] Epoch {epoch:2d}/{n_epochs}, Loss = {epoch_loss:.4f}")

    draw.draw(tracker, "lib_sgd_nesterov.png")
    return model

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

    tracker = lib.Tracker()
    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
<<<<<<< HEAD
            epoch_loss = running_loss/len(loader.dataset)
=======

        epoch_loss = running_loss/len(loader.dataset)
>>>>>>> 4e182fb (pictures)
        tracker.track(epoch_loss, 0)
        print(f"[AdaGrad]   Epoch {epoch:2d}/{n_epochs}, Loss = {epoch_loss:.4f}")

    draw.draw(tracker, "lib_sgd_adagrad.png")
    return model

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

    tracker = lib.Tracker()
    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        epoch_loss = running_loss / len(loader.dataset)
        tracker.track(epoch_loss, 0)
        print(f"[RMSProp]   Epoch {epoch:2d}/{n_epochs}, Loss = {epoch_loss:.4f}")

    draw.draw(tracker, "lib_sgd_rmsprop.png")
    return model

def train_adam(loader, in_features, lr, betas, eps, weight_decay, n_epochs):
    """Тренировка с Adam."""
    model = nn.Sequential(nn.Linear(in_features, 1))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay
    )

    tracker = lib.Tracker()
    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(loader.dataset)
        tracker.track(epoch_loss, 0)
        print(f"[Adam]      Epoch {epoch:2d}/{n_epochs}, Loss = {epoch_loss:.4f}")
    
    draw.draw(tracker, "lib_sgd_adam.png")
    return model


def main():
    # 1) Готовим данные
    loader, X_all, y_all = get_data()

    # 2) Тренируем обычный SGD
    print("=== Training with plain SGD ===")
    model_sgd = train_sgd(loader, X_all.shape[1], 0.01, ERAS)

    # 3) Тренируем SGD+Momentum
    print("\n=== Training with SGD + Momentum ===")
    model_mom = train_sgd_momentum(loader, X_all.shape[1], 0.01, 0.9, ERAS)

    # 3) Тренируем SGD+Nesterov
    print("\n=== Training with SGD + Nesterov ===")
    model_nes = train_sgd_nesterov(loader, X_all.shape[1], 0.01, 0.9, ERAS)

    # 4) Тренируем AdaGrad
    print("\n=== Training with AdaGrad ===")
    model_ada = train_adagrad(loader, X_all.shape[1], 0.99, 0, 0, 0.1, ERAS)

    # 5) Тренируем RMSProp
    print("\n=== Training with RMSProp ===")
    model_rms = train_rmsprop(loader, X_all.shape[1], 0.01, 0.8, 1e-6, 0.9, 1e-4, ERAS)

    # 6) Тренируем Adam
    print("\n=== Training with AdamProp ===")
    model_adam = train_adam(loader, X_all.shape[1], 0.3, (0.8, 0.999), 1e-5, 0, ERAS)

    # 4) Финальная проверка (на тех же данных)
    criterion = nn.MSELoss()
    with torch.no_grad():
        loss_sgd = criterion(model_sgd(X_all), y_all).item()
        loss_mom = criterion(model_mom(X_all), y_all).item()
        loss_nes = criterion(model_nes(X_all), y_all).item()
        loss_ada = criterion(model_ada(X_all), y_all).item()
        loss_rms = criterion(model_rms(X_all), y_all).item()
        loss_adam = criterion(model_adam(X_all), y_all).item()
    print(f"\nFinal MSE plain SGD:   {loss_sgd:.4f}")
    print(f"Final MSE Momentum:    {loss_mom:.4f}")
    print(f"Final MSE Nesterov:    {loss_nes:.4f}")
    print(f"Final MSE AdaGrad:     {loss_ada:.4f}")
    print(f"Final MSE RMSProp:     {loss_rms:.4f}")
    print(f"Final MSE Adam:        {loss_adam:.4f}")


if __name__ == "__main__":
    main()
