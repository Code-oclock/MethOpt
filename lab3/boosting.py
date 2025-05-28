import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# ===============================
# 1) Генерация и подготовка данных
# ===============================
X, y = make_regression(
    n_samples=1000,      # 1000 примеров
    n_features=10,       # 10 признаков
    noise=20.0,          # гауссов шум σ=20
    random_state=42
)
# Стандартизация признаков для устойчивости
X = StandardScaler().fit_transform(X)

# Разделение на train и test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ===============================
# 2) Обучение градиентного бустинга
# ===============================
gbr = GradientBoostingRegressor(
    n_estimators=100,    # число деревьев
    learning_rate=0.1,   # размер шага
    max_depth=3,         # глубина деревьев
    random_state=42
)
gbr.fit(X_train, y_train)

# ===============================
# 3) Сбор MSE на этапах (staged_predict)
# ===============================
train_mse = []
test_mse  = []
for y_pred_train in gbr.staged_predict(X_train):
    train_mse.append(mean_squared_error(y_train, y_pred_train))
for y_pred_test in gbr.staged_predict(X_test):
    test_mse.append(mean_squared_error(y_test, y_pred_test))

# ===============================
# 4) Визуализация сходимости
# ===============================
plt.figure(figsize=(10, 6))
iterations = np.arange(1, len(train_mse) + 1)
plt.plot(iterations, train_mse, label='Train MSE', linewidth=2)
plt.plot(iterations, test_mse,  label='Test MSE',  linewidth=2)
plt.xlabel('Boosting Iteration', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Gradient Boosting: Train vs Test MSE Convergence', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('gradient_boosting_convergence.png')
