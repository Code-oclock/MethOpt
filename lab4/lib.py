import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# Параметры
n_iter = 200
T0 = 25.0
T_end = 0.1
alpha = (T_end / T0)**(1/(n_iter - 1))
lr = 0.1          # шаг градиентного спуска
eps = 1e-3        # для численного градиента

# Комплексная функция с множеством локальных минимумов
def f(x):
    return np.sin(5*x) + np.sin(2*x) + 0.1 * x**2

# Численный градиент
def grad(x):
    return (f(x + eps) - f(x - eps)) / (2 * eps)

# Общая инициация
x0 = np.random.uniform(-5, 5)

# Истории для SA и GD
xs_sa = []
Ts_sa = []
xs_gd = []

# Начальные состояния
x_sa, x_gd = x0, x0
f_sa = f(x_sa)

# Запуск алгоритмов
for i in range(n_iter):
    # SA
    T = T0 * (alpha**i)
    Ts_sa.append(T)
    xs_sa.append(x_sa)
    
    x_new = x_sa + np.random.normal(scale=1.0)
    if f(x_new) < f_sa or np.random.rand() < np.exp(-(f(x_new) - f_sa) / T):
        x_sa = x_new
        f_sa = f(x_sa)
    
    # GD
    grad_val = (f(x_gd + eps) - f(x_gd - eps)) / (2 * eps)
    x_gd = x_gd - lr * grad_val
    xs_gd.append(x_gd)

# Подготовка кадров
out_dir = '/mnt/data/sa_vs_gd_frames'
os.makedirs(out_dir, exist_ok=True)
filenames = []

# Предвычисление графика функции
x_vals = np.linspace(-5, 5, 2000)
y_vals = f(x_vals)
y_min, y_max = y_vals.min(), y_vals.max()

for i in range(n_iter):
    fig, ax = plt.subplots(figsize=(6, 3), dpi=80)
    ax.plot(x_vals, y_vals, linewidth=1, color='gray')
    ax.axvline(xs_sa[i], color='red', linewidth=2, label='SA')
    ax.scatter(xs_gd[i], f(xs_gd[i]), color='blue', s=30, label='GD')
    ax.set_xlim(-5, 5)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Temperature: {Ts_sa[i]:.2f}   Iter: {i}', loc='left', fontsize=10)
    if i == 0:
        ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout(pad=0)
    
    frame_path = os.path.join(out_dir, f'frame_{i:03d}.png')
    fig.savefig(frame_path)
    plt.close(fig)
    filenames.append(frame_path)

# Сборка GIF
frames = [imageio.v2.imread(fn) for fn in filenames]
gif_path = '/mnt/data/compare_sa_gd.gif'
imageio.mimsave(gif_path, frames, duration=0.05)

print("GIF saved to", gif_path)
