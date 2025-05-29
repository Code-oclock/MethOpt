import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# Параметры алгоритма отжига
n_iter = 200
T0 = 25.0
T_end = 0.1
alpha = (T_end / T0)**(1/(n_iter - 1))

# Целевая функция
def f(x):
    return np.sin(5*x) + np.sin(2*x) + 0.1 * x**2

# Начальное состояние
x = np.random.uniform(-10, 10)
E = f(x)

# История переменных
xs = []
Ts = []

# Основной цикл SA
for i in range(n_iter):
    T = T0 * (alpha**i)
    Ts.append(T)
    xs.append(x)
    
    x_new = x + np.random.normal(scale=1.0)
    E_new = f(x_new)
    delta = E_new - E
    if delta < 0 or np.random.rand() < np.exp(-delta / T):
        x, E = x_new, E_new

# Подготовка папки и списка кадров
out_dir = './sa_frames'
os.makedirs(out_dir, exist_ok=True)
filenames = []

# Предвычисляем значения функции для графика
x_vals = np.linspace(-10, 10, 1000)
y_vals = f(x_vals)
y_min, y_max = y_vals.min(), y_vals.max()

for i, xi in enumerate(xs):
    fig, ax = plt.subplots(figsize=(6, 3), dpi=80)
    ax.plot(x_vals, y_vals, linewidth=1)
    ax.axvline(x=xi, color='red', linewidth=2)
    ax.set_xlim(-10, 10)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Temperature: {Ts[i]:.2f}', loc='left', fontsize=10)
    plt.tight_layout(pad=0)
    
    frame_path = os.path.join(out_dir, f'frame_{i:03d}.png')
    fig.savefig(frame_path)
    plt.close(fig)
    filenames.append(frame_path)

# Сборка GIF
frames = [imageio.v2.imread(fn) for fn in filenames]
gif_path = './sa_function.gif'
imageio.mimsave(gif_path, frames, duration=0.05)

print("Гифка готова:", gif_path)
