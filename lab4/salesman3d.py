import numpy as np
import matplotlib.pyplot as plt
from lib import temperature_schedule  # ваша функция

# --- генерируем 30 городов в кубе [0,1]^3
np.random.seed(0)
cities = np.random.rand(30, 3)

# --- функция длины тура по перестановке perm, теперь в 3D
def tour_length(perm):
    perm_cycle = np.concatenate([perm, perm[:1]])
    return sum(
        np.linalg.norm(cities[perm_cycle[i]] - cities[perm_cycle[i+1]])
        for i in range(len(perm))
    )

# --- SA для TSP
def simulated_annealing_tsp(obj_func, perm0, T0, t_schedule, n_iter):
    x, f_x   = perm0.copy(), obj_func(perm0)
    best, bf = x.copy(), f_x
    T        = T0
    history  = [f_x]
    for i in range(1, n_iter):
        i1, i2 = np.random.choice(len(x), 2, replace=False)
        x_new  = x.copy()
        x_new[i1], x_new[i2] = x_new[i2], x_new[i1]
        f_new  = obj_func(x_new)
        delta  = f_new - f_x
        if (delta < 0) or (np.random.rand() < np.exp(-delta/T)):
            x, f_x = x_new, f_new
            if f_x < bf:
                best, bf = x.copy(), f_x
        T = t_schedule(T)
        history.append(bf)
    return best, bf, history

# --- параметры SA
T0      = 1.0
n_iter  = 500
t_sched = temperature_schedule('exponential')

# --- запуск
perm0 = np.arange(len(cities))
best_perm, best_len, hist = simulated_annealing_tsp(
    tour_length, perm0, T0, t_sched, n_iter
)

print("Best length:", best_len)
print("Best tour:", best_perm)

# --- график сходимости (2D) ---
plt.figure(figsize=(6,3))
plt.plot(hist)
plt.xlabel("Iteration")
plt.ylabel("Tour length")
plt.title("SA on TSP (30 cities, 3D)")
plt.tight_layout()
plt.savefig("salesman3d_30_iters.png")



# --- 3D-визуализация маршрута ---
from mpl_toolkits.mplot3d import Axes3D  # регистрирует '3d' projection, можно неявно

tour_cycle = np.concatenate([best_perm, best_perm[:1]])
# coords      = cities[tour_cycle]

fig = plt.figure(figsize=(6,6))
ax  = fig.add_subplot(111, projection='3d')

# 1) Отрисовка точек с глубинным затенением
ax.scatter(
    cities[:,0], cities[:,1], cities[:,2],
    color='black', s=30, depthshade=True
)

# 2) Соединяем их в найденном порядке
coords = cities[tour_cycle]
ax.plot(
    coords[:,0], coords[:,1], coords[:,2],
    '-o', color='blue', markersize=5, linewidth=1
)

# 3) Подписи осей
ax.set_xlabel('X', labelpad=8)
ax.set_ylabel('Y', labelpad=8)
ax.set_zlabel('Z', labelpad=8)

# 4) Единый масштаб по осям
ax.set_box_aspect([1,1,1])

# 5) Сетка для ориентира
ax.grid(True, linestyle='--', linewidth=0.5)

# 6) Удобный ракурс
ax.view_init(elev=25, azim=45)

# 7) Заголовок и лишнее выключаем
ax.set_title(f"Best TSP Tour in 3D\nlength={best_len:.3f}")
# ax.set_axis_off()

plt.tight_layout()
plt.savefig("salesman3d_30_way.png")