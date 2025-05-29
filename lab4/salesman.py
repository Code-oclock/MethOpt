import numpy as np
import matplotlib.pyplot as plt
from lib import temperature_schedule  # ваша функция

# --- генерируем 20 городов в квадрате [0,1]^2
np.random.seed(0)
cities = np.random.rand(30,2)

# --- функция длины тура по перестановке perm
def tour_length(perm):
    perm_cycle = np.concatenate([perm, perm[:1]])  # замкнуть в цикл
    return sum(
        np.linalg.norm(cities[perm_cycle[i]] - cities[perm_cycle[i+1]])
        for i in range(len(perm))
    )

# --- SA для TSP (см. выше)
def simulated_annealing_tsp(obj_func, perm0, T0, t_schedule, n_iter):
    x, f_x     = perm0.copy(), obj_func(perm0)
    best, bf   = x.copy(), f_x
    T          = T0
    history    = [f_x]
    for i in range(1, n_iter):
        # swap-neighbor
        i1, i2 = np.random.choice(len(x), 2, replace=False)
        x_new  = x.copy()
        x_new[i1], x_new[i2] = x_new[i2], x_new[i1]
        f_new   = obj_func(x_new)
        delta   = f_new - f_x
        if (delta < 0) or (np.random.rand() < np.exp(-delta/T)):
            x, f_x = x_new, f_new
            if f_x < bf:
                best, bf = x.copy(), f_x
        T = t_schedule(T)
        history.append(bf)
    return best, bf, history

# --- параметры SA
T0        = 1.0
n_iter    = 500
t_sched   = temperature_schedule('exponential')

# --- запуск
perm0 = np.arange(len(cities))
best_perm, best_len, hist = simulated_annealing_tsp(
    tour_length, perm0, T0, t_sched, n_iter
)

print("Best length:", best_len)
print("Best tour:", best_perm)

plt.plot(hist)
plt.xlabel("Iteration")
plt.ylabel("Tour length")
plt.title("SA on TSP (20 cities)")
plt.savefig("salesman_iters_30.png")


# замыкаем цикл
tour_cycle = np.concatenate([best_perm, best_perm[:1]])

plt.figure(figsize=(6,6))
# рисуем города
plt.scatter(cities[:,0], cities[:,1], color='black')
for idx, (x,y) in enumerate(cities):
    plt.text(x, y, str(idx), color='red', fontsize=12)

# соединяем в найденном порядке
xs = cities[tour_cycle,0]
ys = cities[tour_cycle,1]
plt.plot(xs, ys, '-o', color='blue')

plt.title(f"Best TSP Tour, length={best_len:.3f}")
plt.axis('off')
plt.savefig("salesman_way_30.png")