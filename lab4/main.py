import numpy as np
from lib import simulated_annealing, genetic_algorithm
import matplotlib.pyplot as plt

def rosenbrock(x):
    return sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rastrigin(x):
    A = 10
    return A * x.size + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def run_experiment(obj, bounds, label):
    results = []
    for _ in range(20):
        x0 = np.random.uniform(bounds[:,0], bounds[:,1])
        _, f_best, _ = simulated_annealing(
            obj, x0,
            T0=5.0, alpha=0.95,
            n_iter=300, bounds=bounds,
            step_size=0.5
        )
        results.append(f_best)
    print(f"{label}: mean={np.mean(results):.3e}, std={np.std(results):.3e}")
    return results


if __name__ == "__main__":
    bounds_ros = np.array([[-2, 2]] * 2)
    bounds_ras = np.array([[-5.12, 5.12]] * 2)
    res_ros = run_experiment(rosenbrock, bounds_ros, "Rosenbrock")
    res_ras = run_experiment(rastrigin, bounds_ras, "Rastrigin")

    plt.boxplot([res_ros, res_ras], labels=["Rosen.", "Rastr."])
    plt.yscale("log")
    plt.ylabel("best f(x)")
    plt.title("Сравнение качества SA на разных функциях")
    plt.show()