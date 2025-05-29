import os
import imageio
from matplotlib import pyplot as plt
import numpy as np
import numpy as np
import optuna
from genetic import genetic_algorithm
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Реализация вещественно-кодированного генетического алгоритма с рулеточным отбором
# для функции f(x,y)=x^2+y^2

def fitness_from_loss(loss, eps=(1 - 0.5 ** 50) / (1 - 0.5) + 1):
    """
    Преобразует значение loss в fitness: обратно пропорционально loss.
    Добавляем eps, чтобы избежать деления на ноль.
    """
    return 1.0 / (loss + eps)


def evaluate_population(pop, func):
    """
    Вычисляет loss и fitness для каждого индивида в популяции.
    Возвращает два массива: losses, fitnesses.
    """
    losses = np.array([func(ind) for ind in pop])
    fitnesses = np.array([fitness_from_loss(l) for l in losses])
    return losses, fitnesses


def roulette_selection(pop, fitnesses):
    """
    Рулеточный отбор: выбирает одного родителя пропорционально fitness.
    Если все fitness=0, выбирает случайный.
    """
    total_fit = np.sum(fitnesses)
    if total_fit <= 0:
        # все особи равны по fitness или равны нулю
        idx = np.random.randint(len(pop))
    else:
        probs = fitnesses / total_fit
        idx = np.random.choice(len(pop), p=probs)
    return pop[idx]


def arithmetic_crossover(parent1, parent2):
    """
    Арифметическое скрещивание двух родителей.
    """
    alpha = np.random.rand()
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2
    return child1, child2


def gaussian_mutation(individual, gene_mut_prob=0.1, sigma=0.1, bounds=None):
    """
    Простая векторизованная мутация:
    - Для каждого гена с вероятностью gene_mut_prob добавляем гауссов шум.
    - Если заданы bounds, ограничиваем мутант в пределах.
    """
    # Генерируем шум и маску мутации для всех генов сразу
    noise = np.random.normal(scale=sigma, size=individual.shape)
    mask = np.random.rand(*individual.shape) < gene_mut_prob
    mutant = individual + noise * mask
    # Применяем границы, если нужно
    if bounds is not None:
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        mutant = np.clip(mutant, lower, upper)
    return mutant


def genetic_algorithm(
    func,
    bounds,
    pop_size=20,
    generations=50,
    crossover_prob=0.8,
    mutation_prob=0.1,
    mutation_sigma=0.1,
    elitism_size=2
):
    """
    Основная функция генетического алгоритма.
    func        — целевая функция потерь (loss), минимизируемая.
    bounds      — список пар (min, max) для каждой переменной.
    pop_size    — размер популяции родителей.
    generations — число поколений.
    crossover_prob — вероятность применения скрещивания.
    mutation_prob  — вероятность мутации гена.
    mutation_sigma — стандартное отклонение гауссова шума.
    elitism_size    — число лучших сохраняемых родителей.
    """
    dim = len(bounds)
    # Инициализация популяции: создаём pop_size индивидов
    pop = np.array([
        [np.random.uniform(bounds[j][0], bounds[j][1]) for j in range(dim)]
        for _ in range(pop_size)
    ])

    best_history = []
    history = []
    for gen in range(generations):
        history.append(pop.copy())
        losses, fitnesses = evaluate_population(pop, func)
        # Сохраняем лучший
        best_idx = np.argmin(losses)
        best_history.append((gen, pop[best_idx], losses[best_idx]))
        
        # Элитарность: сохраняем лучших
        elite_indices = np.argsort(losses)[:elitism_size]
        new_pop = [pop[i] for i in elite_indices]

        # Создание потомков
        while len(new_pop) < pop_size:
            # Отбор родителей через рулетку
            p1 = roulette_selection(pop, fitnesses)
            p2 = roulette_selection(pop, fitnesses)
            # Скрещивание
            if np.random.rand() < crossover_prob:
                c1, c2 = arithmetic_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            # Мутация
            c1 = gaussian_mutation(c1, gene_mut_prob=mutation_prob, sigma=mutation_sigma, bounds=bounds)
            c2 = gaussian_mutation(c2, gene_mut_prob=mutation_prob, sigma=mutation_sigma, bounds=bounds)
            new_pop.extend([c1, c2])

        # Обрезаем до нужного размера
        pop = np.array(new_pop[:pop_size])

    # Финальный вывод лучшего решения
    best_gen, best_ind, best_loss = min(best_history, key=lambda x: x[2])
    return {
        'best_generation': best_gen,
        'best_individual': best_ind,
        'best_loss': best_loss,
        'best_history': best_history,
        'history': history
    }


# ----------------------------------------
# 1. Определяем тестовые функции
# ----------------------------------------

def rosenbrock(v):
    x, y = v[0], v[1]
    # Стандартная форма: f(x,y) = (a - x)^2 + b*(y - x^2)^2, a=1, b=100
    return (1 - x)**2 + 100 * (y - x**2)**2

def rastrigin(v):
    x, y = v[0], v[1]
    # f(x,y) = 10*2 + x^2 - 10*cos(2*pi*x) + y^2 - 10*cos(2*pi*y)
    return 20 + x**2 - 10*np.cos(2*np.pi*x) + y**2 - 10*np.cos(2*np.pi*y)

# ----------------------------------------
# 2. Генерация синтетического регрессионного датасета
# ----------------------------------------

def generate_regression_loss(n_samples=200, noise_sigma=0.5, seed=42):
    """
    Генерирует X и y для простой линейной регрессии с двумя признаками:
    y = theta0 + theta1*x1 + theta2*x2 + шум
    Возвращает функцию потерь MSE в виде reg_loss(v), где
    v = [theta0, theta1, theta2].
    """
    np.random.seed(seed)
    X = np.random.uniform(-5, 5, size=(n_samples, 2))
    theta_true = np.array([1.5, -2.0, 0.5])
    y = theta_true[0] + X.dot(theta_true[1:]) + np.random.normal(scale=noise_sigma, size=n_samples)

    def reg_loss(v):
        pred = v[0] + X.dot(v[1:])
        return np.mean((y - pred)**2)

    return reg_loss


# ----------------------------------------
# 3. Функции-объективы для Optuna
# ----------------------------------------

def make_objective(func, bounds):
    """
    Возвращает функцию objective(trial), минимизирующую func
    с помощью genetic_algorithm и Optuna.
    """
    def objective(trial):
        # Предлагаем гиперпараметры
        pop_size = trial.suggest_int('pop_size', 10, 100)
        generations = trial.suggest_int('generations', 50, 500)
        crossover_prob = trial.suggest_float('crossover_prob', 0.5, 1.0)
        mutation_prob = trial.suggest_float('mutation_prob', 0.0, 0.5)
        mutation_sigma = trial.suggest_float('mutation_sigma', 0.01, 2.0, log=True)
        elitism_size = trial.suggest_int('elitism_size', 1, max(1, pop_size // 5))

        result = genetic_algorithm(
            func=func,
            bounds=bounds,
            pop_size=pop_size,
            generations=generations,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            mutation_sigma=mutation_sigma,
            elitism_size=elitism_size
        )
        return result['best_loss']

    return objective


# show_progress_bar=True
# ----------------------------------------
# 4. Основной блок: оптимизация гиперпараметров
# ----------------------------------------
if __name__ == "__main__":
    n_trials = 50

    # 4.1 Подбираем гиперпараметры для Розенброка
    study_ros = optuna.create_study(direction='minimize')
    study_ros.optimize(
        make_objective(rosenbrock, bounds=[(-5,5), (-5,5)]),
        n_trials=n_trials,
        show_progress_bar=True
    )
    print("Лучшие параметры для Rosenbrock:", study_ros.best_params)
    print("Лучший loss:", study_ros.best_value)

    # 4.2 Для Rastrigin
    study_ras = optuna.create_study(direction='minimize')
    study_ras.optimize(
        make_objective(rastrigin, bounds=[(-5,5), (-5,5)]),
        n_trials=n_trials,
        show_progress_bar=True
    )
    print("Лучшие параметры для Rastrigin:", study_ras.best_params)
    print("Лучший loss:", study_ras.best_value)

    # 4.3 Для синтетической линейной регрессии
    reg_loss = generate_regression_loss(n_samples=200, noise_sigma=0.5)
    study_reg = optuna.create_study(direction='minimize')
    study_reg.optimize(
        make_objective(reg_loss, bounds=[(-10,10)] * 3),
        n_trials=n_trials,
        show_progress_bar=True
    )
    print("Лучшие параметры для регрессии:", study_reg.best_params)
    print("Лучший MSE:", study_reg.best_value)



    # -------------------------------------------------

    # --- Функция Растригина (интересная многоминимальная) ---
    # def weierstrass_1d(ind, a=0.5, b=3, N=50):
    #     """
    #     Одномерная функция Вейерштрасса:
    #     W(x) = sum_{n=0..N-1} a^n * cos(b^n * π * x)
    #     """
    #     x = ind[0]
    #     return sum(a**n * np.cos((b**n) * np.pi * x) for n in range(N))

    # # Параметры и запуск ГА
    # bounds = [(-2, 3.5)]  # поэкспериментируйте с диапазоном, где видна фрактальность
    # hist = genetic_algorithm(
    #     func=weierstrass_1d,
    #     bounds=bounds,
    #     pop_size=50,
    #     generations=100,
    #     crossover_prob=0.8,
    #     mutation_prob=0.1,
    #     mutation_sigma=0.1,
    #     elitism_size=2
    # )['history']

    # # --- Визуализация ---
    # # Значения функции для построения кривой
    # x_vals = np.linspace(bounds[0][0], bounds[0][1], 1000)
    # y_vals = np.array([[weierstrass_1d(np.array([x]))] for x in x_vals]).flatten()

    # out_dir = './ga1d_frames'
    # os.makedirs(out_dir, exist_ok=True)
    # filenames = []

    # for gen, pop in enumerate(hist):
    #     fig, ax = plt.subplots(figsize=(6, 3), dpi=80)
    #     # фоновая кривая Вейерштрасса
    #     ax.plot(x_vals, y_vals)
    #     # текущие особи
    #     ax.scatter(pop[:, 0], [weierstrass_1d(ind) for ind in pop], s=20, color='red')
    #     ax.set_xlim(bounds[0][0], bounds[0][1])
    #     ax.set_ylim(y_vals.min(), y_vals.max())
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_title(f'Поколение {gen+1}', loc='left', fontsize=10)
    #     plt.tight_layout(pad=0)
    #     path = os.path.join(out_dir, f'frame_{gen:03d}.png')
    #     fig.savefig(path)
    #     plt.close(fig)
    #     filenames.append(path)

    # # Собираем GIF
    # frames = [imageio.v2.imread(fn) for fn in filenames]
    # gif_path = './ga1d_weierstrass_evolution.gif'
    # imageio.mimsave(gif_path, frames, duration=0.1)

    # print("GIF сохранён:", gif_path)
