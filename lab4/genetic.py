import numpy as np

# Реализация вещественно-кодированного генетического алгоритма с рулеточным отбором
# для функции f(x,y)=x^2+y^2

def fitness_from_loss(loss, eps=1e-8):
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
    Мутация: с вероятностью gene_mut_prob добавляет гауссов шум.
    При опции bounds ограничивает значения.
    """
    mutant = individual.copy()
    for j in range(len(mutant)):
        if np.random.rand() < gene_mut_prob:
            mutant[j] += np.random.normal(scale=sigma)
            if bounds is not None:
                mutant[j] = np.clip(mutant[j], bounds[j][0], bounds[j][1])
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
    for gen in range(generations):
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
        'history': best_history
    }


if __name__ == "__main__":
    f = lambda v: v[0]**2 + v[1]**2
    bounds = [(-5,5), (-5,5)]

    result = genetic_algorithm(
        func=f,
        bounds=bounds,
        pop_size=20,
        generations=100,
        crossover_prob=0.8,
        mutation_prob=0.1,
        mutation_sigma=0.1,
        elitism_size=2
    )

    print(f"Лучшее поколение: {result['best_generation']}")
    print(f"Лучшее решение: x={result['best_individual'][0]:.4f}, y={result['best_individual'][1]:.4f}")
    print(f"Значение функции: {result['best_loss']:.4f}")
