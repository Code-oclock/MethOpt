from matplotlib import pyplot as plt


def draw(filename: str, title: str, history: list, n_iter: int):
    plt.figure(figsize=(8,5))
    plt.plot(range(n_iter), history)
    plt.xlabel('Iteration')
    plt.ylabel('best f(x)')
    plt.title(title)
    # plt.legend()
    plt.grid(True)
    plt.savefig(filename)

