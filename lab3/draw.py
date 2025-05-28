from typing import List
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from lib import Tracker

def draw(tracker: Tracker, picture_name: str):
    fig, ax = plt.subplots()
    ax.plot(tracker.history_errors)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title('SGD на винном датасете')
    plt.tight_layout()
    plt.savefig(picture_name)

def draw_title(tracker: Tracker, picture_name: str, title: str):
    fig, ax = plt.subplots()
    ax.plot(tracker.history_errors)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(picture_name)

def draw_more(trackers: List[Tracker]):
    plt.figure(figsize=(8,5))
    for tracker in trackers:
        plt.plot(range(1, len(tracker.history_errors)+1), tracker.history_errors, label=tracker.name)
    plt.xlabel("Epoch")
    plt.ylabel("Training MSE")
    plt.legend()
    plt.title("Сравнение сходимости оптимизаторов")
    plt.grid(True)
    plt.savefig("compare_optimizers.png", dpi=150)


def draw_dataset(X, y):
    plt.scatter(X[:,1], X[:,2], c=y, cmap='viridis', s=15)
    plt.xlabel('Feature 1 (std)')
    plt.ylabel('Feature 2 (std)')
    plt.title('датасет для линейной регрессии')
    plt.colorbar(label='y')
    plt.savefig('anisotropic_regression.png')

    
