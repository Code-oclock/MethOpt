import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from lib import Tracker

def draw(tracker: Tracker):
    fig, ax = plt.subplots()
    ax.plot(tracker.history_errors)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title('SGD на винном датасете')
    plt.tight_layout()
    plt.savefig('sgd_wine_loss.png')

