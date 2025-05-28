from matplotlib import pyplot as plt
import config
import draw
import lib


def our_methods():
    tracker = lib.Tracker()
    x, y = lib.load_dataset(config.DATASET_ID)
    w = lib.sgd(
        tracker, x, y, config.ERAS, config.BATCH_SIZE, 
        config.STEP_NAME, config.STEP_SIZE, 
        config.DECAY_RATE, config.REG_TYPE, config.REG_LAMBDA, config.L1_RATIO, config.EPS)
    draw.draw(tracker, "sgd_wine_loss.png")

def modifications():
    tracker = lib.Tracker()
    x, y = lib.make_anisotropic_regression()
    draw.draw_dataset(x, y)
    w = lib.sgd(
        tracker, x, y, config.ERAS, config.BATCH_SIZE, 
        config.STEP_NAME, config.STEP_SIZE, 
        config.DECAY_RATE, config.REG_TYPE, config.REG_LAMBDA, config.L1_RATIO, config.EPS)
    draw.draw_title(tracker, "sgd_wine.png", "SGD on Dataset")

    tracker = lib.Tracker()
    w_momentum = lib.sgd_momentum(
        tracker, x, y, config.ERAS, config.BATCH_SIZE, 
        config.STEP_NAME, config.STEP_SIZE, 
        config.DECAY_RATE, config.BETA)
    draw.draw_title(tracker, "sgd_momentum_wine.png", "SGD with Momentum on Dataset")

    tracker = lib.Tracker()
    w_nesterov = lib.sgd_nesterov(
        tracker, x, y, config.ERAS, config.BATCH_SIZE, 
        config.STEP_NAME, config.STEP_SIZE, 
        config.DECAY_RATE, config.BETA)
    draw.draw_title(tracker, "sgd_nesterov_wine.png", "SGD with Nesterov Momentum on Dataset")


def sgd_effective():
    tracker = lib.Tracker()
    x, y = lib.load_dataset(config.DATASET_ID)
    batch_sizes = [1, 8, 32, 128, 512]

    for bs in batch_sizes:
        print(f'Running experiment with batch_size={bs}...')
        res = lib.run_experiment(tracker, x, y, config.ERAS, bs, 
        config.STEP_NAME, config.STEP_SIZE, 
        config.DECAY_RATE, config.REG_TYPE, config.REG_LAMBDA, config.L1_RATIO, config.EPS)

        print(res)

if __name__ == "__main__":
    # our_methods()
    # sgd_effective()
    modifications()
