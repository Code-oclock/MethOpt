import numpy as np
import matplotlib.pyplot as plt
import lib
import draw
from config import *


# Parameters
bounds_ros = np.array([[-2, 2]] * 2)
bounds_ras = np.array([[-5.12, 5.12]] * 2)

_, _, history_ros = lib.simulated_annealing(
    lib.rosenbrock, 
    np.array(start_point),
    T0, lib.temperature_schedule(t_schedule), n_iter,
    bounds_ros, step_size
)

_, _, history_ras = lib.simulated_annealing(
    lib.rastrigin, 
    np.array(start_point),
    T0, lib.temperature_schedule(t_schedule), n_iter,
    bounds_ras, step_size
)


draw.draw('simulated_annealing_rosenbrock.png', 'simulated annealing rosenbrock', history_ros, n_iter)
draw.draw('simulated_annealing_rastrigin.png', 'simulated annealing rastrigin', history_ras, n_iter)

# plt.figure(figsize=(8,5))

