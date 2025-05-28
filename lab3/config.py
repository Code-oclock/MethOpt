ERAS=20
BATCH_SIZE=64
STEP_SIZE=0.01
EPS=0.01
DECAY_RATE=0.1
STEP_NAME="constant"  # "constant", "linear", "inverse_time", "exponential"
BETA=0.9  # для momentum и для nesterov, 0.0 - без momentum
REG_TYPE=None  # None, 'l1', 'l2', 'elastic'
REG_LAMBDA=0.0 # сила штрафа
L1_RATIO=0.5  # для elastic, 0.0 - только l2, 1.0 - только l1
PICTURE_NAME="sgd_wine.png"  # имя файла для сохранения графика

DATASET_ID=186 # DONT CHANGE THIS VALUE

