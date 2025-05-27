import config
import lib


def our_methods():
    x, y = lib.load_dataset(config.DATASET_ID)
    w = lib.sgd(x, y, config.ERAS, config.BATCH_SIZE, config.STEP_NAME, config.STEP_SIZE, config.DECAY_RATE, config.EPS)
    lib.test_sgd(w, x[401], y[401])
    # print("Weights:", w)


our_methods()


# 3) Разбиваем (например, 70% train, 15% val, 15% test)
# X_trainval, X_test, y_trainval, y_test = train_test_split(
#     X, y, test_size=0.15, random_state=42
# )

# X_train, X_val, y_train, y_val = train_test_split(
#     X_trainval, y_trainval, test_size=0.1765,  # ≈15% от всего
#     random_state=42
# )
# print(X_train)
# print(y_train)

# 4) Стандартизируем только по train
# scaler = StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_val   = scaler.transform(X_val)
# X_test  = scaler.transform(X_test)

# 5) Добавляем bias (столбец единиц)
# X_train = np.hstack([np.ones((X_train.shape[0],1)), X_train])
# X_val   = np.hstack([np.ones((X_val.shape[0],1)),   X_val])
# X_test  = np.hstack([np.ones((X_test.shape[0],1)),  X_test])

# 6) Вызываем manual_sgd
# w, history = manual_sgd(
#     X_train, y_train,
#     lr=0.01,
#     batch_size=32,
#     n_epochs=100,
#     reg=None,
# )

# Можно сразу смотреть, как меняется loss:
# import matplotlib.pyplot as plt
# plt.plot(history['loss'])
# plt.xlabel('Epoch')
# plt.ylabel('MSE')
# plt.title('SGD на винном датасете')
# plt.savefig('sgd_wine_loss.png')
