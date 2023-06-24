import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import datetime
import os

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

mode = 0
number_of_classes = 10
HISTORY_DIR = f"./history/"
if mode == 0:
    HISTORY_DIR += "mnist"
    X_train = np.load("data/mnist/kmnist-train-imgs.npz")["arr_0"]
    y_train = np.load("data/mnist/kmnist-train-labels.npz")["arr_0"]

    X_test = np.load("data/mnist/kmnist-test-imgs.npz")["arr_0"]
    y_test = np.load("data/mnist/kmnist-test-labels.npz")["arr_0"]
elif mode == 1:
    HISTORY_DIR += "k49"
    number_of_classes = 49
    X_train = np.load("data/k49/k49-train-imgs.npz")["arr_0"]
    y_train = np.load("data/k49/k49-train-labels.npz")["arr_0"]

    X_test = np.load("data/k49/k49-test-imgs.npz")["arr_0"]
    y_test = np.load("data/k49/k49-test-labels.npz")["arr_0"]

X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

logdir = os.path.join(HISTORY_DIR, datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(logdir, 'model'),
    save_best_only=True
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    os.path.join(logdir, 'logs'),
)

model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(number_of_classes, activation='linear')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[model_checkpoint_callback, tensorboard_callback])

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)