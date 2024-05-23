import pennylane as qml
import tensorflow as tf
import matplotlib.pyplot as plt
from pennylane import numpy as np
from pennylane.templates import RandomLayers
from tensorflow import keras
import pickle

import logging
import time
from functools import wraps

logging.basicConfig(filename='4x4_trainingas.txt', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# Filter out PennyLane's DEBUG messages
logging.getLogger("pennylane").setLevel(logging.WARNING)

def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logging.info(f"Function '{func.__name__}' started")

        result = func(*args, **kwargs) 

        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Function '{func.__name__}' finished in {execution_time:.2f} seconds")
        return result
    return wrapper

n_epochs = 1000
n_layers = 1
n_train = 1000
n_test = 200

SAVE_PATH = "dataset/quanvolution/" 
PREPROCESS = True
np.random.seed(0)
tf.random.set_seed(0)

cifar_dataset = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar_dataset.load_data()

train_images = train_images[:n_train]
train_labels = train_labels[:n_train]
test_images = test_images[:n_test]
test_labels= test_labels[:n_test]

train_images = train_images / 255
test_images = test_images / 255

q_train_images = np.load(SAVE_PATH + "4x4_final_train_cifar.npy")
q_test_images = np.load(SAVE_PATH + "4x4_final_test_cifar.npy")


def MyQuantumModel():
    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer='adam',
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

@log_execution_time
def RunQCNN():
    model = MyQuantumModel()

    history = model.fit(
        q_train_images,
        train_labels,
        validation_data=(q_test_images, test_labels),
        batch_size=20,
        epochs=n_epochs,
        verbose=2,
        shuffle=True
    )

    return history

q_history = RunQCNN()

def MyClassicalModel():
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (3,3), strides=(4,4), activation='relu', padding='same', input_shape=(32, 32, 3)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation="softmax")
    ])


    model.compile(
        optimizer='adam',
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    return model

@log_execution_time
def RunCNN():
    model = MyClassicalModel()

    history = model.fit(
        train_images,
        train_labels,
        validation_data=(test_images, test_labels),
        batch_size=20,
        epochs=n_epochs,
        verbose=2,
        shuffle=True
    )

    return history

classic_history = RunCNN()


with open('q_history.pkl', 'wb') as f:
    pickle.dump(q_history, f)
    f.close()

with open('c_history.pkl', 'wb') as f:
    pickle.dump(classic_history, f)
    f.close()