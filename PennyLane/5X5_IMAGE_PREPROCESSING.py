import pennylane as qml
import tensorflow as tf
import matplotlib.pyplot as plt
from pennylane import numpy as np
from pennylane.templates import RandomLayers
from tensorflow import keras

import logging
import time
from functools import wraps

logging.basicConfig(filename='5x5_preprocessing_training.txt', level=logging.INFO, 
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

n_epochs = 30
n_layers = 1
n_train = 1
n_test = 1

SAVE_PATH = "dataset/quanvolution/"
PREPROCESS = True
np.random.seed(0)
tf.random.set_seed(0)

cifar_dataset = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar_dataset.load_data()

# Reducing dataset size
train_images = train_images[:n_train]
train_labels = train_labels[:n_train]
test_images = test_images[:n_test]
test_labels= test_labels[:n_test]

# Convert RGB images to grayscale
def rgb_to_grayscale(images):
    return np.dot(images[...,:3], [0.299, 0.587, 0.114])

train_images_gray = rgb_to_grayscale(train_images)
test_images_gray = rgb_to_grayscale(test_images)

train_images_gray = train_images_gray / 255
test_images_gray = test_images_gray / 255

train_images_gray = np.array(train_images_gray[..., tf.newaxis], requires_grad=False)
test_images_gray = np.array(test_images_gray[..., tf.newaxis], requires_grad=False)

dev = qml.device("lightning.qubit", wires=30)
rand_params = np.random.uniform(high=2*np.pi, size=(n_layers, 25))

@qml.qnode(dev, interface='tf')
def circuit(phi):
    for i in range(25):
        qml.RY(np.pi * phi[i], wires=i)
    
    RandomLayers(rand_params, wires=list(range(25)))

    for i in range(0, 25, 5):
        qml.CNOT(wires=[i, i+1])
        qml.CNOT(wires=[i+1, i+2])
        qml.CNOT(wires=[i+2, i+3])
        qml.CNOT(wires=[i+3, i+4])
        qml.SWAP(wires=[i+4, 25 + i//5])

    qml.CNOT(wires=[25, 26])

    return [qml.expval(qml.PauliZ(i)) for i in range(26, 30)]


@log_execution_time
def quanv(image):
    out = np.zeros((6, 6, 4))

    for i in range(0, 32, 5):
        if i == 30:
            break

        for j in range(0, 32, 5):
            if j == 30:
                break

            # Process a squared 5x5 region of the image with a quantum circuit
            q_results = circuit(
                [
                    image[i, j, 0],
                    image[i, j+1, 0],
                    image[i, j+2, 0],
                    image[i, j+3, 0],
                    image[i, j+4, 0],
                    image[i+1, j, 0],
                    image[i+1, j+1, 0],
                    image[i+1, j+2, 0],
                    image[i+1, j+3, 0],
                    image[i+1, j+4, 0],
                    image[i+2, j, 0],
                    image[i+2, j+1, 0],
                    image[i+2, j+2, 0],
                    image[i+2, j+3, 0],
                    image[i+2, j+4, 0],
                    image[i+3, j, 0],
                    image[i+3, j+1, 0],
                    image[i+3, j+2, 0],
                    image[i+3, j+3, 0],
                    image[i+3, j+4, 0],
                    image[i+4, j, 0],
                    image[i+4, j+1, 0],
                    image[i+4, j+2, 0],
                    image[i+4, j+3, 0],
                    image[i+4, j+4, 0],
                ]
            )

            for c in range(4):
                out[i // 5, j // 5, c] = q_results[c]
    
    return out

if PREPROCESS == True:
    q_train_images = []

    print("Quantum pre-processing of train images:")
    for idx, img in enumerate(train_images_gray):
        print("{}/{}".format(idx + 1, n_train), end='\r')
        q_train_images.append(quanv(img))

    q_train_images = np.asarray(q_train_images)

    q_test_images = []
    print("\nQuantum pre-processing of test images:")
    for idx, img in enumerate(test_images_gray):
        print("{}/{}        ".format(idx + 1, n_test), end="\r")
        q_test_images.append(quanv(img))
    
    q_test_images = np.asarray(q_test_images)

    np.save(SAVE_PATH + "5x5_final_train.npy", q_train_images)
    np.save(SAVE_PATH + "5x5_final_test.npy", q_test_images)