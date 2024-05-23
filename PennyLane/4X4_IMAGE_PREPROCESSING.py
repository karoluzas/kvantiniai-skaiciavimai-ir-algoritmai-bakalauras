import pennylane as qml
import tensorflow as tf
import matplotlib.pyplot as plt
from pennylane import numpy as np
from pennylane.templates import RandomLayers
from tensorflow import keras

# ======================================
# --------------LOGGING-----------------
# ======================================
import logging
import time
from functools import wraps

logging.basicConfig(filename='4x4_preprocessing_training.txt', level=logging.INFO, 
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

# ======================================
# --------------VARIABLES---------------
# ======================================

n_epochs = 30 
n_layers = 1
n_train = 1000
n_test = 200

SAVE_PATH = "dataset/quanvolution/" 
PREPROCESS = True
np.random.seed(0)
tf.random.set_seed(0)

# ======================================
# --------LOADING IN CIFAR--------------
# ======================================

cifar_dataset = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar_dataset.load_data()

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

# ======================================
# ---------QUANTUM CIRCUIT--------------
# ======================================

dev = qml.device("default.qubit", wires=20)
rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, 16))

@qml.qnode(dev)
def circuit(phi):
    for i in range(16):
        qml.RY(np.pi * phi[i], wires=i)

    RandomLayers(rand_params, wires=list(range(16)))

    # Pooling operation
    for i in range(0, 16, 4):
        qml.CNOT(wires=[i, i+1])
        qml.CNOT(wires=[i+1, i+2])
        qml.CNOT(wires=[i+2, i+3])
        qml.SWAP(wires=[i+3, 16 + i//4])

    return [qml.expval(qml.PauliZ(i)) for i in range(16, 20)]

@log_execution_time
def quanv(image):
    out = np.zeros((8, 8, 4))

    for i in range(0, 32, 4):
        for j in range(0, 32, 4):
            q_results = circuit(
                [
                    image[i, j, 0],
                    image[i, j+1, 0],
                    image[i, j+2, 0],
                    image[i, j+3, 0],
                    image[i+1, j, 0],
                    image[i+1, j+1, 0],
                    image[i+1, j+2, 0],
                    image[i+1, j+3, 0],
                    image[i+2, j, 0],
                    image[i+2, j+1, 0],
                    image[i+2, j+2, 0],
                    image[i+2, j+3, 0],
                    image[i+3, j, 0],
                    image[i+3, j+1, 0],
                    image[i+3, j+2, 0],
                    image[i+3, j+3, 0]
                ]
            )

            for c in range(4):
                out[i // 4, j // 4, c] = q_results[c]
    
    return out

# ======================================
# -----------PREPROCESSING--------------
# ======================================

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

    # Save pre-processed images
    np.save(SAVE_PATH + "4x4_final_train_cifart.npy", q_train_images)
    np.save(SAVE_PATH + "4x4_final_test_cifar.npy", q_test_images)
