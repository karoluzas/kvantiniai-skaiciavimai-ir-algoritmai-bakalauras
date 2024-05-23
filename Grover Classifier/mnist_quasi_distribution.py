import numpy as np
import tensorflow as tf
import scipy
import pickle
import logging
import time

from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, MCMT, ZGate
from qiskit.primitives import Sampler
from functools import wraps

# Logging setup
logging.basicConfig(filename='groverio_quasi.txt', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

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

# Loading in MNIST dataset
mnist_dataset = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

class_arrays = [[] for _ in range(10)]

for image, label in zip(train_images, train_labels):
    class_arrays[label].append(image)

class_arrays = [tf.convert_to_tensor(images) for images in class_arrays]

for i, class_array in enumerate(class_arrays):
    print(f"Class {i} has {class_array.shape[0]} images.")

# Processing each class of images
@log_execution_time
def process_class(class_array):
    new_array = []
    for image in class_array:
        image = image.numpy() / 255.0

        image = scipy.ndimage.zoom(image, (0.5, 0.5))

        image = np.where(image > 0.75, 1, image)
        image = np.where(image < 0.25, 0, image)

        padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
        
        neighbors_sum = (
            padded_image[:-2, :-2] + padded_image[1:-1, :-2] + padded_image[2:, :-2] +
            padded_image[:-2, 1:-1] + padded_image[1:-1, 1:-1] + padded_image[2:, 1:-1] +
            padded_image[:-2, 2:] + padded_image[1:-1, 2:] + padded_image[2:, 2:]
        )

        average = neighbors_sum / 9

        mask = (image <= 0.75) & (image >= 0.25)

        image = np.where((image <= 0.75) & (image >= 0.25), 0, image)

        filtered_array = np.where(mask, average, 0)

        filtered_array = np.where(filtered_array > 0.6, 1, filtered_array)
        filtered_array = np.where(filtered_array <= 0.6, 0, filtered_array)


        image = np.where(filtered_array == 1, 1, image)

        new_array.append(image)

    return new_array

processed_images = [
    process_class(class_arrays[0]),
    process_class(class_arrays[1]),
    process_class(class_arrays[2]),
    process_class(class_arrays[3]),
    process_class(class_arrays[4]),
    process_class(class_arrays[5]),
    process_class(class_arrays[6]),
    process_class(class_arrays[7]),
    process_class(class_arrays[8]),
    process_class(class_arrays[9]),
]

# Binary conversion
top_pixels = []

for images in processed_images:
    summed_images = np.sum(images, axis=0)

    flat_indices = np.argsort(summed_images, axis=None)[-10:]

    top_pixels.append(flat_indices.tolist())

binary_top_pixels = [[format(index, '08b') + '1' for index in sublist] for sublist in top_pixels]

# Grover
@log_execution_time
def grover_oracle(marked_states):
    if not isinstance(marked_states, list):
        marked_states = [marked_states]
    
    num_qubits = len(marked_states[0])

    qc = QuantumCircuit(num_qubits)
    for target in marked_states:
        rev_target = target[::-1]
        zero_inds = [ind for ind in range(num_qubits) if rev_target.startswith("0", ind)]
        if len(zero_inds)>0:
            qc.x(zero_inds)
            qc.compose(MCMT(ZGate(), num_qubits - 1, 1), inplace=True)
            qc.x(zero_inds)
        else:
            qc.compose(MCMT(ZGate(), num_qubits - 1, 1), inplace=True)
    return qc

grover_oracles = []

for states in binary_top_pixels:
    oracle = grover_oracle(states)

    grover_op = GroverOperator(oracle)
    grover_oracles.append(grover_op)

quasi_distributions = []

for grover_op in grover_oracles:
    qc = QuantumCircuit(grover_op.num_qubits)
    qc.h(range(grover_op.num_qubits))
    qc.compose(grover_op.power(8), inplace=True)
    qc.measure_all()

    sampler = Sampler()

    result = sampler.run(qc).result()
    quasi_dists = result.quasi_dists

    quasi_distributions.append(quasi_dists)

# Saving required information for other scripts
@log_execution_time
def save_information():
    with open('quasi_distributions.pkl', 'wb') as f:
        pickle.dump(quasi_distributions, f)

    with open('top_pixels.pkl', 'wb') as top_pixel_file:
        pickle.dump(top_pixels, top_pixel_file)

    with open('processed_images.pkl', 'wb') as processed_images_file:
        pickle.dump(processed_images, processed_images_file)

save_information()