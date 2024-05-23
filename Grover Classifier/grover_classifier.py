import numpy as np
import tensorflow as tf
import random
import pickle
import logging
import time

from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, MCMT, ZGate
from qiskit.primitives import Sampler
from scipy.spatial.distance import jensenshannon
from functools import wraps

# Logging setup
logging.basicConfig(filename='groverio_classification.txt', level=logging.INFO, 
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

# Loading in data
with open('quasi_distributions.pkl', 'rb') as f:
    quasi_distributions = pickle.load(f)
    f.close()

with open('top_pixels.pkl', 'rb') as f:
    top_pixels = pickle.load(f)
    f.close()

with open('processed_images.pkl', 'rb') as f:
    processed_images = pickle.load(f)
    f.close()

# Making predictions
@log_execution_time
def grover_oracle(binary_states):
    if not isinstance(binary_states, list):
        binary_states = [binary_states]

    num_qubits = len(binary_states[0])

    qc = QuantumCircuit(num_qubits)

    for target in binary_states:
        rev_target = target[::-1]
        zero_inds = [ind for ind in range(num_qubits) if rev_target.startswith("0", ind)]
        if len(zero_inds)>0:
            qc.x(zero_inds)
            qc.compose(MCMT(ZGate(), num_qubits - 1, 1), inplace=True)
            qc.x(zero_inds)
        else:
            qc.compose(MCMT(ZGate(), num_qubits - 1, 1), inplace=True)
    return qc

@log_execution_time
def get_random_image(indice):
    if indice == -1:
        random_class = random.randint(0, 9)
    else:
        random_class = indice

    random_indice = random.randint(1000, 5000)

    random_image = processed_images[random_class][random_indice]

    print(f"Random processed image of class: {random_class}, at position {random_indice}.")
    return random_image, random_class

@log_execution_time
def get_image_binary(image, top_pixels):
    flattened_image = [pixel for row in image for pixel in row]

    random_image_binary = [[format(index, '08b') for index in sublist] for sublist in top_pixels]
    random_image_binary_filtered = []

    for class_indices, class_binary in zip(top_pixels, random_image_binary):
        temp = []
        for indice, binary in zip(class_indices, class_binary):
            binary = binary + str(int(flattened_image[indice]))
            temp.append(binary)
        random_image_binary_filtered.append(temp)

    return random_image_binary_filtered

@log_execution_time
def get_grover_operators(binary_states):
    grover_oracles = []

    for states in binary_states:
        oracle = grover_oracle(states)

        grover_op = GroverOperator(oracle)
        grover_oracles.append(grover_op)

    return grover_oracles

@log_execution_time
def get_image_quasi_distributions(indice):
    random_image, random_image_class = get_random_image(indice)
    random_image_binary = get_image_binary(random_image, top_pixels)

    random_image_quasi_distributions = []

    grover_operators = get_grover_operators(random_image_binary)

    for grover_op in grover_operators:
        qc = QuantumCircuit(grover_op.num_qubits)
        qc.h(range(grover_op.num_qubits))
        qc.compose(grover_op.power(8), inplace=True)
        qc.measure_all()

        sampler = Sampler()

        result = sampler.run(qc).result()
        quasi_dists = result.quasi_dists

        random_image_quasi_distributions.append(quasi_dists)

    return random_image_quasi_distributions, random_image_class

@log_execution_time
def get_jensen_shannon_divergence(quasi_distributions, new_image_quasi_distributions):
    js_divergences = []

    for i in range(10):
        real_distribution = quasi_distributions[i][0]
        predicted_distribution = new_image_quasi_distributions[i][0]

        keys = sorted(real_distribution.keys())
        p = np.array([real_distribution[key] for key in keys])
        q = np.array([predicted_distribution[key] for key in keys])

        js_divergence = jensenshannon(p, q, base=2)

        js_divergences.append(js_divergence)

    return js_divergences

@log_execution_time
def get_prediction(js_distributions):
    return js_distributions.index(min(js_distributions))

# ------------------------------------------------
#       General predictions on random images
# ------------------------------------------------
correct_predictions = 0

for i in range(50):
    distributions, correct_class = get_image_quasi_distributions(indice=-1)
    js = get_jensen_shannon_divergence(quasi_distributions, distributions)

    predicted_class = get_prediction(js)

    print(f"Predicted class: {predicted_class}, actuall class: {correct_class}")

    if predicted_class == correct_class:
        correct_predictions = correct_predictions + 1

    if correct_predictions != 0 and i != 0:
        print(f"Prediction accuracy: {correct_predictions / (i + 1)}")
    
print(correct_predictions)

# ------------------------------------------------
#      Predictions on specific image classes
# ------------------------------------------------
correct_class_predictions = []

for i in range(10):
    pred = 0

    for j in range(30):
        distributions, correct_class = get_image_quasi_distributions(indice=i)
        js = get_jensen_shannon_divergence(quasi_distributions, distributions)

        predicted_class = get_prediction(js)

        print(f"Predicted class: {predicted_class}, actuall class: {correct_class}")

        if predicted_class == correct_class:
            pred = pred + 1

        if pred != 0 and j != 0:
            print(f"Prediction accuracy: {pred / (j + 1)}")
    
    print(f"Correct guesses {pred}/{j}")
    correct_class_predictions.append(pred)

print(correct_class_predictions)