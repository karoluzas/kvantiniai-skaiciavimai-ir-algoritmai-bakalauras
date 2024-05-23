# Script to measure how many qubits can be utilized in a given system
# Code will fail when no more qubits can be simulated with PennyLane

import pennylane as qml

num_qubits = 0

while True:
    try:
        dev = qml.device('default.qubit', wires=num_qubits)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            for i in range(1, num_qubits):
                qml.CNOT(wires=[i-1, i])
            return qml.expval(qml.PauliZ(wires=num_qubits-1))
        
        circuit()
        print(num_qubits, " qubits: simulation successful.")
        num_qubits += 1
    except Exception as e:
        print(f"Failed at {num_qubits} qubits: {str(e)}")
        break


