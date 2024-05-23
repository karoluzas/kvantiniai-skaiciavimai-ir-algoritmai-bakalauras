ram_gb = 2
# 16 X 2^n <= RAM AMOUNT
#      2^n <= RAM AMOUNT / 16
ram_amount = (ram_gb * pow(1024, 3))/16

qubit_count = 0

while pow(2, qubit_count) <= ram_amount:
    qubit_count = qubit_count + 1

qubit_count = qubit_count - 1

print(qubit_count)