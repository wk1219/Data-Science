import numpy as np

W1 = np.random.randn(2, 4)  # Weight
b1 = np.random.randn(4)     # Bias
x = np.random.randn(10, 2)  # Input
h = np.matmul(x, W1) + b1   # h = x * W1 + b1 (h : hidden layer neuron)
