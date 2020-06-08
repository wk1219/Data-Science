import numpy as np

# Matrix Operation Example
W = np.array([[1, 2, 3], [4, 5, 6]])
X = np.array([[0, 1, 2], [3, 4, 5]])

print(W + X)
print(W * X)

# Broadcast Example
A = np.array([[1, 2], [3, 4]])
b = np.array([10, 20])
print(A * b)

# Vector Inner Product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b))

# Matrix Product
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(np.matmul(A, B))
