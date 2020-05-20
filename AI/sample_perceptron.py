import matplotlib.pyplot as plt
import numpy as np

def step(y):
    if y < 1:
        return 0
    else:
        return 1

n = 1000
Data1 = np.random.randn(n, 2)
Data2 = np.random.randn(n, 2) + 5

Label1 = np.zeros((n, 1), dtype=np.int)
Label2 = np.ones((n, 1), dtype=np.int)

plt.plot(Data1[:, 0], Data1[:, 1], "*")
plt.plot(Data2[:, 0], Data2[:, 1], "*")

TrainData = np.vstack([Data1, Data2])
LabelData = np.vstack([Label1, Label2])

num_data = np.size(TrainData, 0)

feature_num = 2     # feature : x, y (count : 2)
num_output = 1

w = np.random.randn(feature_num, num_output)
b = np.random.randn(1)

x1 = np.arange(-3, 8)
x2 = (-w[0]*x1 - b) / w[1]
plt.plot(x1, x2)

num_epoch = 100
learning_rate = 0.5

for epoch in range(num_epoch):
    for i in range(num_data):
        x = TrainData[i]
        target = LabelData[i]

        y = np.dot(x, w) + b
        z = step(y)
        e = target - z

        delta_w = learning_rate * e * np.transpose([x])
        delta_b = learning_rate * e * 1

        w = w + delta_w
        b = b + delta_b

x1 = np.arange()
x1 = np.arange(-3, 8)
x2 = (-w[0]*x1 - b) / w[1]
plt.plot(x1, x2, 'r')
plt.show()
