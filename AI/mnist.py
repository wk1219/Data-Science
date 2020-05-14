import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../DataSet/mnist/", one_hot=True)

print(len(mnist.train.labels))  # Test images count
print(len(mnist.train.images))  # Test labels count

print(mnist.validation.num_examples)    # Validation data
print(mnist.test.num_examples)          # Test data

print(mnist.test.labels.shape)  # Test label shape

print(mnist.test.labels[0:5, :])    # Array data

mnist.test.cls = np.argmax(mnist.test.labels, axis=1)   # Convert Decimal data
print(mnist.test.cls[0:5])

num_epochs = 30
learning_rate = 0.01

num_node_input = 28*28  # Image resolution is 28x28 -> The number of nodes is 784.
num_node_hidden1 = 256  # Hidden layer1 hyper parameter
num_node_hidden2 = 256  # Hidden layer2 hyper parameter
num_node_output = 10    # Output nodes

# Making model
x_true = tf.placeholder(tf.float32, [None, num_node_input])     # Input layer definition
y_true = tf.placeholder(tf.float32, [None, num_node_output])    # Output layer definition

# Hidden layer1 setting
weight_1 = tf.Variable(tf.truncated_normal([num_node_input, num_node_hidden1], stddev=0.01))    # Weight setting
bias_1 = tf.Variable(tf.zeros([num_node_hidden1]))  # Bias setting

# Hidden layer2 setting
weight_2 = tf.Variable(tf.truncated_normal([num_node_hidden1, num_node_hidden2], stddev=0.01))  # Weight setting
bias_2 = tf.Variable(tf.zeros([num_node_hidden2]))  # Bias setting

# Hidden layer3 setting
weight_3 = tf.Variable(tf.truncated_normal([num_node_hidden2, num_node_output], stddev=0.01))   # Weight setting
bias_3 = tf.Variable(tf.zeros([num_node_output]))   #Bias setting

# Matrix Calculation
hidden_1 = tf.nn.relu(tf.add(tf.matmul(x_true, weight_1), bias_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, weight_2), bias_2))
y_pred = tf.nn.relu(tf.add(tf.matmul(hidden_2, weight_3), bias_3))

# Cross entropy (predict - real)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))    # Using softmax
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)                                # Using Adam Optimizer

# Running session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Grouping data and feed
batch_size = 100
total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(num_epochs):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, {x_true:batch_xs, y_true:batch_ys})
        total_cost += sess.run(cost, {x_true:batch_xs, y_true:batch_ys})

    print("Epoch : {%04d}" % (epoch + 1), "Cost : {:.3f}".format(total_cost/total_batch))

print("Optimization Complete")

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy : ", sess.run(accuracy, {x_true: mnist.test.images, y_true: mnist.test.labels}))
