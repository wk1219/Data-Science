import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../DataSet/mnist/", one_hot=True)

print(len(mnist.train.labels))  # test images count
print(len(mnist.train.images))  # test labels count

print(mnist.validation.num_examples)    # validation data
print(mnist.test.num_examples)          # test data

print(mnist.test.labels.shape)  # test label shape

print(mnist.test.labels[0:5, :])    # array data

mnist.test.cls = np.argmax(mnist.test.labels, axis=1)   # convert Decimal data
print(mnist.test.cls[0:5])
