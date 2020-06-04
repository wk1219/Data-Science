from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

plt.rcParams['toolbar'] = 'None'

fg1 = plt.figure(1, figsize=(3, 3))
#fg1.canvas.window().statusBar().setVisible(False)
ax1 = fg1.add_axes([0, 0, 1, 1])
ax1.imshow(train_images[0], cmap='gray', aspect='auto')
ax1.axis('off')
ax1.text(1, 1, "Label : {}".format(train_images[0]), fontsize=10, color='white')

fg2 = plt.figure(2, figsize=(3, 3))
#fg2.canvas.window().statusBar().setVisible(False)
ax2 = fg2.add_axes([0, 0, 1, 1])
ax2.imshow(test_images[0], cmap='gray', aspect='auto')
ax2.axis('off')
ax2.text(1, 1, "Label : {}".format(test_labels[0]), fontsize=10, color='white')

train_input = train_images
test_input = test_images

train_target = keras.utils.to_categorical(train_labels)
test_target = keras.utils.to_categorical(test_labels)

model = keras.Sequential(
    layers=[
        keras.layers.Flatten(input_shape=(28, 28), name='Input'),
        keras.layers.Dense(128, activation='sigmoid', name='Hidden_1'),
        keras.layers.Dense(128, activation='sigmoid', name='Hidden_2'),
        keras.layers.Dense(128, activation='sigmoid', name='Hidden_3'),
        keras.layers.Dense(10, activation='sigmoid', name='Output')
        ],
    name="MNIST_Classifier"
)
model.summary()

model.compile(optimizer=keras.optimizers.SGD(lr=0.2),
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit(train_input, train_target, epochs=20, verbose=2)

test_loss, test_acc = model.evaluate(test_input, test_target, verbose=2)
print('\nTest accuracy:', test_acc)

predictions = model.predict(test_input)

