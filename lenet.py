
from pickletools import optimize
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

LEARNING = 0.001
ACTIVATION = 'sigmoid'
BATCH = 10
OPTI = 'sgd'
EPOCHS = 50
LOSSY = 'mean'

# downloads the cifar10 dataset
(train_images, train_labels), (test_images, test_labels) = \
    datasets.cifar10.load_data()

# normalizes pixel values btwn 0 & 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# label names if I ever need those
class_names = ['airplane', 'automobile', 'bird', 'cat', \
     'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


model = models.Sequential([
    layers.Conv2D(6, (5, 5), activation=ACTIVATION, input_shape=(32, 32, 3)),
    layers.AveragePooling2D((2, 2), strides=(2, 2)),
    layers.Conv2D(16, (5, 5), activation=ACTIVATION),
    layers.AveragePooling2D((2, 2), strides=(2, 2)),
    layers.Flatten(),
    layers.Dense(120, activation=ACTIVATION),
    layers.Dense(84, activation=ACTIVATION),
    layers.Dense(10),
])

# this prints out the model network
model.summary()
print('Learning Rate: ', LEARNING)
print('Activation: ', ACTIVATION)
print('Batch Size: ', BATCH)
print('Optimizer: ', OPTI)
print('Epochs: ', EPOCHS)

adam = tf.keras.optimizers.Adam(learning_rate=LEARNING)
sgd = tf.keras.optimizers.SGD(learning_rate=LEARNING)

if OPTI == 'sgd':
    optim = sgd
elif OPTI == 'adam':
    optim = adam

if LOSSY == 'mean':
    print('Loss function: Mean Squared Error\n')
    model.compile(optimizer=optim,
                    loss= tf.keras.losses.MeanSquaredError(),
                    metrics=['accuracy'])
elif LOSSY == 'cross':
    print('Loss function: Sparse Cat. CrossEntropy\n')
    model.compile(optimizer=optim,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])



history = model.fit(train_images, train_labels, batch_size=BATCH, epochs=EPOCHS,
                    validation_data=(test_images, test_labels), workers=2)

fig, (ax1, ax2) = plt.subplots(1, 2)

fig.suptitle("Accuracy and loss")

ax1.plot(history.history['accuracy'], label='accuracy')
ax1.plot(history.history['val_accuracy'], label='val_accuracy')
ax1.set_title('Accuracy %')
# ax1.xlabel('Epoch')
# ax1.ylabel('Accuracy')
# ax1.ylim([0.0, .8])
ax1.legend(loc='lower right')


# ax2 = plt.subplot(1, 2, 2)
ax2.plot(history.history['loss'], label='loss')
ax2.plot(history.history['val_loss'], label='val_loss')
ax2.set_title('Loss Value')
# ax2.xlabel('Epoch')
# ax2.ylabel('Accuracy')
# ax2.ylim([0.5, 3])
ax2.legend(loc='upper right')
plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(test_acc)
