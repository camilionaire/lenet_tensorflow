
# tensorflow and keras for making the model
from os import sep
from numpy import expand_dims
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
# this so we can plot that data afterwards
import matplotlib.pyplot as plt

# this is so I can change the output of the model
from keras.models import Model

LEARNING = 0.1
ACTIVATION = 'sigmoid'
BATCH = 10
OPTI = 'sgd'
EPOCHS = 50
LOSSY = 'cross'
KERNEL = 5

# downloads the cifar10 dataset
(train_images, train_labels), (test_images, test_labels) = \
    datasets.cifar10.load_data()

# normalizes pixel values btwn 0 & 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# label names if I ever need those
class_names = ['airplane', 'automobile', 'bird', 'cat', \
     'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


model = models.Sequential([
    layers.Conv2D(6, (KERNEL, KERNEL), activation=ACTIVATION, input_shape=(32, 32, 3)),
    layers.AveragePooling2D((2, 2), strides=(2, 2)),
    layers.Conv2D(16, (KERNEL, KERNEL), activation=ACTIVATION),
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
print(f'Kernel size: {KERNEL}x{KERNEL}')

print("third layer is: ", model.layers[2].name)

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
ax1.grid(visible=True)
ax1.legend(loc='lower right')


# ax2 = plt.subplot(1, 2, 2)
ax2.plot(history.history['loss'], label='loss')
ax2.plot(history.history['val_loss'], label='val_loss')
ax2.set_title('Loss Value')
ax2.grid(visible=True)
# ax2.xlabel('Epoch')
# ax2.ylabel('Accuracy')
# ax2.ylim([0.5, 3])
ax2.legend(loc='upper right')
plt.show()

# this just runs the testing info through the finalized model.
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
# prints it
print(test_acc)

################################################################################
# This is all stuff to print out the feature maps.
model = Model(inputs=model.inputs, outputs=model.layers[2].output)

# first 10 test images... I think, we'll see...
# this is just the size of the images?...
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])

    # CIFAR labels are arrays,
    # which is why you need the extra index.
    plt.xlabel(class_names[train_labels[i][0]])
plt.savefig('features/before.png')
plt.show()


for index in range(0, 10):
    # first 0 is the index, second zero is there for something else...
    print("I am trying to do the feature maps of a", class_names[train_labels[index][0]])
    img = expand_dims(train_images[index], axis=0)
    feature_maps = model.predict(img)
    # plot all 16 maps in a 4x4 square of squares
    square = 4
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn off axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(feature_maps[0, :, :, ix-1])#, cmap='gray')
            ix += 1
    # can change the first 0 to i... 
    plt.suptitle(class_names[train_labels[index][0]])
    # might save the figure to the spot... can change 0 with i
    plt.savefig('features/'+ str(index) + '.png')
# show the figure... if it works :-\
# plt.show()
