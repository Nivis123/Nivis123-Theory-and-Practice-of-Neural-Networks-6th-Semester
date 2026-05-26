import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = tf.pad(x_train, [[0,0],[2,2],[2,2]]).numpy()
x_test = tf.pad(x_test, [[0,0],[2,2],[2,2]]).numpy()

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]


y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = models.Sequential([
    layers.Conv2D(6, (5,5), activation='tanh', input_shape=(32,32,1)),
    layers.AveragePooling2D((2,2), strides=2),
    layers.Conv2D(16, (5,5), activation='tanh'),
    layers.AveragePooling2D((2,2), strides=2),
    layers.Flatten(),
    layers.Dense(120, activation='tanh'),
    layers.Dense(84, activation='tanh'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.title('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('library_training.png')
plt.show()