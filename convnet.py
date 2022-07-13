import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import random

# path to the directory with binary data
DATA_PATH = './stl10_data'

# number of classes in the STL-10 dataset.
N_CLASSES = 10


def read_labels(path_to_labels):
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

def read_data(path_to_data):
    with open(path_to_data, 'rb') as f:
        bytesread = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(bytesread, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def load_dataset():
    x_train = read_data(DATA_PATH + '/train_X.bin')
    y_train = read_labels(DATA_PATH + '/train_y.bin')
    x_test = read_data(DATA_PATH + '/test_X.bin')
    y_test = read_labels(DATA_PATH + '/test_y.bin')

    x_train = x_train.astype('float32')
    x_train = x_train / 256.0
    x_test = x_test.astype('float32')
    x_test = x_test / 256.0
    
    y_train -= 1
    y_test -= 1

    y_train = tf.keras.utils.to_categorical(y_train, N_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, N_CLASSES)

    return (x_train, y_train), (x_test, y_test)


tf.random.set_seed(1234)
(x_train, y_train), (x_test, y_test) = load_dataset()

#build the cnn
input_shape = (96, 96, 3)
input_img = layers.Input(shape=input_shape)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Dropout(0.5)(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
y = layers.Dense(N_CLASSES, activation='softmax')(x)

convnet = tf.keras.Model(input_img, y)
convnet.summary()

convnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    zoom_range=0.2,
    horizontal_flip=True)

convnet.fit(datagen.flow(x_train, y_train, batch_size=100), steps_per_epoch=len(x_train)/100, epochs=100, verbose=2)
convnet.evaluate(x_test, y_test, verbose=2)


