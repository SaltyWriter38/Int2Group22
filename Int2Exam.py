import tensorflow as tf
import torch
import numpy as np
from scipy.io import loadmat
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.utils import np_utils

#Defining hyper=parameters
IMG_WIDTH = 244
IMG_HEIGHT = 244
IMG_CHANNELS = 3
BATCH_SIZE =200
NUM_EPOCHS = 10000
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.4

def main():
    #Loading in the data
    train_data, train_labels = load_data('trnid')
    valid_data, valid_labels = load_data('valid')
    train_data, train_labels = preprocess(train_data, train_labels, augment=True)
    valid_data, valid_labels = preprocess(valid_data, valid_labels)

    #Convert the labels to one-hot encoding
    train_labels = np_utils.to_categorical(train_labels)
    valid_labels = np_utils.to_categorical(valid_labels)
    #Create the model
    model = create_model(input_shape=train_data.shape[1:], num_classes=valid_labels.shape[1])
    optimizer = Adam(learning_rate= LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    history = model.fit(train_data, train_labels, validation_data=(valid_data, valid_labels), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

    # Evaluate model
    scores = model.evaluate(valid_data, valid_labels, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

def load_data(dataset):
    setid = loadmat('setid.mat')
    data_ids = setid[dataset][0]
    labelfile = loadmat('imagelabels.mat')
    labels = labelfile['labels'][0]
    for i in range(len(labels)):
        labels[i] = labels[i] - 1
    return data_ids, labels[data_ids - 1]

def preprocess(image_ids, labels, augment=False):
    images = []
    for i, image_id in enumerate(image_ids):
        # Read image and convert to float
        image = tf.io.decode_jpeg(tf.io.read_file(f'jpg/image_{str(image_id).zfill(5)}.jpg'))
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT])
        # Normalize image
        image = image / 255.0
        # Add to list of preprocessed images and corresponding label
        images.append(image)
    return np.asarray(images), np.asarray(labels, dtype=np.float32)

def create_model(input_shape, num_classes):
    print(f"input_shape: {input_shape}")
    #The models architecture
    model = tf.keras.Sequential()
    model.add(keras.layers.Conv2D(32, 3, input_shape=input_shape, padding='same'))
    model.add(keras.layers.Activation('tanh'))
    model.add(keras.layers.Conv2D(32, 3, activation='tanh', padding='same'))
    model.add(keras.layers.MaxPooling2D(2))
    model.add(keras.layers.Conv2D(64, 3, activation='tanh', padding='same'))
    model.add(keras.layers.MaxPooling2D(2))
    model.add(keras.layers.Conv2D(128, 3, activation='tanh', padding='same'))
    model.add(keras.layers.MaxPooling2D(2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='tanh'))
    model.add(keras.layers.Dropout(DROPOUT_RATE))
    model.add(keras.layers.Dense(256, activation='tanh'))
    model.add(keras.layers.Dropout(DROPOUT_RATE))
    model.add(keras.layers.Dense(128, activation='tanh'))
    model.add(keras.layers.Dropout(DROPOUT_RATE))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    print(model.summary())
    return model
main()
