import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(x_train.shape)
#plt.imshow(x_train[0])
x_train = np.expand_dims(x_train, axis=-1).astype(np.float32)/255.0
x_test = np.expand_dims(x_test, axis=-1).astype(np.float32)/255.0

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
#model.summary()

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5
)

validdatagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True
)

datagen.fit(x_train)
validdatagen.fit(x_train)

model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train)//32,
                    epochs=20,
                    validation_data=validdatagen.flow(
    x_test, y_test, batch_size=32),
    validation_steps=len(x_test)//32)
save_dir = os.path.join(os.getcwd(), 'mnist_model')
with open(os.path.join(save_dir,'validdatagen.pkl'), 'wb') as pickle_file:
  pickle.dump(validdatagen, pickle_file)
with open(os.path.join(save_dir,'model.pkl'), 'wb') as pickle_file:
  pickle.dump(model, pickle_file)
