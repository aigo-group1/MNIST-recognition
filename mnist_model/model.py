import pickle
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers import BatchNormalization, Activation, Dropout
from keras.layers import Add, Concatenate
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 padding='same', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

#load data train              
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_test.shape)

index = 0
a = 37
for i in range(len(x_train)):
  path = "frame/frame "+str(index)+".jpg"
  frame = cv2.imread(path, 0)
  x_train[i] = np.maximum(np.array(x_train[i]), np.array(frame))
  index = (index+a) % 120

index = 0
b = 43
for i in range(len(x_test)):
  path = "frame/frame "+str(index)+".jpg"
  frame = cv2.imread(path, 0)
  x_test[i] = np.maximum(np.array(x_test[i]), np.array(frame))
  index = (index+a) % 120

#print(x_train[0])
x_train = np.expand_dims(x_train, axis=-1).astype(np.float32)/255.0
x_test = np.expand_dims(x_test, axis=-1).astype(np.float32)/255.0
plt.imshow(x_test[65].reshape(28, 28))

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.1,
    channel_shift_range=0.05
)
validdatagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True
)

datagen.fit(x_train)
validdatagen.fit(x_train)

with open('validdatagen.pkl', 'wb') as pickle_file:
  pickle.dump(validdatagen, pickle_file)

model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train)//32,
                    epochs=5,
                    validation_data=validdatagen.flow(
    x_test, y_test, batch_size=32),
    validation_steps=len(x_test)//32)

with open('model.pkl', 'wb') as pickle_file:
    pickle.dump(model, pickle_file)