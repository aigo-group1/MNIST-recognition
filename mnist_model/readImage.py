#import sys
#imagePath = sys.argv[1].replace('\\','/')

from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = load_model('mnist_model/Model1.h5')
model.load_weights('mnist_model/Weights1.h5')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
for image in x_train:
      image[0:2,:]=image[26:28, :]=image[:, 26:28]=image[:, 0:2]=255
x_train = np.expand_dims(x_train, axis=-1).astype(np.float)/255.0

validdatagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    zoom_range=0.1
)
validdatagen.fit(x_train)

def Prediction(image):
    thre = np.expand_dims(image, axis=-1).astype(np.float32)/255.0
    results = model.predict_generator(
        validdatagen.flow(np.array([thre]), batch_size=1, shuffle=False),
        steps=1
    )
    y_pred = np.argmax(results, axis=-1)
    return y_pred

