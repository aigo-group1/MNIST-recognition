from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = load_model('MNIST-recognition/mnist_model/Model.h5')
model.load_weights('MNIST-recognition/mnist_model/Weights.h5')
#model.summary()

validdatagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    zoom_range=0.1
)
validdatagen.fit(np.expand_dims(mnist.load_data()[0][0], axis=-1).astype(np.float)/255.0)

def Prediction(path):
    image = cv2.imread(path,0)
    blur = cv2.GaussianBlur(image, (1, 1), 0)
    th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    thre = np.expand_dims(th, axis=-1).astype(np.float)/255.0
    results = model.predict_generator(
        validdatagen.flow(np.array([thre]), batch_size=1, shuffle=False),
        steps=1
    )
    y_pred = np.argmax(results, axis=-1)
    return y_pred
#print(Prediction("MNIST-recognition/mnist_model/image/7.jpg"))
