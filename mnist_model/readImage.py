#import sys
#imagePath = sys.argv[1].replace('\\','/')

from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.models import load_model
import keras.backend as K
import cv2
import numpy as np
import matplotlib.pyplot as plt

def loadmodel():
    K.clear_session()
    model = load_model('mnist_model/model (1).h5')
    model.load_weights('mnist_model/weights (1).h5')
    return model

def getvaliddatagen():
    x_train = mnist.load_data()[0][0]
    index = 0
    a = 37
    for i in range(len(x_train)):
        path = "frame/frame "+str(index)+".jpg"
        frame = cv2.imread(path, 0)
        x_train[i] = np.maximum(np.array(x_train[i]), np.array(frame))
        index = (index+a) % 120

    x_train = np.expand_dims(x_train, axis=-1).astype(np.float)/255.0

    validdatagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
    )
    validdatagen.fit(x_train)
    return validdatagen

def Prediction(image,model,validdatagen):
    thre = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    thre = np.expand_dims(image, axis=-1).astype(np.float32)/255.0
    results = model.predict_generator(
        validdatagen.flow(np.array([thre]), batch_size=1, shuffle=False),
        steps=1
    )
    y_pred = np.argmax(results, axis=-1)
    return y_pred
