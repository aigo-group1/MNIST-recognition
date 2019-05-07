#import sys
#imagePath = sys.argv[1].replace('\\','/')

from keras.models import load_model
import keras.backend as K
import cv2
import numpy as np
import pickle

def loadmodel():
    K.clear_session()
    with open('mnist_model/model.pkl', 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    return model

def getvaliddatagen():
    with open('mnist_model/validdatagen.pkl', 'rb') as pickle_file:
        validdatagen = pickle.load(pickle_file)
    return validdatagen

def Prediction(image,model,validdatagen):
    results = model.predict_generator(
        validdatagen.flow(image, batch_size=len(image)//4, shuffle=False),
        steps=4
    )
    y_pred = np.argmax(results, axis=-1)
    return y_pred
