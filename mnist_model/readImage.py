from keras.models import load_model
import keras.backend as K
import cv2
import numpy as np
import pickle
import os

save_dir = os.path.join(os.getcwd(), 'mnist_model')
def loadmodel():
    K.clear_session()
    with open(os.path.join(save_dir, 'model.pkl'), 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    return model

def getvaliddatagen():
    with open(os.path.join(save_dir, 'validdatagen.pkl'), 'rb') as pickle_file:
        validdatagen = pickle.load(pickle_file)
    return validdatagen

def Prediction(image,model,validdatagen):
    image = np.expand_dims(image, axis=-1).astype(np.float32)/255.0
    results = model.predict_generator(
        validdatagen.flow(image, batch_size=len(image)//4, shuffle=False),
        steps=4
    )
    y_pred = np.argmax(results, axis=-1)
    return y_pred
