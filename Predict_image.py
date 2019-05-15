from mnist_detection import detection
from mnist_model import readImage
import numpy as np
import pickle
import cv2
import os

def predict(imagePath,model,validdatagen):
    listimage = detection.detect_image(imagePath)
    list_predict = readImage.Prediction(listimage, model, validdatagen)
    listiden = []
    listdate = []
    i = 0
    while i < len(list_predict):
        listiden.append(list_predict[i:i+12])
        i+=12
        listdate.append(list_predict[i:i+8])
        i+=8
    return listiden,listdate

"""
dir1 = os.path.join(os.getcwd(), 'mnist_model')
with open(os.path.join(dir1, 'model.pkl'), 'rb') as pickle_file:
    model = pickle.load(pickle_file)
with open(os.path.join(dir1, 'validdatagen.pkl'), 'rb') as pickle_file:
    validdatagen = pickle.load(pickle_file)
dir2 = os.path.join(os.getcwd(), 'mnist_detection')
listiden, listdate = predict(os.path.join(dir2, 'image1.jpg'), model, validdatagen)
for line in listiden:
    print(line)
for line in listdate:
    print(line)
"""