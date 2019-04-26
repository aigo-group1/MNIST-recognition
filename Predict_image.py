from mnist_detection import detection
from mnist_model import readImage
import numpy 
import pickle
import cv2

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


#with open('mnist_model/model.pkl', 'rb') as pickle_file:
#    model = pickle.load(pickle_file)
#with open('mnist_model/validdatagen.pkl', 'rb') as pickle_file:
#    validdatagen = pickle.load(pickle_file)
#listiden, listdate = predict('mnist_detection/test3.jpg', model, validdatagen)
#for line in listiden:
#    print(line)
#for line in listdate:
#    print(line)
