from mnist_detection import detection
from mnist_model import readImage
import numpy as np
#import sys 

model = readImage.loadmodel()

def predict(imagePath,model):
    listimage = detection.detect_image(imagePath)
    validdatagen = readImage.getvaliddatagen()
    predictlist = []
    for line in listimage:
        linepre = []
        for image in line:
            linepre.append(readImage.Prediction(image,model,validdatagen))  
        predictlist.append(linepre)
    listiden = []
    listdate = []
    for i in range(len(predictlist)):
        if i % 2==0:
            listiden.append(predictlist[i])
        else:
            listdate.append(predictlist[i])
    return listiden,listdate


(listiden, listdate) = predict("mnist_detection/test3.jpg", model)
for line in listiden:
    print(line)
for line in listdate:
    print(line)
