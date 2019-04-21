from mnist_detection import detection
from mnist_model import readImage

import sys 

def predict(imagePath,model):

    """
    predict_iden1 = []
    predict_iden2 = []
    for i in range(len(iden1)):
        predict_iden1.append(readImage.Prediction(iden1[i]))
    for i in range(len(iden2)):
        predict_iden2.append(readImage.Prediction(iden2[i]))

    predict_date1 = []
    predict_date2 = []
    for j in range(len(date1)):
        predict_date1.append(readImage.Prediction(date1[j]))   
    for j in range(len(date2)):
        predict_date2.append(readImage.Prediction(date2[j]))

    for x in predict_iden1:
        print("%d" %x,end='')
    print('')
    for x in predict_date1:
        print("%d" %x,end='')
    print('')
    for x in predict_iden2:
        print("%d" %x,end='')
    print('')
    for x in predict_date2:
        print("%d" %x,end='')       
    """
    (identity,date) = detection.detect_image(imagePath)

    validdatagen = readImage.getValiddatagen()

    predict_iden = []
    predict_date = []
    for line in identity:
        predict_line = []
        for image in line:
            predict_line.append(readImage.Prediction(image,model,validdatagen))
        predict_iden.append(predict_line)

    for line in date:
        predict_line = []
        for image in line:
            predict_line.append(readImage.Prediction(image,model,validdatagen))
        predict_date.append(predict_line)
           
    return (predict_iden,predict_date)


