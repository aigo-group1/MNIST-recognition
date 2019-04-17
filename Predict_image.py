from mnist_detection import detection
from mnist_model import readImage

import sys 

imagePath = sys.argv[1].replace('\\','/')

(identity,date) = detection.detect_image(imagePath)

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



predict_iden = []
predict_date = []

for line in identity:
    predict_line = []
    for image in line:
        predict_line.append(readImage.Prediction(image))
    predict_iden.append(predict_line)

for line in date:
    predict_line = []
    for image in line:
        predict_line.append(readImage.Prediction(image))
    predict_date.append(predict_line)   

pre = []
for iden,date in zip(predict_iden,predict_date):
    for x in iden:
        print("%d" %x,end='')
    print('')
    for y in date:
        print("%d" %y,end='')
    print('')