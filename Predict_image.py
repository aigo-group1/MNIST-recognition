from mnist_detection import detection
from mnist_model import readImage

(iden1, iden2), (date1, date2) = detection.detect_image()

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
print(predict_iden1)
print(predict_date1)
print(predict_iden2)
print(predict_date2)
