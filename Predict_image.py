from mnist_detection import detection
from mnist_model import readImage

#import sys 

#imagePath = sys.argv[1].replace('\\','/')

list_image = detection.detect_image("mnist_detection/test3.jpg")
prediction = []
for line_image in list_image:
    predict_line = []
    for image in line_image:
        predict_line.append(readImage.Prediction(image))
    prediction.append(predict_line)

for line in prediction:
    print(line)
    print("")