import sys
#imagePath = sys.argv[1].replace('\\','/')

sys.path.insert(0,'D:\\MNIST-recognition\\mnist_detection')

print(sys.path)

from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

<<<<<<< HEAD
model = load_model('/MNIST-recognition/mnist_model/Model.h5')
model.load_weights('/MNIST-recognition/mnist_model/Weights.h5')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#plt.imshow(x_test[0])
x_train = np.expand_dims(x_train, axis=-1).astype(np.float)/255.0
#x_test = np.expand_dims(x_test, axis=-1).astype(np.float)/255.0
#x_train.shape
=======
model = load_model('mnist_model/Model.h5')
model.load_weights('mnist_model/Weights.h5')
#model.summary()
>>>>>>> 44fb8a6c2f6940934a13a0e2b6d8212fb0e5881d

validdatagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    zoom_range=0.1
)
validdatagen.fit(np.expand_dims(mnist.load_data()[0][0], axis=-1).astype(np.float)/255.0)

def Prediction(image):
    blur = cv2.GaussianBlur(image, (1, 1), 0)
    th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    thre = np.expand_dims(th, axis=-1).astype(np.float)/255.0
    results = model.predict_generator(
        validdatagen.flow(np.array([thre]), batch_size=1, shuffle=False),
        steps=1
    )
    y_pred = np.argmax(results, axis=-1)
    return y_pred

<<<<<<< HEAD
import detection as detec

imagedata = detec.detectimage('D:/MNIST-recognition/server/upload/image2.jpg')

print(imagedata)



=======

#image = cv2.imread("mnist_model/image/4.jpg", 0)
#print(Prediction(iden1[2]))
>>>>>>> 44fb8a6c2f6940934a13a0e2b6d8212fb0e5881d
