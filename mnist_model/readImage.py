from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = load_model('model.h5')
model.load_weights('weights1.h5')
#model.summary()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#plt.imshow(x_test[0])
x_train = np.expand_dims(x_train, axis=-1).astype(np.float)/255.0
#x_test = np.expand_dims(x_test, axis=-1).astype(np.float)/255.0
#x_train.shape

validdatagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    zoom_range=0.1
)
validdatagen.fit(x_train)
image = cv2.imread("image/4.jpg",0)
blur = cv2.GaussianBlur(image, (1, 1), 0)
ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
thre = np.expand_dims(th, axis=-1).astype(np.float)/255.0
results = model.predict_generator(
    validdatagen.flow(np.array([thre]), batch_size=1, shuffle=False),
    steps=1
)
y_pred = np.argmax(results, axis=-1)
print(y_pred)
