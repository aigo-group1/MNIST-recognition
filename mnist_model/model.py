from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers import BatchNormalization
from keras.layers import Add, Concatenate
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def reset_model():
    input = Input(shape=(28, 28, 1))
    x = Conv2D(32, kernel_size=(5, 5), activation='relu')(input)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])
                
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype(np.float32)/255.0
    x_test = np.expand_dims(x_test, axis=-1).astype(np.float32)/255.0

    model.save('Model1.h5')
    model.summary()

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.1,
        channel_shift_range=0.05
    )
    validdatagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True
    )

    datagen.fit(x_train)
    validdatagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                        steps_per_epoch=len(x_train)//32,
                        epochs=20,
                        validation_data=validdatagen.flow(x_test, y_test, batch_size=32),
                        validation_steps=len(x_test)//32)

    model.save_weights('Weights1.h5')

    return model
