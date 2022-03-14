from tensorflow import keras
from tensorflow.python.keras.layers import Conv2D

def model_vdcnn_functional(num_classes):
    inputs = keras.Input(shape=(2048,300, 1))
    x = Conv2D(8, (3, 3), name='conv_1')(inputs)
    x = Conv2D(16, (5, 5), name='conv_2')(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(32, (5, 5), name='conv_3')(x)
    x = Conv2D(64, (7, 7), name='conv_4')(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(x)

    x = Conv2D(128, (7, 7), name='conv_5')(x)

    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D(strides=(2, 2))(x)
    x = keras.layers.Dense(2048)(x)
    x = keras.layers.Dense(256)(x)
    preds = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, preds)
    return model

model = model_vdcnn_functional(7)
model.summary()