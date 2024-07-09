import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D

def build_srcnn_model():
    input_shape = (None, None, 3)  # Dynamic size
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (9, 9), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    outputs = Conv2D(3, (5, 5), activation='linear', padding='same')(x)
    model = Model(inputs, outputs)
    return model
