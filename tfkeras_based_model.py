import keras
from keras import datasets, layers, models
from keras.layers import Dense, Conv2D, Flatten, Dropout, Input
from keras.optimizers import Adam
from keras.models import Model

def TFKNvidiaModel():
    inputs = Input(shape=(120, 180, 3))
    conv_1 = Conv2D(24, (5, 5), strides=(2,2), activation='elu')(inputs)
    conv_2 = Conv2D(36, (5, 5), strides=(2,2), activation='elu')(conv_1)
    conv_3 = Conv2D(48, (5, 5), strides=(2,2), activation='elu')(conv_2)
    conv_4 = Conv2D(64, (3, 3), activation='elu')(conv_3)
    conv_5 = Dropout(0.2)(conv_4)
    conv_6 = Conv2D(64, (3, 3), activation='elu')(conv_5)

    flatten = Flatten()(conv_6)
    drop = Dropout(0.5)(flatten)

    dense1 = Dense(100, activation='elu')(drop)
    dense2 = Dense(50, activation='elu')(dense1)
    dense3 = Dense(10, activation='elu')(dense2)
    output_1 = Dense(1, name='output_1')(dense3)

    dense1_1 = Dense(100, activation='elu')(drop)
    dense2_1 = Dense(50, activation='elu')(dense1_1)
    dense3_1 = Dense(10, activation='elu')(dense2_1)
    output_2 = Dense(1, name='output_2')(dense3_1)
    optimizer = Adam(lr=1e-3)
    model = Model(inputs=inputs, outputs=[output_1, output_2])
    model.compile(optimizer= optimizer, loss={'output_1': 'mean_squared_error', 'output_2': 'mean_squared_error'}, loss_weights = [1, 1], metrics=['accuracy', 'accuracy'])
    return model
