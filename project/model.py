from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Bidirectional, LSTM, GRU
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding
from tensorflow.keras.layers import Concatenate, Add, GRU
from tensorflow.keras.models import Model
from keras.layers import concatenate
from tcn import TCN

def get_model_v1(X):
    model= keras.models.Sequential([
        Input(shape=X.shape[-2:]),
        Bidirectional(LSTM(1024, return_sequences=True)),
        Bidirectional(LSTM(512, return_sequences=True)),
        Bidirectional(LSTM(256, return_sequences=True)),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dense(128, activation='selu'),
        #Dropout(0.05),
        Dense(64, activation='selu'),
        Dense(1),
    ])
    model.compile(optimizer = "adam", loss = "mae")
    return model


def get_model_v2(X):
    x_input = Input(shape=(X.shape[-2:]))
    x1 = Bidirectional(LSTM(units=1024, return_sequences=True))(x_input)
    x2 = Bidirectional(LSTM(units=512, return_sequences=True))(x1)
    x3 = Bidirectional(LSTM(units=256, return_sequences=True))(x2)
    x4 = Bidirectional(LSTM(units=128, return_sequences=True))(x3)
    z2 = Bidirectional(GRU(units=256, return_sequences=True))(x2)
    z3 = Bidirectional(GRU(units=128, return_sequences=True))(Add()([x3, z2]))
    z4 = Bidirectional(GRU(units=64, return_sequences=True))(Add()([x4, z3]))
    x = Concatenate(axis=2)([x4, z2, z3, z4])
    x = Dense(units=128, activation='selu')(x)
    x_output = Dense(units=1)(x)
    model = Model(inputs=x_input, outputs=x_output, 
                  name='DNN_Model')
    model.compile(optimizer = "adam", loss = "mae")
    return model


def get_model_v3(X):
    model = keras.models.Sequential([
        Input(shape=X.shape[-2:]),
        Bidirectional(LSTM(1024, return_sequences=True, activation='relu')),
        Bidirectional(LSTM(512, return_sequences=True, activation='relu')),
        Bidirectional(LSTM(256, return_sequences=True, activation='relu')),
        Bidirectional(LSTM(128, return_sequences=True, activation='relu')),
        Bidirectional(LSTM(64, return_sequences=True, activation='relu')),
        Dense(64, activation='gelu'),
        Dropout(0.05),
        Dense(32, activation='gelu'),
        Dense(1)
    ])
    model.compile(optimizer = "adam", loss = "mae")
    return model


def get_model_v4(X):
    x_input = Input(shape=(X.shape[-2:]))
    x = TCN(64, return_sequences=True, activation = 'gelu')(x_input)
    x_output = Dense(1, activation='gelu')(x) 
    model = Model(inputs=x_input, outputs=x_output, 
                  name='TCN_Model')
    model.compile(optimizer = "adam", loss = "mae")
    return model


def get_model_v5(X):
    model = keras.models.Sequential([
        Input(shape=X.shape[-2:]),
        Bidirectional(LSTM(512, return_sequences=True)),
        Dropout(0.05),
        Bidirectional(LSTM(256, return_sequences=True)),
        Dropout(0.05),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.05),
        Dense(128, activation='gelu'),
        Dropout(0.05),
        Dense(64, activation='gelu'),
        Dense(1)
    ])
    model.compile(optimizer = "adam", loss = "mae")
    return model


def get_model_v6(X):
    model = keras.models.Sequential([
        Input(shape=X.shape[-2:]),
        TCN(64, return_sequences=True, activation = 'gelu'),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.05),
        Dense(64, activation='gelu'),
        Dropout(0.05),
        Dense(1)
    ])
    model.compile(optimizer = "adam", loss = "mae")
    return model


def get_model_v7(X):
    model = keras.models.Sequential([
        Input(shape=X.shape[-2:]),
        TCN(64, return_sequences=True, activation = 'gelu'),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.05),
        Bidirectional(LSTM(32, return_sequences=True)),
        Dropout(0.05),
        Dense(32, activation='selu'),
        Dropout(0.05),
        Dense(1)
    ])
    model.compile(optimizer = "adam", loss = "mae")
    return model


def get_model_v8(X):
    model = keras.models.Sequential([
        Input(shape=X.shape[-2:]),
        TCN(128, return_sequences=True, activation = 'gelu'),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.05),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.05),
        Bidirectional(LSTM(32, return_sequences=True)),
        Dropout(0.05),
        Dense(32, activation='selu'),
        Dropout(0.05),
        Dense(1)
    ])
    model.compile(optimizer = "adam", loss = "mae")
    return model


def get_model_v9(X):
    model= keras.models.Sequential([
        Input(shape=X.shape[-2:]),
        Bidirectional(LSTM(1024, return_sequences=True)),
        Bidirectional(LSTM(512, return_sequences=True)),
        Bidirectional(LSTM(256, return_sequences=True)),
        Bidirectional(LSTM(128, return_sequences=True)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.05),
        Dense(64, activation='gelu'),
        Dropout(0.05),
        Dense(32, activation='gelu'),
        Dense(1),
    ])
    model.compile(optimizer = "adam", loss = "mae")
    return model


MODEL_MAP = {'v1': get_model_v1,
             'v2': get_model_v2,
             'v3': get_model_v3,
             'v4': get_model_v4,
             'v5': get_model_v5, 
             'v6': get_model_v6,
             'v7': get_model_v7,
             'v8': get_model_v8,
             'v9': get_model_v9}

