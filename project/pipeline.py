
import os
from numpy.core.fromnumeric import mean
import pandas as pd
from project.utils import set_gpus
from project.data import get_vent_data
from project.utils import record_split_idx
from project.model import MODEL_MAP

import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold, ShuffleSplit
pd.set_option('display.max_columns', None)
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau


def train_cv(args):
    set_gpus(args['general_config']['gpus'])
    epoch = args['general_config']['num_epochs']
    batch_size = args['data_config']['batch_size']
    X, y = get_vent_data(args)
    kf = KFold(n_splits=5, shuffle=True, random_state=2021)
    epoch_record = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print('-'*15, '>', f'Fold {fold+1}', '<', '-'*15)
        X_train, X_valid = X[train_idx], X[val_idx]
        y_train, y_valid = y[train_idx], y[val_idx]
        record_split_idx(args, train_idx, val_idx, fold)
        model = MODEL_MAP[args['model_config']['model']](X)
        es = EarlyStopping(monitor="val_loss", patience=50, verbose=1, mode="min", restore_best_weights=True)
        checkpoint_filepath = os.path.join(args['general_config']['result_path'], f'models/folds{fold}.hdf5')
        sv = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True,
            save_weights_only=False, mode='auto', save_freq='epoch',
            options=None
        )
        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1)
        model.fit(X_train, y_train, validation_data = (X_valid, y_valid), epochs = epoch, batch_size = batch_size, callbacks = [lr, es, sv])
        epoch_record.append(es.stopped_epoch)
    args['general_config']['num_epochs'] = int(np.median(epoch_record)) - 50
    return args


def train_valid(args):
    set_gpus(args['general_config']['gpus'])
    epoch = args['general_config']['num_epochs']
    batch_size = args['data_config']['batch_size']
    X, y = get_vent_data(args)
    rs = ShuffleSplit(n_splits=1,test_size=.2,random_state=42)
    rs.get_n_splits(X)
    train_idx, val_idx = next(rs.split(X, y))
    X_train, X_valid = X[train_idx], X[val_idx]
    y_train, y_valid = y[train_idx], y[val_idx]
    record_split_idx(args, train_idx, val_idx, fold=0)
    model = MODEL_MAP[args['model_config']['model']](X)
    es = EarlyStopping(monitor="val_loss", patience=50, verbose=1, mode="min", restore_best_weights=True)
    checkpoint_filepath = os.path.join(args['general_config']['result_path'], f'models/folds0.hdf5')
    sv = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=False, mode='auto', save_freq='epoch',
        options=None
    )
    lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1)
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epoch, batch_size=batch_size, callbacks=[lr, es, sv])
    args['general_config']['num_epochs'] = es.stopped_epoch
    return args

def train_all(args):
    set_gpus(args['general_config']['gpus'])
    epoch = args['general_config']['num_epochs']
    batch_size = args['data_config']['batch_size']
    X, y = get_vent_data(args)
    rs = ShuffleSplit(n_splits=1,test_size=.2,random_state=42)
    rs.get_n_splits(X)
    train_idx, val_idx = next(rs.split(X, y))
    X_train, X_valid = X[train_idx], X[val_idx]
    y_train, y_valid = y[train_idx], y[val_idx]
    model = MODEL_MAP[args['model_config']['model']](X)
    checkpoint_filepath = os.path.join(args['general_config']['result_path'], f'models/folds5.hdf5')
    sv = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=False, mode='auto', save_freq='epoch',
        options=None
    )
    lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1)
    model.fit(X, y, validation_data=(X_valid, y_valid), epochs=epoch, batch_size=batch_size, callbacks=[lr, sv])
