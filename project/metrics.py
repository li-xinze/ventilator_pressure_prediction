
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K



class Vent_MAE(Callback):
    def __init__(self, validation_data=()):
        super(Callback, self).__init__()
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X_val)
        w = 1 - tf.expand_dims(self.X_val[:, :, 2], axis=2) - 1
        mae = w * K.abs(self.y_val - y_pred)
        mae = K.sum(mae) / K.sum(w)
        w_ = 1 - w
        mae_ = w_ * K.abs(self.y_val - y_pred)
        mae_ = K.sum(mae_) / K.sum(w_)
        print("evaluation - epoch: {:d} - score: {:.6f} - {:.6f}".format(epoch, mae, mae_))