from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model, Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from collections import Counter
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import pickle
import math
import sys
import os


sys.path.append("/lib")
from lib.quantflow.model.models.seq2seq_attn import seq2seq_attention, one_step_attention
from lib.quantflow.model.models.resnet.resnet501D import ResNet50
from lib.quantflow.model.models.lstm_model import generate_lstm
from lib.quantflow.generators import WindowGenerator
from lib.quantflow.model.callback import LossHistory
from lib.quantflow.model.loss import focal_loss
from lib.quantflow.utils import AutoSaver
from lib.quantflow.trade import TradeMode


def train_config(path, model, stride, window, forward, BN, mode, window_norm, algo, dropout, hidden_size, batch_size):
    mod_path = path
    if not os.path.exists(mod_path):
        os.makedirs(mod_path)
    with open(mod_path+'model_info.txt', 'a') as f:
        f.write("DataInfo==================================================\n")
        f.write("stride: %s\nwindow_width: %s\nlabel_width: %s\nshift: %s\nloss: %s\nmode: %s\nModel: %s\n WindowNorm: %s\n" % (str(stride), str(window), str(forward), str(window_norm), str(model.loss[0]),str(mode),str(algo),str(window_norm)))
        f.write("Model Info==================================================\n")
        f.write("BatchNorm: %s\nDropout: %s\nHiddenSize: %s\n" % (str(BN), str(dropout), str(hidden_size)))
        f.write("Model==================================================\n")
        try:
            model.summary(print_fn= f.write)
        except:
            print("Cannot print model architectures!!")
    csv_log = CSVLogger(mod_path+'trn_log.csv')
    log_dir = mod_path+"log/"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    if not os.path.exists(mod_path+"mod/"):
        os.makedirs(mod_path+"mod/")
    checkpointer = AutoSaver(filepath=mod_path+"mod/"+"weights-{epoch:02d}-{val_loss:.2f}.hdf5", load_weights_on_restart=True,
                            monitor='val_loss', mode='auto', verbose=1, save_best_only=True, save_weights_only=True)
    return mod_path, csv_log, tensorboard_callback, checkpointer


########################################################
plot_path = "./results/statics/"
label_column = ['close']
save_path = "./results/"
window_norm = False
dropout_rate = 0.2
mode = 'absolute'
hidden_size = 24
input_width = 156
label_width = 1
shift = 1
batch_size = 8
stride = 1
BN = True
########################################################


# Load Dats
with open("../data/data.pk", 'rb') as f:
    raw = pickle.load(f)
# raw=raw[['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trade_num']]
# Split training and testing set
test = raw['2019-11-30 23:00:00':'2020-02-27 16:00:00']
data = raw['2019-01-01 00:00:00':'2019-11-30 23:00:00']


# 按列归一化数据 normalization
train_mean = data.mean()
train_std = data.std()
data = (data - train_mean) / train_std
test = (test - train_mean) / train_std

# 提琴图 violin plot
# raw_std = (raw - train_mean) / train_std
# raw_std = raw_std.melt(var_name='Column', value_name='Normalized')
# plt.figure(figsize=(12, 6))
# ax = sns.violinplot(x='Column', y='Normalized', data=raw_std)
# _ = ax.set_xticklabels(raw.keys(), rotation=90)

# Trade mode
tMode = TradeMode(rf=0.005, commission=0.005, mode=mode)

# Windows generator
wininst = WindowGenerator(input_width=input_width, label_width=label_width, shift=shift, label_columns=label_column, train_df=data, val_df=test,
                     y_map_fn=tMode.cal_profit, plot_path=plot_path, stride=stride, plot_fn=tMode.plot)
print(wininst)

# Generate datasets
trainset = wininst.train
valset = wininst.val
a = iter(trainset)
b, c = next(a)
# Resnet50
model_name = "LSTM_cls"

model = tf.keras.models.Sequential([
    tf.keras.layers.Input((input_width, len(data.columns))),
    # Shape [batch, time, features] => [batch, time, lstm_units]
    # tf.compat.v1.keras.layers.CuDNNLSTM(24, return_sequences=True),
    # tf.keras.layers.Activation('relu'),
    # tf.keras.layers.Dropout(dropout_rate),
    # Shape = > [batch, time, features]
    # tf.compat.v1.keras.layers.CuDNNLSTM(hidden_size, return_sequences=True),
    # tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.compat.v1.keras.layers.CuDNNLSTM(hidden_size, return_sequences=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(dropout_rate),
    # tf.keras.layers.Dense(units=32),
    # tf.keras.layers.Dropout(dropout_rate),
    # tf.keras.layers.Dense(units=32),
    # tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(units=3, activation='softmax')
])


mod_path = save_path + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")+"/"
model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['acc'])
history = LossHistory(labels=[0, 1, 2], trade_mode=wininst, plot_fn=wininst.plot, save_path=mod_path+'images/')
mod_path, csv_log, tensorboard_callback, checkpointer = train_config(mod_path, model, stride, input_width, label_width, BN, mode, window_norm,
                                                              model_name, dropout_rate, hidden_size, batch_size)

# Train model
model.fit(trainset, epochs=500, validation_data=valset, batch_size=batch_size,
          callbacks=[tensorboard_callback, csv_log, history, checkpointer], shuffle=True)# history, class_weight=class_weights, sample_weight=sample_weights,

for i in range(10):
    wininst.plot(model=model, plot_mode='iter', plot_name='./results/statics/' + str(i) + '.png')

