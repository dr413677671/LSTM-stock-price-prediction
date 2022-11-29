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


sys.path.append("../lib")
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
        f.write("stride: %s\nwindow: %s\nforward: %s\nloss: %s\nmode: %s\nModel: %s\n WindowNorm: %s\n" % (str(stride), str(window), str(forward), str(model.loss[0]),str(mode),str(algo),str(window_norm)))
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
hidden_size = 200
input_width = 24
label_width = 24
batch_size = 32
stride = 1
shift = 1
BN = True
########################################################


# Load Dats
with open("./data/data.pk", 'rb') as f:
    raw = pickle.load(f)

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
tMode = TradeMode(rf=0.002, commission=0.005, mode=mode)

# Windows generator
wininst = WindowGenerator(input_width=input_width, label_width=label_width, shift=shift, label_columns=label_column, train_df=data, val_df=test,
                      plot_path=plot_path, stride=stride)
print(wininst)

# Generate datasets
trainset = wininst.train
valset = wininst.val

# DEBUG WIN GEN FUN
# example_window = tf.stack([np.array(data[:wininst.total_window_size]),
#                            np.array(data[100:100+wininst.total_window_size]),
#                            np.array(data[200:200+wininst.total_window_size])])
#
# example_inputs, example_labels = wininst.split_window(example_window)
# print('All shapes are: (batch, time, features)')
# print(f'Window shape: {example_window.shape}')
# print(f'Inputs shape: {example_inputs.shape}')
# print(f'labels shape: {example_labels.shape}')

# ResLSTM
model_name = "ResLSTM"
class ResidualWrapper(tf.keras.Model):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def call(self, inputs, *args, **kwargs):
    delta = self.model(inputs, *args, **kwargs)
    selected_inputs = tf.stack(
        [inputs[:, :, wininst.column_indices[name]] for name in wininst.label_columns],
        axis=-1)
    # The prediction for each timestep is the input
    # from the previous time step plus the delta
    # calculated by the model.
    return selected_inputs + delta

model = ResidualWrapper(
 tf.keras.models.Sequential([
    tf.keras.layers.Input((input_width, len(data.columns))),
    # Shape [batch, time, features] => [batch, time, lstm_units]
    # tf.compat.v1.keras.layers.CuDNNLSTM(24, return_sequences=True),
    # tf.keras.layers.Activation('relu'),
    # tf.keras.layers.Dropout(dropout_rate),
    tf.compat.v1.keras.layers.CuDNNLSTM(24, return_sequences=True),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(dropout_rate),

    # tf.keras.layers.Dense(units=32),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(units=32),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Activation('relu'),

    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1, activation='linear', kernel_initializer=tf.initializers.zeros)
]))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
mod_path = save_path + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")+"/"
history = LossHistory(labels=[0, 1, 2], trade_mode=wininst, plot_fn=wininst.plot, save_path=mod_path+'images/')
mod_path, csv_log, tensorboard_callback, checkpointer = train_config(mod_path, model, stride, input_width, label_width, BN, mode, window_norm,
                                                              model_name, dropout_rate, hidden_size, batch_size)

# Train model
model.fit(trainset, epochs=500, validation_data=valset, batch_size=batch_size,
          callbacks=[tensorboard_callback, csv_log, history, checkpointer], shuffle=True)# history, class_weight=class_weights, sample_weight=sample_weights,

for i in range(10):
    wininst.plot(model=model, plot_mode='iter', plot_name='./results/statics/' + str(i) + '.png')

# Each element is an (inputs, label) pair
# w1.train.element_spec
