from tensorflow.keras.callbacks import CSVLogger
import tensorflow as tf
import pickle
import sys
import os
sys.path.append("/lib")
from kerastuner import RandomSearch
from kerastuner.applications.resnet1d import HyperResNet1D
from lib.quantflow.generators import WindowGenerator
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
        model.summary(print_fn= f.write)
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
hidden_size = 64
input_width = 240
label_width = 1
shift = 1
batch_size = 32
stride = 1
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
                     y_map_fn=tMode.cal_profit, plot_path=plot_path, stride=stride, plot_fn=tMode.plot)
print(wininst)
mod_path = save_path + 'hyper_resnet1d/'
# Generate datasets
trainset = wininst.train
valset = wininst.val
hypermodel = HyperResNet1D(input_shape=(input_width, len(data.columns)), classes=3)
tuner = RandomSearch(
                    hypermodel,
                    objective= 'val_loss',
                    max_trials=240,
                    directory=mod_path,
                    project_name='HyperResnet')

tuner.search(trainset,
             epochs=240,
             batch_size=64,
             validation_data=valset)