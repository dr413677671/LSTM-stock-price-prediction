from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
import time
import os
import re

class AutoSaver(tf.keras.callbacks.ModelCheckpoint):
    def on_train_begin(self, logs=None):
        if self.load_weights_on_restart:
            filepath_to_load = (
                self._get_most_recently_modified_file_matching_pattern(self.filepath))
            if (filepath_to_load is not None and
                    self._checkpoint_exists(filepath_to_load)):
                try:
                    # `filepath` may contain placeholders such as `{epoch:02d}`, and
                    # thus it attempts to load the most recently modified file with file
                    # name matching the pattern.
                    self.model.load_weights(filepath_to_load)
                except (IOError, ValueError) as e:
                    raise ValueError('Error loading file from {}. Reason: {}'.format(
                        filepath_to_load, e))

    def on_epoch_end(self, epoch, logs=None):
        super(AutoSaver, self).on_epoch_end(epoch=epoch, logs=logs)
        filepath = self._get_file_path(epoch, logs)
        info_path = os.path.join(os.path.split(filepath)[0], 'cpkt_int.txt')
        with open(info_path, 'w') as f:
            f.write(filepath)

def scale_columns(data,scalars=None):
    data_new = pd.DataFrame([])
    if not scalars:
        scalars = []
        for column in data.columns:
            scalar = MinMaxScaler()
            scalar.fit(np.expand_dims(data[column],axis=1))
            scalars.append(scalar)
            data_new[column]=scalar.transform(np.expand_dims(data[column],axis=1)).reshape(-1)
    else:
        for index, column in enumerate(data.columns):
            data_new[column]=scalars[index].transform(np.expand_dims(data[column],axis=1)).reshape(-1)
    return data_new, scalars


def engtime_timestamp(time_sj):                #传入单个时间比如'2019-8-01 00:00:00'，类型为str
    if re.search(':',time_sj):
        data_sj = time.strptime(time_sj,"%Y-%m-%d %H:%M:%S")       #定义格式
        time_int = int(time.mktime(data_sj))
    else:
        data_sj = time.strptime(time_sj,"%Y-%m-%d")       #定义格式
        time_int = int(time.mktime(data_sj))
    return time_int  

def chstime_timestamp(time_sj):                #传入单个时间比如'2019-8-01 00:00:00'，类型为str
    data_sj = time.strptime(time_sj,u"%Y年%m月%d日")       #定义格式
    time_int = int(time.mktime(data_sj))
    return time_int  