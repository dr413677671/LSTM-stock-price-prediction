from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.callbacks import Callback
from collections import Counter
import pandas as pd
import numpy as np
import os


def cal_classweight(y):
    weights=[]
    total = np.sum(list(Counter(y_train_sig).values()))
    for cls in range(len(list(Counter(y_train_sig)))):
        weight = (1 /Counter(y_train_sig)[cls])*(total)/len(list(Counter(y_train_sig)))
        weights.append(weight)
    return weights


class LossHistory(Callback):
    def __init__(self, labels=[0, 1], trade_mode=None, plot_fn=None, save_path=None, validation_data=None):
        super(LossHistory, self).__init__()
        self.save_path = save_path
        self.labels = labels
        self.trade_mode = trade_mode
        self.val_f1s = []
        self.val_recalls = []
        self.validation_data = validation_data
        self.val_precisions = []
        self.plot_fn = plot_fn
        self.metrics = pd.DataFrame(
            columns=['F1', 'F1_micro', 'F1_macro', 'F1_weighted', 'Precision', 'Precision_micro',
                     'Precision_macro', 'Precision_weighted', 'Recall', 'Recall_micro', 'Recall_macro',
                     'Recall_weighted'])

    # def on_train_begin(self, logs={}):
    #     self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        """F1, precision and recall"""
        _val_f1 = []
        _val_recall = []
        _val_precision = []
        if self.validation_data is not None:
            val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
            val_predict = [np.argmax(i) for i in val_predict]
            val_targ = self.validation_data[1]
            val_targ = [np.argmax(i) for i in val_targ]
            for average in [None, 'micro', 'macro', 'weighted']:
                _val_f1.append(f1_score(val_targ, val_predict, average=average, labels=self.labels))
                _val_recall.append(recall_score(val_targ, val_predict, average=average, labels=self.labels))
                _val_precision.append(precision_score(val_targ, val_predict, average=average, labels=self.labels))
            self.metrics = self.metrics.append(
                {'F1': _val_f1[0], 'F1_micro': _val_f1[1], 'F1_macro': _val_f1[2], 'F1_weighted': _val_f1[3],
                 'Precision': _val_precision[0], 'Precision_micro': _val_precision[1],
                 'Precision_macro': _val_precision[2], 'Precision_weighted': _val_precision[3],
                 'Recall': _val_recall[0], 'Recall_micro': _val_recall[1],
                 'Recall_macro': _val_recall[2], 'Recall_weighted': _val_recall[3]}, ignore_index=True)
            print('-val_f1: %s --val_precision: %s --val_recall: %s' % (np.array(_val_f1[0]).round(4), np.array(_val_precision[0]).round(4), np.array(_val_recall[0]).round(4)))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.plot_fn(model=self.model, plot_name=self.save_path+str(epoch)+'.png')
        return self.metrics

    def on_train_end(self, logs=None):
        return self.metrics

    def return_metrics(self):
        return self.metrics
