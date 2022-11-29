import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# 窗口生成函数
class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df=None, test_df=None, stride=1, plot_fn=None,
                 label_columns=None, y_map_fn=None, batchsize=32, plot_path = ""):
        # Store the raw data.
        self.train_df = train_df
        if val_df is not None:
            self.val_df = val_df
        if test_df is not None:
            self.test_df = test_df

        # Label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        self.y_map_fn = y_map_fn
        self.stride = stride
        self.plot_fn=plot_fn
        # Train parameters.
        self.batchsize = batchsize
        # Output paths
        self.plot_path = plot_path
        self.iter_val = None

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]  # (None, None, feats)
        if self.label_columns is not None:
            if self.y_map_fn is not None:
                labels = tf.stack(
                    [self.y_map_fn(inputs[:, :, self.column_indices[name]], labels[:, :, self.column_indices[name]]) for
                     name in self.label_columns],
                    axis=-1)

            else:
                labels = tf.stack(
                    [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                    axis=-1)
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        if self.y_map_fn is None:
            labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def split_window_test(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]  # (None, None, feats)
        labels1 = None
        labels2 = None
        if self.label_columns is not None:
            if self.y_map_fn is not None:
                labels1 = tf.stack(
                    [self.y_map_fn(inputs[:, :, self.column_indices[name]], labels[:, :, self.column_indices[name]]) for
                     name in self.label_columns],
                    axis=-1)

                labels2 = tf.stack(
                    [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                    axis=-1)
            else:
                labels1 = tf.stack(
                    [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                    axis=-1)
                labels2 = tf.stack(
                    [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                    axis=-1)
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        if self.y_map_fn is None:
            labels1.set_shape([None, self.label_width, None])
        return inputs, labels1, labels2

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data, targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.stride, shuffle=True,
            batch_size=self.batchsize)
        ds = ds.map(self.split_window, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
        return ds

    def make_testset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data, targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.stride, shuffle=False,
            batch_size=self.batchsize)
        ds = ds.map(self.split_window_test, num_parallel_calls=tf.data.AUTOTUNE) .prefetch(tf.data.AUTOTUNE)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_testset(self.val_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            if self.iter_val is None:
                result = next(iter(self.test))
            else:
                result = next(self.iter_val)
            # And cache it for next time
            self._example = result
        return result

    def plot(self, model=None, plot_col='close', max_subplots=8, plot_fn=None, plot_name="", plot_mode='same'):
        # Customized plot_fn
        if plot_fn is not None:
            self.plot_fn = plot_fn
        # Whether to iter samples when plotting
        if plot_mode == 'same':
            self.iter_val = None
        elif plot_mode == 'iter':
            self._example = None
            if self.iter_val is None:
                self.iter_val = iter(self.test)
        # Get samples
        inputs, signal, labels = self.example
        fig = plt.figure(figsize=(12, 10))
        plot_col_index = self.column_indices[plot_col]
        # Find sample size fro plotting
        max_n = min(max_subplots, len(inputs))
        # Iter sample in batch
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if len(self.label_columns) >= 0:
                label_col_index = self.column_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index
            if label_col_index is None:
                continue

            if self.plot_fn is not None:  #######################
                if model is not None:
                    predictions = model(tf.expand_dims(inputs[n, :, :], axis=0))
                    y = predictions[0, :]
                    if self.label_indices.shape[0] != y.shape[0]:
                        y = [np.argmax(y)]
                    self.plot_fn(x=np.concatenate(([self.input_indices[-1]], self.label_indices)),
                            y=np.concatenate(
                                (np.array([inputs[n, -1, label_col_index]]), np.array(labels[n, :]).reshape(-1))),
                            signals=y)
                else:
                    self.plot_fn(x=np.concatenate(([self.v[-1]], self.label_indices)),
                        y=np.concatenate((np.array([inputs[n, -1, label_col_index]]), np.array(labels[n, :]).reshape(-1))),
                        signals=signal[n, :])
            else:
                plt.scatter(self.label_indices, labels[n, :, -1],
                            edgecolors='k', label='Labels', c='#2ca02c', s=64)
                if model is not None:
                    predictions = model(tf.expand_dims(inputs[n, :, :], axis=0))
                    y = predictions[0, :]
                    if self.label_indices.shape[0] != y.shape[0]:
                        y = [np.argmax(y)]
                    plt.scatter(self.label_indices, y,
                                marker='X', edgecolors='k', label='Predictions',
                                c='#ff7f0e', s=64)
            if n == 0:
                plt.legend()
        if plot_name == "":
            plt.savefig(self.plot_path + 'example_' + str(n) + '.png')
        else:
            plt.savefig(plot_name)
        plt.xlabel('Time [h]')
        plt.close(fig)


