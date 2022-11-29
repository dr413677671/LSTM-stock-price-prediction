from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization,Dropout,Flatten, Input
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from ..activation import LayerNormalization

def generate_lstm(input_shape,output_unit=3,hidden_size=50,dropout_rate=0.2,normalization=False, output_activation='softmax',use_cuda=False):
    model = Sequential()
    #     model.add(Dense(units = 132,activation='relu'))
    #     model.add(Dropout(0.2))
    #     model.add(Dense(units = 132,activation='relu'))
    #     model.add(Dropout(0.2))
    model.add(Input(input_shape))
    if normalization == 'BN':
        model.add(BatchNormalization(axis=-1))
    if use_cuda:
        model.add(CuDNNLSTM(units = hidden_size, return_sequences = True))
        if normalization == 'BN':
            model.add(BatchNormalization(axis=-1))
        elif normalization == 'LN':
            model.add(LayerNormalization())
        model.add(Dropout(dropout_rate))
        model.add(CuDNNLSTM(units = hidden_size, return_sequences = True))
        if normalization == 'BN':
            model.add(BatchNormalization(axis=-1))
        elif normalization == 'LN':
            model.add(LayerNormalization())
        model.add(Dropout(dropout_rate))
        model.add(CuDNNLSTM(units = hidden_size, return_sequences = False))
    else:
        model.add(LSTM(units = hidden_size,return_sequences = True,activation='relu'))
        if normalization == 'BN':
            model.add(BatchNormalization(axis=-1))
        elif normalization == 'LN':
            model.add(LayerNormalization())
        model.add(Dropout(0.2))
        model.add(LSTM(units = hidden_size,return_sequences = False,activation='relu'))
        if normalization == 'BN':
            model.add(BatchNormalization(axis=-1))
        elif normalization == 'LN':
            model.add(LayerNormalization())
        model.add(Dropout(0.2))
    # model.add(Dense(units = 32,activation='relu'))
    model.add(Flatten())
    model.add(Dense(units = output_unit, activation=output_activation))
    return model
    


if __name__=='__main__':
    optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
    model.fit_generator(train_generator,epochs=24,                    
                    workers=1, 
                    validation_data=test_generator,
                    validation_steps=10,
                    steps_per_epoch= (len(data) - window)/(batch*stride),
                    use_multiprocessing=False, 
                    shuffle=False)