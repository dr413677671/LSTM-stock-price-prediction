from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Reshape, BatchNormalization, Lambda, Bidirectional
from tensorflow.keras.layers import RepeatVector, Concatenate, Dot, Activation, Dropout

from ..activation import softmax

def one_step_attention(a, s_prev, repeator, concatenator, densor, activator, dotor):
    s_prev = repeator(s_prev)
    concat = concatenator([s_prev, a])
    e = densor(concat)
    alphas = activator(e)
    context =  dotor([alphas, a])
    return context

def seq2seq_attention(output_size=3, after_day=1, dropout_rate=0,
                      input_shape=(20, 1), time_step=20, BN=False, output_bias=None,hidden_size=200):
    # Define the inputs of your model with a shape (Tx, feature)
    X = Input(shape=input_shape)
    s0 = Input(shape=(hidden_size, ), name='s0')
    c0 = Input(shape=(hidden_size, ), name='c0')
    s = s0
    c = c0
    if BN:
        BATCHN = BatchNormalization(axis=1)(X)
    else:
        BATCHN = X
    # Initialize empty list of outputs
    all_outputs = []

    # Encoder: pre-attention LSTM
    # encoder_outputs = LSTM(units=200, return_state=False, return_sequences=True, name='encoder')(X)
    A = LSTM(units=hidden_size, return_state=False, return_sequences=True, name='encoder')(BATCHN)
    encoder_outputs=Dropout(dropout_rate)(A)
    # Decoder: post-attention LSTM
    decoder = LSTM(units=hidden_size, return_state=True, name='decoder')
    # Output
    decoder_output = Dense(units=output_size, activation='softmax', name='output')
    model_output = Reshape((1, output_size))

    # Attention
    repeator = RepeatVector(time_step)
    concatenator = Concatenate(axis=-1)
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
        densor = Dense(1, activation = "relu", bias_initializer=output_bias)
    else:
        densor = Dense(1, activation = "relu")
    activator = Activation(softmax, name='attention_weights')
    dotor =  Dot(axes = 1)

    for t in range(after_day):
        context = one_step_attention(encoder_outputs, s, repeator, concatenator, densor, activator, dotor)

        a, s, c = decoder(context, initial_state=[s, c])

        outputs = decoder_output(a)
        outputs = model_output(outputs)
        all_outputs.append(outputs)

    all_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
    model = Model(inputs=[X, s0, c0], outputs=all_outputs)
    return model