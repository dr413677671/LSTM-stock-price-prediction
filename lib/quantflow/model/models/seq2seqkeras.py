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