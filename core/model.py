from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, CuDNNLSTM, Conv3D
from keras.layers import Input, Dense, Activation, Flatten, Dropout
from keras.layers import TimeDistributed, Lambda
from keras.layers import Reshape, Permute, BatchNormalization, LeakyReLU
from keras.layers.merge import add, concatenate
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.regularizers import *
from keras.layers.noise import GaussianNoise
from keras.models import Model
from custom_activation import ParametricSoftplus, lateral_layer

def LNL1(weight_matrix):

    # the threshold for each pixel depends on the strength for neighboring pixels
    return K.abs(weight_matrix) / (0.01 + K.sum(K.abs(weight_matrix)))

def cnn_model(bc_num=128, bc_size=25, rolling_window=20, l1_Wreg=0, cell_num=80):
    input_shape = (90, 90, rolling_window)
    l2_reg = 1e-3
    l1_reg = 1e-3
    #l1_Wreg = 1e-3
    gau_sigma = 0.1
    #cell_num = 80

    inputs = Input(name='the_input', shape=input_shape, dtype='float32')

    # convolutional layer
    inner = Conv2D(bc_num, (bc_size, bc_size), 
            padding='valid', name='conv1', 
            #kernel_regularizer=l2(l2_reg))(inputs)
            kernel_regularizer=l1_l2(l1=l1_Wreg, l2=l2_reg))(inputs)
    inner = BatchNormalization()(inner)
    inner = GaussianNoise(gau_sigma, name='guassian_noise1')(inner)
    inner = Activation('relu')(inner)

    inner = Conv2D(64, (11, 11),
            padding='valid', name='conv2',
            kernel_regularizer=l2(l2_reg))(inner)
    inner = BatchNormalization()(inner)
    inner = GaussianNoise(gau_sigma, name='gaussian_noise2')(inner)
    inner = Activation('relu')(inner)
    print(inner.shape)

    inner = Flatten(name='flat')(inner)
    #inner = Dropout(0.2)(inner)
    pred = Dense(cell_num, kernel_initializer='he_normal',
            kernel_regularizer=l2(l2_reg),
            #kernel_regularizer=l1_l2(l1=l1_Wreg, l2=l2_reg),
            activity_regularizer=l1(l1_reg),
            name='dense1')(inner)

    pred = BatchNormalization(axis=-1)(pred)
    pred = ParametricSoftplus()(pred)

    model = Model(inputs=[inputs], outputs=[pred])
    model.summary()

    return model

def crnn_model(num_hidden, bc_num=128, bc_size=25, rolling_window=20, l1_Wreg=0, cell_num=80):
    #input_shape = (150, 200, rolling_window)
    input_shape = (90, 90, rolling_window)
    l2_reg = 1e-3
    l1_reg = 1e-3
    #l1_Wreg = 1e3
    gau_sigma = 0.1

    inputs = Input(name='the_input', shape=input_shape, dtype='float32')

    # convolutional layer
    inner = Conv2D(bc_num, (bc_size, bc_size),  
            padding='valid', name='conv1', kernel_initializer='normal',
            #kernel_regularizer=l2(l2_reg))(inputs)
            kernel_regularizer=l1_l2(l1=l1_Wreg, l2=l2_reg))(inputs)

            #kernel_regularizer=l1_l2(l1=l1_Wreg, l2=l2_reg))(inputs)
    inner = BatchNormalization(axis=-1)(inner)

    inner = GaussianNoise(gau_sigma, name='guassian_noise1')(inner)
    inner = Activation('relu')(inner)
    #inner = BatchNormalization()(inner)

    convF_num = 64
    inner = Conv2D(convF_num, (11, 11),
            padding='valid', name='conv2', kernel_initializer='normal',
            kernel_regularizer=l2(l2_reg))(inner)
    inner = BatchNormalization(axis=-1)(inner)
    inner = GaussianNoise(gau_sigma, name='gaussian_noise2')(inner)
    inner = Activation('relu')(inner)
    #inner = BatchNormalization()(inner)

    print(inner.shape)

    # CNN to RNN
    inner = Reshape(target_shape=((-1, convF_num)), name='reshape')(inner)
    inner = Permute((2, 1))(inner)
    pred = LSTM(num_hidden, return_sequences=True, kernel_initializer='normal',
            name='lstm1', kernel_regularizer=l2(l2_reg))(inner)

    pred = BatchNormalization()(pred)

    pred = Flatten()(pred)

    pred = Dense(cell_num, kernel_initializer='he_normal',
            kernel_regularizer=l2(l2_reg),
            activity_regularizer=l1(l1_reg),
            name='dense1')(pred)

    pred = BatchNormalization()(pred)
    pred = ParametricSoftplus()(pred)

    model = Model(inputs=[inputs], outputs=[pred])
    model.summary()

    return model

def crnn_gru_model(num_hidden, bc_num=128, bc_size=25, rolling_window=20, l1_Wreg=0, cell_num=80):
    input_shape = (90, 90, rolling_window)
    l2_reg = 1e-3
    l1_reg = 1e-3
    gau_sigma = 0.1

    inputs = Input(name='the_input', shape=input_shape, dtype='float32')

    # convolutional layer
    inner = Conv2D(bc_num, (bc_size, bc_size),  
            padding='valid', name='conv1', kernel_initializer='normal',
            #kernel_regularizer=l2(l2_reg))(inputs)
            kernel_regularizer=l1_l2(l1=l1_Wreg, l2=l2_reg))(inputs)

    inner = BatchNormalization(axis=-1)(inner)

    inner = GaussianNoise(gau_sigma, name='guassian_noise1')(inner)
    inner = Activation('relu')(inner)

    convF_num = 64
    inner = Conv2D(convF_num, (11, 11),
            padding='valid', name='conv2', kernel_initializer='normal',
            kernel_regularizer=l2(l2_reg))(inner)
    inner = BatchNormalization(axis=-1)(inner)
    inner = GaussianNoise(gau_sigma, name='gaussian_noise2')(inner)
    inner = Activation('relu')(inner)

    print(inner.shape)

    # CNN to RNN
    inner = Reshape(target_shape=((-1, convF_num)), name='reshape')(inner)
    inner = Permute((2, 1))(inner)
    pred = GRU(num_hidden, return_sequences=True, kernel_initializer='normal',
            name='lstm1', kernel_regularizer=l2(l2_reg))(inner)

    pred = BatchNormalization()(pred)

    pred = Flatten()(pred)

    pred = Dense(cell_num, kernel_initializer='he_normal',
            kernel_regularizer=l2(l2_reg),
            activity_regularizer=l1(l1_reg),
            name='dense1')(pred)

    pred = BatchNormalization()(pred)
    pred = ParametricSoftplus()(pred)

    model = Model(inputs=[inputs], outputs=[pred])
    model.summary()

    return model

def crnn_vanilla_model(num_hidden, bc_num=128, bc_size=25, rolling_window=20, l1_Wreg=0, cell_num=80):
    input_shape = (90, 90, rolling_window)
    l2_reg = 1e-3
    l1_reg = 1e-3
    gau_sigma = 0.1

    inputs = Input(name='the_input', shape=input_shape, dtype='float32')

    # convolutional layer
    inner = Conv2D(bc_num, (bc_size, bc_size),  
            padding='valid', name='conv1', kernel_initializer='normal',
            kernel_regularizer=l1_l2(l1=l1_Wreg, l2=l2_reg))(inputs)

    inner = BatchNormalization(axis=-1)(inner)

    inner = GaussianNoise(gau_sigma, name='guassian_noise1')(inner)
    inner = Activation('relu')(inner)

    convF_num = 64
    inner = Conv2D(convF_num, (11, 11),
            padding='valid', name='conv2', kernel_initializer='normal',
            kernel_regularizer=l2(l2_reg))(inner)
    inner = BatchNormalization(axis=-1)(inner)
    inner = GaussianNoise(gau_sigma, name='gaussian_noise2')(inner)
    inner = Activation('relu')(inner)

    print(inner.shape)

    # CNN to RNN
    inner = Reshape(target_shape=((-1, convF_num)), name='reshape')(inner)
    inner = Permute((2, 1))(inner)
    pred = SimpleRNN(num_hidden, return_sequences=True, kernel_initializer='normal', 
            name='vanilla_rnn', kernel_regularizer=l2(l2_reg))(inner)

    pred = BatchNormalization()(pred)

    pred = Flatten()(pred)

    pred = Dense(cell_num, kernel_initializer='he_normal',
            kernel_regularizer=l2(l2_reg),
            activity_regularizer=l1(l1_reg),
            name='dense1')(pred)

    pred = BatchNormalization()(pred)
    pred = ParametricSoftplus()(pred)

    model = Model(inputs=[inputs], outputs=[pred])
    model.summary()

    return model


def cnn_lateral_model(bc_num=128, bc_size=25, rolling_window=20, l1_Wreg=0, lateral_size=3, cell_num=80, 
        session=None, batch_size=None):
    input_shape = (90, 90, rolling_window)
    l2_reg = 1e-3
    l1_reg = 1e-3
    #l1_Wreg = 1e-3
    gau_sigma = 0.1
    #cell_num = 80

    inputs = Input(name='the_input', shape=input_shape, batch_shape=(batch_size, 90, 90, rolling_window), dtype='float32')

    # convolutional layer
    inner = Conv2D(bc_num, (bc_size, bc_size), 
            padding='valid', name='conv1', 
            #kernel_regularizer=l2(l2_reg))(inputs)
            kernel_regularizer=l1_l2(l1=l1_Wreg, l2=l2_reg))(inputs)
    inner = BatchNormalization()(inner)
    inner = GaussianNoise(gau_sigma, name='guassian_noise1')(inner)
    inner = Activation('relu')(inner)

    inner = Conv2D(64, (11, 11),
            padding='valid', name='conv2',
            kernel_regularizer=l2(l2_reg))(inner)
    inner = BatchNormalization()(inner)
    inner = GaussianNoise(gau_sigma, name='gaussian_noise2')(inner)
    inner = Activation('relu')(inner)

    tmp_shape = inner.shape[1:]

    print('tmp_inner shape: ' + str(tmp_shape))
    inner = Reshape(target_shape=(int(tmp_shape[0]*tmp_shape[1]*tmp_shape[2]), ))(inner)
    print(inner.shape)

    #inner = Dropout(0.2)(inner)
    pred = Dense(cell_num, kernel_initializer='he_normal',
            kernel_regularizer=l2(l2_reg),
            #kernel_regularizer=l1_l2(l1=l1_Wreg, l2=l2_reg),
            activity_regularizer=l1(l1_reg),
            name='dense1')(inner)
    print('RGC shape: ' + str(pred.shape))

    # add lateral inhibition neurons
    pred = lateral_layer(group_size=lateral_size)(pred)
    
    print('lambda layer output: ' + str(pred.shape))

    pred = BatchNormalization(axis=-1)(pred)
    pred = ParametricSoftplus()(pred)

    model = Model(inputs=[inputs], outputs=[pred])
    model.summary()

    return model