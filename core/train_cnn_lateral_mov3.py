import os
import h5py
import scipy.io as scio
import numpy as np

from keras import backend as K
from model import cnn_lateral_model
from data_generator import load_data, load_mov90_data, ImageGenerator 
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
import tensorflow as tf
from utils import *
from custom_activation import ParametricSoftplus
from scipy.stats import pearsonr
from keras import backend as K

BATCHSIZE = 64
VAL_BATCHSIZE = 64

def cc_nips(r, rhat):
    
    return pearsonr(r, rhat)[0]

def train_model(stim_name, lateral_size):

    model = cnn_lateral_model(bc_size=25, l1_Wreg=5e-4, lateral_size=lateral_size, session=sess, batch_size=BATCHSIZE)

    log_path = './log/cnn_lateral_' + stim_name + '_log'

    '''
    BATCHSIZE = 32
    VAL_BATCHSIZE = 32
    '''
    
    tensorboard = TensorBoard(log_path, batch_size=BATCHSIZE)

    # preprocessing data

    rolling_window = 20
    triger_train = ImageGenerator(stim_name, BATCHSIZE, 'train', rolling_window, func='load_mov90_data')
    triger_test = ImageGenerator(stim_name, BATCHSIZE, 'test', rolling_window, func='load_mov90_data')

    model.compile(loss='poisson', optimizer=Adam(), metrics=[cc])

    early_stop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=100, mode='min', verbose=1)

    # model training hyperparameters
    num_epochs = 1000
    print('train data number ' + str(triger_train.num_data))
    print('validatin data number ' + str(triger_test.num_data))
    model.fit_generator(generator=triger_train.next_batch(),
            steps_per_epoch=int(triger_train.num_data / BATCHSIZE),
            epochs=num_epochs, 
            callbacks=[early_stop, tensorboard],
            validation_data=triger_test.next_batch(), 
            validation_steps=int(triger_test.num_data / VAL_BATCHSIZE))

    return model


data_prefix = '/path_to_data'
stim_type = ['mov1', 'mov3']
stim = 2
print('training on ' + stim_type[stim] + ' data')

output_path = os.path.join(data_prefix, 'output', stim_type[stim], 'cnn_lateral')

random_repeat_num = 10
lateral_size = 8
results_filename = stim_type[stim] + '_lateral_' + str(lateral_size) + 'results.txt'
f = open(results_filename, 'w')

val_data = load_mov90_data(stim_type[stim], 'test', 20)

for i in range(random_repeat_num):

    # training model
    model = train_model(stim_type[stim], lateral_size)

    # saving the learned model 
    make_path(output_path)
    save_model(model, output_path) 

    cc_res, avg_cc, cc_std = test_model(output_path, stim_type[stim], 20, batch_size=VAL_BATCHSIZE)

    tmp_result = 'model %d: cc_res:%.4f cc_std:%.6f %.4f\n' % (i, cc_res, cc_std, avg_cc)
    f.write(tmp_result)

f.close()

