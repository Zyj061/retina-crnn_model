import os
import sys
import h5py
import scipy.io as scio
import numpy as np

from keras import backend as K
from model_off import cnn_model, crnn_model
from off_data_generator import load_data, ImageGenerator 
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
import tensorflow as tf
from utils_off import *
from custom_activation import ParametricSoftplus
from scipy.stats import pearsonr
#from metrics import allmetrics, cc

data_prefix = './data/'

bc_size = int(sys.argv[1])

def cc_nips(r, rhat):
    
    return pearsonr(r, rhat)[0]

def train_model():
	
	# cnn model 
    model = cnn_model(bc_size=bc_size, l1_Wreg=0)
	
	# crnn model with 8 lstm units
	#model = crnn_model(num_hidden=8, bc_size=bc_size, l1_Wreg=0) 
	
    log_path = './off_cnn_bc' + str(bc_size) + '_log'
    BATCHSIZE = 4096
    VAL_BATCHSIZE = 2048
    tensorboard = TensorBoard(log_path, batch_size=BATCHSIZE)

    # preprocessing data

    rolling_window = 20
    triger_train = ImageGenerator(BATCHSIZE, 'train', rolling_window)
    triger_test = ImageGenerator(BATCHSIZE, 'test', rolling_window)

    model.compile(loss='poisson', optimizer=Adam(), metrics=[cc])
    early_stop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=50, mode='min', verbose=1)

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

data_prefix = './data/'
stim = 2
print('training on simulated off data')

output_path = os.path.join('./results', 'off', 'cnn_bc' + str(bc_size))

# training model
model = train_model()

# saving the learned model 
make_path(output_path)
save_model(model, output_path) 

test_model(output_path, 20)

