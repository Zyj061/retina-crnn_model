import os
import h5py
import scipy.io as scio
import numpy as np

from keras import backend as K
from model import crnn_vanilla_model, cnn_model, crnn_model
from data_generator import load_data, ImageGenerator 
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.optimizers import Adam, SGD
import tensorflow as tf
from utils import *
from custom_activation import ParametricSoftplus
from scipy.stats import pearsonr
from vis_lstm import *
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '10'

parser = argparse.ArgumentParser(description='retinal models for dynamic natural scenes')
parser.add_argument('--model', type=str, default='crnn_lstm', help='encoding model')
parser.add_argument('--save_result', type=bool, default=True, help='save the test results or not')
parser.add_argument('--result_dir', type=str, default='results/', help='path of results')
args = parser.parse_args()

def cc_nips(r, rhat):
    
    return pearsonr(r, rhat)[0]

def train_model(stim_name):

    rolling_window = 20
    lstm_neuronN = 32
    model = crnn_vanilla_model(lstm_neuronN, bc_size=25, l1_Wreg=5e-4)
    #model = crnn_model(lstm_neuronN, bc_size=25, l1_Wreg=5e-4) # crnn with lstm units 
    #model = cnn_model(bc_size=25, l1_Wreg=5e-4) # cnn model
    log_path = './log/crnn_vanilla_' + stim_name + '_log'

    BATCHSIZE = 64
    VAL_BATCHSIZE = 64

    tensorboard = TensorBoard(log_path, batch_size=BATCHSIZE)

    # preprocessing data

    triger_train = ImageGenerator(stim_name, BATCHSIZE, 'train', rolling_window, func='load_mov90_data')
    triger_test = ImageGenerator(stim_name, BATCHSIZE, 'test', rolling_window, func='load_mov90_data')

    #sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
    
    model.compile(loss='poisson', optimizer=Adam(), metrics=[cc])
    early_stop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=100, mode='min', verbose=1)
    #callback.set_model(model)

    # model training hyperparameters
    num_epochs = 1000
    print('train data number ' + str(triger_train.num_data))
    print('validatin data number ' + str(triger_test.num_data))
    model.fit_generator(generator=triger_train.next_batch(),
            steps_per_epoch=int(triger_train.num_data / BATCHSIZE),
            epochs=num_epochs, 
            callbacks=[tensorboard, early_stop],
            validation_data=triger_test.next_batch(), 
            validation_steps=int(triger_test.num_data / VAL_BATCHSIZE))

    return model


model_path = os.path.join('./model/movie2', args.model)
print('testing on ' + args.stim + ' data')
# stimulus, mov3 corresponding to the movie2 describsed in the paper
if args.stim == 'movie2':
	stim = 'mov3' 
elif args.stim == 'movie1':
	stim = 'mov1'
else:
	print "Please input valid stimulus type"

'''
# training model
model = train_model(stim)

# saving the learned model 
make_path(output_path)
save_model(model, output_path) 
'''

test_model(model_path, stim, 20, display=args.save_result, result_dir=args.result_dir)

