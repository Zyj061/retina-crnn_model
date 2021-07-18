import os, random
import h5py
import scipy.io as scio
import numpy as np
from collections import namedtuple
from scipy.ndimage.filters import gaussian_filter1d

data_prefix = 'path_to_the_data/retina_data'

stim_filename = 'complete_stimulus.mat'
filename = 'mov3_data_1800.mat'
filename_1k = 'mov_1k_rgc5.mat'

filepath = os.path.join(data_prefix, filename)
filepath_1k = os.path.join(data_prefix, filename_1k)
stim_filepath = os.path.join(data_prefix, stim_filename) 
Exptdata = namedtuple('Exptdata', ['X', 'r', 'y'])
Exptdata_wn = namedtuple('Exptdata', ['X', 'r'])

def rolling_window(arr, history, time_axis=-1):

    if time_axis == 0:
        arr = arr.T
    elif time_axis == -1:
        pass
    else:
        raise ValueError('Time axis must be 0 (first dimension) or -1 (last)')
    
    assert history >= 1, "'window' must be least 1."
    assert history < arr.shape[-1], "'window' is longer than array"

    #with strides 
    shape = arr.shape[:-1] + (arr.shape[-1] - history, history)
    strides = arr.strides + (arr.strides[-1], )
    arr = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    if time_axis == 0:
        return np.rollaxis(arr, 1, 0)
    else:
        return arr

# loading movie data of pixel 90 * 90
def load_mov1K_data(stim_type, phase, history, data_ratio=None):

    train_filepath = os.path.join(data_prefix, 'train_mov1k.hdf5')
    print('train dataset' + train_filepath)
    test_filepath = os.path.join(data_prefix, 'test_mov1k.hdf5')

    if(os.path.exists(train_filepath)):
        train_data = h5py.File(train_filepath, 'r')
        test_data = h5py.File(test_filepath, 'r')
        
        X = train_data[stim_type + '_trn_x']
        r = train_data[stim_type + '_trn_r']

        if history > 1:
            X = rolling_window(np.array(X), history)
            X = np.transpose(X, (0, 1, 3, 2)) 
            r = r[:, history:]
        else:
            X = np.array(X)
            r = np.array(r)
            X = X.reshape(X.shape[0], X.shape[1], 1, X.shape[2])
        
        r = r.T

        trn_data = Exptdata_wn(X, r)

        tst_X = test_data[stim_type + '_tst_x']
        tst_r = test_data[stim_type + '_tst_r']

        if history > 1:
            tst_X = rolling_window(np.array(tst_X), history)
            tst_X = np.transpose(tst_X, (0, 1, 3, 2))
            tst_r = tst_r[:, history:]
        else:
            tst_X = np.array(tst_X)
            tst_r = np.array(tst_r)
            tst_X = tst_X.reshape(tst_X.shape[0], tst_X.shape[1], 1, tst_X.shape[2])

        tst_r = tst_r.T
        tst_data = Exptdata_wn(tst_X, tst_r)

        if phase == 'train':
            return trn_data

        elif phase == 'all':
            # for transfer testing
            all_X = np.concatenate((X, tst_X), axis=-1)
            print('r shape: ' + str(r.shape)) 
            all_r = np.concatenate((r, tst_r))

            all_data = Exptdata_wn(all_X, all_r)
            return all_data

        else:
            return tst_data

# loading movie data of pixel 90 * 90
def load_mov90_data(stim_type, phase, history, data_ratio=None):

    train_filepath = os.path.join(data_prefix, 'train_mov3.hdf5')
    #train_filepath = os.path.join(data_prefix, 'train_90.hdf5')
    print('train dataset' + train_filepath)
    test_filepath = os.path.join(data_prefix, 'test_mov3.hdf5')
    #test_filepath = os.path.join(data_prefix, 'test_90.hdf5')

    if(os.path.exists(train_filepath)):
        train_data = h5py.File(train_filepath, 'r')
        test_data = h5py.File(test_filepath, 'r')
        
        X = train_data[stim_type + '_trn_x']
        r = train_data[stim_type + '_trn_r']
        y = train_data[stim_type + '_trn_y']

        if history > 1:
            print('data_ratio: ' + str(data_ratio))
            X = rolling_window(np.array(X), history)
            X = np.transpose(X, (0, 1, 3, 2)) 
            r = r[:, history:]
            y = y[:, :, history:]
            y[y>1] = 1
        else:
            X = np.array(X)
            r = np.array(r)
            y = np.array(y)
            X = X.reshape(X.shape[0], X.shape[1], 1, X.shape[2])
        
        r = r.T
        if data_ratio is not None:
            trn_whole_size = X.shape[3]
            trn_num = int(trn_whole_size * data_ratio / 100) 
            X = X[:, :, :, :trn_num]

        trn_data = Exptdata(X, r, y)

        tst_X = test_data[stim_type + '_tst_x']
        tst_r = test_data[stim_type + '_tst_r']
        tst_y = test_data[stim_type + '_tst_y']

        if history > 1:
            tst_X = rolling_window(np.array(tst_X), history)
            tst_X = np.transpose(tst_X, (0, 1, 3, 2))
            tst_r = tst_r[:, history:]
            tst_y = tst_y[:, :, history:]
            tst_y[tst_y>1] = 1
        else:
            tst_X = np.array(tst_X)
            tst_r = np.array(tst_r)
            tst_y = np.array(tst_y)
            tst_X = tst_X.reshape(tst_X.shape[0], tst_X.shape[1], 1, tst_X.shape[2])

        tst_r = tst_r.T
        tst_data = Exptdata(tst_X, tst_r, tst_y)

        if phase == 'train':
            return trn_data

        elif phase == 'all':
            # for transfer testing
            all_X = np.concatenate((X, tst_X), axis=-1)
            print('r shape: ' + str(r.shape)) 
            all_r = np.concatenate((r, tst_r))
            print('y shape: ' + str(y.shape))
            all_y = np.concatenate((y, tst_y), axis=-1)

            all_data = Exptdata(all_X, all_r, all_y)
            return all_data

        else:
            return tst_data


def load_data(stim_type, phase, history, data_ratio=None):

    train_filepath = os.path.join(data_prefix, 'train_86.hdf5')
    test_filepath = os.path.join(data_prefix, 'test_86.hdf5')

    if(os.path.exists(train_filepath)):
        train_data = h5py.File(train_filepath, 'r')
        test_data = h5py.File(test_filepath, 'r')
        
        X = train_data[stim_type + '_trn_x']
        r = train_data[stim_type + '_trn_r']
        y = train_data[stim_type + '_trn_y']

        if history > 1:
            X = rolling_window(np.array(X), history)
            X = np.transpose(X, (0, 1, 3, 2)) 
            r = r[:, history:]
            y = y[:, :, history:]
        else:
            X = np.array(X)
            r = np.array(r)
            X = X.reshape(X.shape[0], X.shape[1], 1, X.shape[2])
        
        r = r.T
        y = np.transpose(y, (2, 0, 1))

        trn_data = Exptdata(X, r, y)

        tst_X = test_data[stim_type + '_tst_x']
        tst_r = test_data[stim_type + '_tst_r']
        tst_y = test_data[stim_type + '_tst_y']

        if history > 1:
            tst_X = rolling_window(np.array(tst_X), history)
            tst_X = np.transpose(tst_X, (0, 1, 3, 2))
            tst_r = tst_r[:, history:]
            tst_y = tst_y[:, :, history:]
        else:
            tst_X = np.array(tst_X)
            tst_r = np.array(tst_r)
            tst_X = tst_X.reshape(tst_X.shape[0], tst_X.shape[1], 1, tst_X.shape[2])

        tst_r = tst_r.T
        tst_y = np.transpose(tst_y, (2, 0, 1))
        tst_data = Exptdata(tst_X, tst_r, tst_y)

        if phase == 'train':
            return trn_data
        else:
            return tst_data

# Data generator for training
class ImageGenerator:

    def __init__(self, stim_type, batch_size, phase, history, func='load_data', cell_num=80, data_ratio=None):

        if func == 'load_data':
            self.img_h = 150
            self.img_w = 200

        elif func == 'load_mov90_data' or func == 'load_wn_data' or 'load_mov1K_data':

            self.img_h = 90
            self.img_w = 90

        #self.cell_num = 86
        self.cell_num = cell_num
        print('cell number: ' + str(self.cell_num))
        self.batch_size = batch_size
        self.stim_type = stim_type
        self.phase = phase
        self.func = func

        self.history = history
        func_name = func + '(\'' + self.stim_type + '\',\'' + self.phase + '\',' + str(self.history) + ',' + str(data_ratio) + ')'
        self.data = eval(func_name)
        print('stimulus shape: ' + str(self.data.X.shape))
        self.num_data = self.data.X.shape[-1]
        self.indexes = list(range(self.num_data))
        self.cur_idx = 0


    def next_sample(self):

        self.cur_idx += 1
        if self.cur_idx >= self.num_data:
            self.cur_idx = 0
            random.shuffle(self.indexes)
        
        x = self.data.X[:, :, :, self.indexes[self.cur_idx]]
        y = self.data.r[self.indexes[self.cur_idx], :]

        return x, y

    def next_batch(self):

        while True:
            if self.func == 'load_wn_data':
                X = np.ones([self.batch_size, self.img_h, self.img_w, self.history], dtype='float16')

            else:
                X = np.ones([self.batch_size, self.img_h, self.img_w, self.history], dtype='float32')

            Y = np.ones([self.batch_size, self.cell_num])

            for i in range(self.batch_size):
                x, y = self.next_sample()
                if self.func == 'load_wn_data':
                    x = x.astype('float16')

                X[i, :, :, :] = x
                Y[i, :] = y

            yield(X, Y)

# Data generator for training
class ImageTrailGenerator:

    def __init__(self, stim_type, batch_size, phase, history, func='load_data', cell_num=80, data_ratio=None):

        if func == 'load_data':
            self.img_h = 150
            self.img_w = 200

        elif func == 'load_mov90_data' or func == 'load_wn_data' or 'load_mov1K_data':

            self.img_h = 90
            self.img_w = 90

        self.cell_num = cell_num
        print('cell number: ' + str(self.cell_num))
        self.batch_size = batch_size
        self.stim_type = stim_type
        self.phase = phase
        self.func = func

        self.history = history
        func_name = func + '(\'' + self.stim_type + '\',\'' + self.phase + '\',' + str(self.history) + ',' + str(data_ratio) + ')'
        self.data = eval(func_name)
        print('stimulus shape: ' + str(self.data.X.shape))
        self.num_data = self.data.X.shape[-1]
        self.indexes = list(range(self.num_data))
        self.cur_idx = 0

    def next_sample(self):

        self.cur_idx += 1
        if self.cur_idx >= self.num_data:
            self.cur_idx = 0
            random.shuffle(self.indexes)
        
        x = self.data.X[:, :, :, self.indexes[self.cur_idx]]
        #y = self.data.y[0, :, self.indexes[self.cur_idx]]
        y = self.data.y[-1, :, self.indexes[self.cur_idx]]
        #y = self.data.r[self.indexes[self.cur_idx], :]

        '''
        if self.phase == 'train':
            y = gaussian_filter1d(y, 1)  #smooth the response help to fit the model
        '''

        return x, y

    def next_batch(self):

        while True:
            if self.func == 'load_wn_data':
                X = np.ones([self.batch_size, self.img_h, self.img_w, self.history], dtype='float16')

            else:
                X = np.ones([self.batch_size, self.img_h, self.img_w, self.history], dtype='float32')

            Y = np.ones([self.batch_size, self.cell_num])

            for i in range(self.batch_size):
                x, y = self.next_sample()
                if self.func == 'load_wn_data':
                    x = x.astype('float16')
                    #x = x - 0.5

                X[i, :, :, :] = x
                Y[i, :] = y

            yield(X, Y)