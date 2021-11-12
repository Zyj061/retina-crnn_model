import os, random
import h5py
import scipy.io as scio
import numpy as np
from collections import namedtuple
from scipy.ndimage.filters import gaussian_filter1d

data_prefix = './data'

filename = 'cell_simpleNL_off_2GC_v3.mat'

filepath = os.path.join(data_prefix, filename)
Exptdata = namedtuple('Exptdata', ['X', 'r'])

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


def load_data(phase, history):

    train_filepath = os.path.join(data_prefix, 'train_off3.hdf5')
    test_filepath = os.path.join(data_prefix, 'test_off3.hdf5')

    if(os.path.exists(train_filepath)):
        train_data = h5py.File(train_filepath, 'r')
        test_data = h5py.File(test_filepath, 'r')
        
        
        X = train_data['trn_x']
        r = train_data['trn_r']

        if history > 1:
            X = rolling_window(np.array(X), history)
            X = np.transpose(X, (0, 1, 3, 2)) 
            r = r[:, history:]
        else:
            X = np.array(X)
            r = np.array(r)
            X = X.reshape(X.shape[0], X.shape[1], 1, X.shape[2])
        
        r = r.T

        trn_data = Exptdata(X, r)

        tst_X = test_data['tst_x']
        tst_r = test_data['tst_r']

        if history > 1:
            tst_X = rolling_window(np.array(tst_X), history)
            tst_X = np.transpose(tst_X, (0, 1, 3, 2))
            tst_r = tst_r[:, history:]
        else:
            tst_X = np.array(tst_X)
            tst_r = np.array(tst_r)
            tst_X = tst_X.reshape(tst_X.shape[0], tst_X.shape[1], 1, tst_X.shape[2])

        tst_r = tst_r.T
        tst_data = Exptdata(tst_X, tst_r)

        if phase == 'train':
            return trn_data
        else:
            return tst_data

    else:
        data = scio.loadmat(filepath)

        # loading different type of data
        stim = data['CB']

        # split data into train set and validation set
        stim_size = 3e5
        trn_stim_size = int(stim_size / 6 * 5)
        trn_x = stim[:, :, :trn_stim_size]
        tst_x = stim[:, :, trn_stim_size:]
        
        r = data['fr']
        r = r.T

        trn_r = r[:, :trn_stim_size]
        tst_r = r[:, trn_stim_size:]

        with h5py.File(train_filepath, 'w') as fw:
            fw.create_dataset('trn_x', data=trn_x)
            fw.create_dataset('trn_r', data=trn_r)

        with h5py.File(test_filepath, 'w') as fw:
            fw.create_dataset('tst_x', data=tst_x)
            fw.create_dataset('tst_r', data=tst_r)


# Data generator for training
class ImageGenerator:

    def __init__(self, batch_size, phase, history, func='load_data'):

        self.img_h = 8
        self.img_w = 8

        self.cell_num = 2 # 2 RGC off cell to fit
        self.batch_size = batch_size
        self.phase = phase

        self.history = history
        self.data = load_data(self.phase, self.history)
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

        '''
        if self.phase == 'train':
            y = gaussian_filter1d(y, 1)  #smooth the response help to fit the model
        '''

        return x, y

    def next_batch(self):

        while True:
            X = np.ones([self.batch_size, self.img_h, self.img_w, self.history])
            Y = np.ones([self.batch_size, self.cell_num])

            for i in range(self.batch_size):
                x, y = self.next_sample()
                X[i, :, :, :] = x
                Y[i, :] = y

            yield(X, Y)

