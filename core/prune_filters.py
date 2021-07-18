from __future__ import absolute_import, division, print_function
import matplotlib
matplotlib.use('Agg')

from json import dumps
from keras.models import model_from_json
from keras.optimizers import Adam
import numpy as np
import os, h5py

from keras.models import Sequential
from model import cnn_model, crnn_model
from custom_activation import ParametricSoftplus
import tensorflow as tf
from utils import *
import pyret.filtertools as ft
from scipy import stats

# return the adjacent weight
def get_adjW(row, col):
        N = col * row
        elen = 2 * col * row - row - col
        E = np.zeros((elen, 2), dtype='uint8')
        cnt = 0
        for i in range(N):
                nx = np.floor(i/col)
                ny = i % col
                if ny + 1 < col:
                        E[cnt, :] = [i, i+1]
                        cnt += 1
                if nx + 1 < row:
                        E[cnt, :] = [i, i + col]
                        cnt += 1

        adj = np.zeros((N, N))
        for e in range(elen):
                i = E[e, 0]
                j = E[e, 1]
                adj[i, j] = 1
                adj[j, i] = 1

        return adj

# compute the spatial autocorrelation of the convolution filters
def get_spaCorr(filters, row, col):

        mean_filter = np.mean(filters)
        N = row * col
        cov = 0
        for i in range(row):
            for j in range(col):
                cov += (filters[i, j] - mean_filter) ** 2

        B = filters - mean_filter
        tmp = 0
        n_adj = 0
        for x in range(row):
                for y in range(col):
                    if x>0:
                        tmp += B[x, y] * B[x-1, y]
                        n_adj += 1
                    if x<row-1:
                        tmp += B[x, y] * B[x+1, y]
                        n_adj += 1
                    if y>0:
                        tmp += B[x, y] * B[x, y-1]
                        n_adj += 1
                    if y<col-1:
                        tmp += B[x, y] * B[x, y+1]
                        n_adj += 1

        norm_factor = N / n_adj
        res = norm_factor * tmp / cov
        return res

def get_tCorr(temporal):

    nt = len(temporal)
    tAbs = abs(temporal)
    tMax = np.max(tAbs)
    tmax_idx = np.argmax(tAbs)
    nt = len(temporal)
    ntIdx = range(nt)

    diff = abs(tAbs - tMax)
    diff = diff.reshape(1, nt)
    #idx_diff = abs(ntIdx - tmax_idx)
    idx_diff = np.ones((nt, ))
    idx_diff = idx_diff.reshape(nt, 1)

    moreNum = np.mean(temporal)
    l2_reg = np.linalg.norm((tAbs-moreNum), ord=2)

    alpha = 5e-4

    res = np.dot(diff, idx_diff)/nt - alpha / (l2_reg + alpha)

    return res 

if __name__ == '__main__':

    with_time = 1
    nt = 20 if with_time == 1 else 1
    
    rootpath = '/path_to_model'
    stim_type = ['mov1', 'mov3']
    model_type = ['cnn2', 'crnn2_2']

    list_merge = [(stim, model_name) for stim in stim_type for model_name in model_type]

    #for stim, model_name in zip(stim_type, model_type):
    for stim, model_name in list_merge:

        filepath = os.path.join(rootpath, stim, model_name)

        print('Pruning filters of model ' + model_name + ' on stimulus ' + stim)

        bc_num = 64
        if 'cnn' in model_name:
            prune_model = cnn_model(bc_num=bc_num)
        elif 'crnn' in model_name:
            lstm_neuronNum = 32
            prune_model = crnn_model(lstm_neuronNum, bc_num=bc_num)
        else:
            print(model_name + ' doesn\'t exist')
            
        prune_model.compile(loss='poisson', optimizer=Adam(), metrics=[cc])

        filename = os.path.join(filepath, 'architecture.json')
        json_string = open(filename, 'r').read()
        model = model_from_json(json_string, custom_objects={'ParametricSoftplus':ParametricSoftplus})

        filename = os.path.join(filepath, 'weights.h5')
        model.load_weights(filename)

        layer_number = len(model.layers)

        for k in range(layer_number):

            print('processing layer ' + str(k))
            layer = model.get_layer(index=k)
            layer_config = layer.get_config()
            layer_name = layer_config['name']
            w = layer.get_weights()

            if layer_name == 'conv1':
                param_0 = w[0]  # convolutional filter weights
                param_1 = w[1]  # bias values
                
                [h, w, c, kernels] = param_0.shape

                p0_len = param_0.shape[3]
                p0 = np.zeros((p0_len, ))
                l1_norm = np.zeros((p0_len, ))
                temporal_stat = np.zeros((p0_len, ))

                adjW = get_adjW(h, w)
                all_spatial = np.zeros((h, w, p0_len))
               
                for i in range(p0_len):
                    tmp_sta = param_0[:, :, :, i]
                    tmp_sta = tmp_sta.transpose((2, 0, 1))
                    spatial, temporal = ft.decompose(tmp_sta)
                    temporal_stat[i] = get_tCorr(temporal)

                    all_spatial[:, :, i] = spatial

                spMax = np.max(abs(all_spatial))
                all_spatial = all_spatial / spMax

                for i in range(p0_len):

                    print('conv %d kernel' % (i))
                    spatial = all_spatial[:, :, i]
                    p0[i] = get_spaCorr(spatial, adjW, h, w)
                    l1_norm[i] = np.linalg.norm(spatial, ord=1)
                    print('spaCorr coef: %f\t l1_norm: %f \t tCorr: %f' %(p0[i], l1_norm[i], temporal_stat[i]))


                p0_idx = np.argsort(-p0)
                l1_sort = np.argsort(l1_norm)

                p0_idx10 = p0_idx[:bc_num]

                p0_tmp = np.zeros((h, w, c, bc_num))
                p1_tmp = np.zeros((bc_num, ))
                for i in range(bc_num):
                    p0_tmp[:, :, :, i] = param_0[:, :, :, p0_idx10[i]]
                    p1_tmp[i] = param_1[p0_idx10[i]]

                ws = []
                ws.append(p0_tmp)
                ws.append(p1_tmp)

                prune_model.layers[k].set_weights(ws)
            elif layer_name == 'batch_normalization_1':

                w = np.array(w)
                ws = np.zeros((w.shape[0], bc_num))
                for i in range(bc_num):
                    ws[:, i] = w[:, p0_idx10[i]]
                prune_model.layers[k].set_weights(ws)

            elif layer_name == 'conv2':

                param_0 = w[0]  # convolutional filter weights
                param_1 = w[1]  # bias values
                
                [h, w, c, kernels] = param_0.shape

                p0_tmp = np.zeros((h, w, bc_num, kernels))
                for i in range(bc_num):
                    p0_tmp[:, :, i, :] = param_0[:, :, p0_idx10[i], :]

                ws = []
                ws.append(p0_tmp)
                ws.append(param_1)

                prune_model.layers[k].set_weights(ws)

            else:
                prune_model.layers[k].set_weights(w)

        conv1_statfile = os.path.join(filepath, 'conv1_stat.npy')

        statRes = []
        statRes.append(p0)
        statRes.append(l1_norm)
        statRes.append(temporal_stat)
        statRes = np.array(statRes)
        np.save(conv1_statfile, statRes)
        print('saving file to ' + conv1_statfile)
        print('----------------------------------------------------------------')



