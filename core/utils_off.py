import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import scipy.io as scio
from scipy.sparse import *
from keras import backend as K
from keras.models import model_from_json
from custom_activation import ParametricSoftplus
from off_data_generator import load_data
import h5py
from visualizations import *
import math
import gc

import tensorflow as tf

filepath = 'path_to_off_data'

# generating random array
def random_array(shape, mean=128., std=20.):
    """Creates a uniformly distributed random array with the given `mean` and `std`.
    Args:
        shape: The desired shape
        mean: The desired mean (Default value = 128)
        std: The desired std (Default value = 20)
    Returns: Random numpy array of given `shape` uniformly distributed with desired `mean` and `std`.
    """
    x = np.random.random(shape)
    # normalize around mean=0, std=1
    x = (x - np.mean(x)) / (np.std(x) + K.epsilon())
    # and then around the desired mean/std
    x = (x * std) + mean
    return x

# def make the dirs
def make_path(filepath):
    
    if os.path.exists(filepath) == False:
        os.makedirs(filepath)

# save the final results of the model
def save_model(model, output_path):

    # save the network architecture
    filepath = os.path.join(output_path, 'architecture.json') 
    
    model_json = model.to_json()
    open(filepath, 'w').write(model_json)

    # save the weights of the model
    filepath = os.path.join(output_path, 'weights.h5')

    model.save_weights(filepath)

def matrix_rotate(A):
    h = A.shape[0]
    w = A.shape[1]
    Ar = np.zeros_like(A)
    for i in range(h):
        for j in range(w):
            Ar[i,j] = A[h-1-i,w-1-j]
    return Ar

# plot the firing rate 
def plot_fr(r, r_pre, save_path, cell_id_name=''):

    timewindow = len(r)

    plt.subplot(1, 1, 1)
    plt.plot(r, 'b', linewidth=3, alpha=0.4, label='data')
    plt.plot(r_pre, 'r', linewidth=3, alpha=0.4, label='prediction')
    plt.ylabel('Rate', fontsize=18)
    plt.xlim([0, timewindow])
    plt.legend(loc='upper left')

    cc_res = pearsonr(r, r_pre)[0]
    print('cc: ' + str(cc_res))
    print('-----------------------------')

    plt.title(cell_id_name + 'cc: ' + str(cc_res), fontsize=18)

    plt.savefig(save_path)
    plt.close()
    return cc_res


# save the predicting result comparing the data 
def vis_fr(model, data, output_path='', display=True):

    X = data.X
    y = data.r

    X = np.transpose(X, (3, 0, 1, 2))
    y_pred = model.predict_on_batch(X)
    
    timewindow = y_pred.shape[0]
    num_cell = y_pred.shape[1]
    response_filepath = os.path.join(output_path, 'response_test')
    make_path(response_filepath)
    cc_res = cc_np(y, y_pred)
    print('cc: ' + str(cc_res))

    if display is True:

        with h5py.File(response_filepath + '/result.hdf5', 'w') as fw:
            fw.create_dataset('r_data', data=y)
            fw.create_dataset('r_pre', data=y_pred)

        for i in range(num_cell):
            y_max = y[:, i].max()
            y_min = y[:, i].min() 
            y_pre_max = y_pred[:, i].max()
            y_pre_min = y_pred[:, i].min()

            if (y_max - y_min) == 0:
                break
            
        for i in range(num_cell):

            figpath = response_filepath + '/' + str(i) + '.png'
            plot_fr(y[:, i].T, y_pred[:, i].T, figpath)

    # computing the average firing rate of the population cells of data and model
    y_avg = np.mean(y, axis=1)
    y_pre_avg = np.mean(y_pred, axis=1)
    figpath = response_filepath + '/avg.png'

    if display is True:
        avg_cc = plot_fr(y_avg, y_pre_avg, figpath)
    else:
        avg_cc = pearsonr(y_avg, y_pre_avg)[0]

    print('cc on population cell: %f' %(avg_cc))

    return cc_res, avg_cc

# computing the pearson correlation for multi cell under array mode
def cc_np(x, y):

    num_cell = x.shape[1]
    cc_res = np.zeros((num_cell, ))
    for i in range(num_cell):
        cc_res[i] = pearsonr(x[:, i], y[:, i])[0]

    return np.mean(cc_res)

# computing the multivaribale pearsonr correlation
def cc(x, y):
    """Pearson's correlation coefficient

    If r, rhat are matrices, cc() computes the average pearsonr correlation
    of each column vector
    """

    print('firing rate shape: ' + str(x.shape))
    mx = K.mean(x, 0)
    my = K.mean(y, 0)

    xm, ym = x-mx, y-my
    r_num = K.batch_dot(xm, ym, axes=0)
    
    epsilon = 1e-4
    def ss(a):
        return K.batch_dot(a, a, axes=0)

    r_den = K.sqrt(ss(xm) * ss(ym) + epsilon)

    r = r_num / r_den
    r = K.mean(r)

    # Presumably, if abs(r) > 1, then it is only some small artifact of floating
    # point arithmetic.

    return r

# computing the multivaribale pearsonr correlation
def cc_loss(x, y):
    """Pearson's correlation coefficient

    If r, rhat are matrices, cc() computes the average pearsonr correlation
    of each column vector
    """

    mx = K.mean(x, 0)
    my = K.mean(y, 0)

    xm, ym = x-mx, y-my
    r_num = K.batch_dot(xm, ym, axes=0)
    
    #r_num = np.add.reduce(xm * ym)
    epsilon = 1e-4
    def ss(a):
        return K.batch_dot(a, a, axes=0)

    r_den = K.sqrt(ss(xm) * ss(ym) + epsilon)

    r = r_num / r_den
    print(r.shape)

    # Presumably, if abs(r) > 1, then it is only some small artifact of floating
    # point arithmetic.

    #r = K.max(K.min(r, np.int(1.0)), np.int(-1.0))
    return -r

def get_denseSta(model, val_data, layer_id=-1, samples=10000, batch_size=1000, print_every=None, subunit_id=None):

    stim_shape = (8, 8, 20)

    print_every = 1
    if subunit_id is not None:
        activations = K.function([model.layers[0].input], [model.layers[layer_id].output])

        def get_activations(stim):
            activity = activations([stim])[0]
            # first dimension is batch size
            return activity[:, :, subunit_id[0], subunit_id[1]]
    else:
        get_activations = K.function([model.layers[0].input], [model.layers[layer_id].output])

    # Initialize STA
    sta = 0
    sta_data = 0
    res = 0
    cell_num = 2
    X = val_data.X
    r = val_data.r
    X = np.transpose(X, (3, 0, 1, 2))
    samples = X.shape[0]

    # Generate white noise and map STA
    for batch in range(int(np.ceil(samples/batch_size))):

        '''
        whitenoise = np.random.randn(*((batch_size,) + stim_shape)).astype('float32')
        #whitenoise = random_array(((batch_size,) + stim_shape), mean=0, std=0.05)
        whitenoise[whitenoise<0.5] = 0
        whitenoise[whitenoise>=0.5] = 1
        whitenoise = whitenoise - 0.5

        r_pre = model.predict_on_batch(whitenoise)
        '''

        X_batch = X[batch*batch_size : (batch+1)*batch_size]
        y_data = r[batch*batch_size : (batch+1)*batch_size, :]
        if(batch == int(np.ceil(samples/batch_size))):
            X_batch = X[batch*batch_size:]
            y_data = r[batch*batch_size:, :]

        len_data = X_batch.shape[0]
        r_pre = model.predict_on_batch(X_batch)
        #y_pre = np.array(np.random.poisson(r_pre)>0)

        #y_flat = y_pre.reshape(batch_size, -1)
        y_flat = r_pre.reshape(len_data, -1)
        x_flat = X_batch.reshape(len_data, -1)
        y_data_flat = y_data.reshape(len_data, -1)

        sta += np.dot(y_flat.T, x_flat)
        sta_data += np.dot(y_data_flat.T, x_flat)
        print('processing on batch %d' % (batch+1))

    #print (response.shape, response_flat.shape, whitenoise_flat.shape)
    sta /= samples
    sta_data /= samples

    sta = sta.reshape(*((cell_num, ) + stim_shape))
    sta_data = sta_data.reshape(*((cell_num, ) + stim_shape))
    sta = sta.transpose((0, 3, 1, 2))
    sta_data = sta_data.transpose((0, 3, 1, 2))
    # computing the sta for conv layer

    return sta, sta_data


def get_sta(model, layer_id, samples=4, batch_size=1, print_every=None, subunit_id=None):

    stim_shape = (8, 8, 20)

    print_every = 1
    if subunit_id is not None:
        #activations = theano.function([model.layers[0].input], model.layers[layer_id].get_output(train=False))
        activations = K.function([model.layers[0].input], [model.layers[layer_id].output])

        #activations = layer0_output[0]
        def get_activations(stim):
            activity = activations([stim])[0]
            # first dimension is batch size
            return activity[:, :, subunit_id[0], subunit_id[1]]
    else:
        get_activations = K.function([model.layers[0].input], [model.layers[layer_id].output])

    # Initialize STA
    sta = 0
    res = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    cnt = 0

    # Generate white noise and map STA
    for batch in range(int(np.ceil(samples/batch_size))):

        whitenoise = np.random.randn(*((batch_size,) + stim_shape)).astype('float32')
        response = get_activations([whitenoise])[0]
        true_response_shape = response.shape[1:]

        channel_num = true_response_shape[2]
        response_flat = response.reshape(batch_size, -1, channel_num)
        response_flat = response_flat.transpose((2, 1, 0))
        response_flat = response_flat.astype('float32')
        response_w = response.shape[1]
        response_h = response.shape[2]
        del response
        print('response_flat shape: ' + str(response_flat.shape))
        whitenoise_flat = whitenoise.reshape(batch_size, -1)
        del whitenoise
        #whitenoise_flat = whitenoise.transpose((1, 2, 0, 3)) # dimension: W x H x batch_size x history
        print('whitenoise_flat shape: ' + str(whitenoise_flat.shape))

        stimSize = whitenoise_flat.shape[1]
        chunkNum = 4
        chunkSize = int(stimSize / chunkNum)
        print('chunkSize: ' + str(chunkSize))

        """
        # sta will be matrix of units x sta dims
        """
        if sta == 0:
            #sta = coo_matrix((channel_num, response_w*response_h, whitenoise_flat.shape[1]), dtype=np.float32).toarray()
            sta1 = np.zeros((channel_num, response_w*response_h, chunkSize), dtype='float32')
            sta2 = np.zeros((channel_num, response_w*response_h, chunkSize), dtype='float32')
            sta3 = np.zeros((channel_num, response_w*response_h, chunkSize), dtype='float32')
            sta4 = np.zeros((channel_num, response_w*response_h, chunkSize), dtype='float32')
         
            #sta = np.zeros((channel_num, response_w*response_h, whitenoise_flat.shape[1]), dtype='float32')

            #sta = csc_matrix((channel_num, response.shape[1]*response.shape[2], whitenoise_flat.shape[1]), dtype=float32).todense()

        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default() as g:

            resp = tf.placeholder(tf.float32, shape=(response_flat.shape[1], batch_size))
            stim_flat = tf.placeholder(tf.float32, shape=(batch_size, chunkSize))
            sta_res = tf.placeholder(tf.float32, shape=(response_flat.shape[1], chunkSize))

            mat_op = tf.matmul(resp, stim_flat)
            add_op = tf.add(mat_op, sta_res)

            with tf.Session(config=config, graph=graph) as sess:
                init = tf.global_variables_initializer()
                sess.run(init)

                for iChnl in range(channel_num):
                    print('processing channel: %d' % (iChnl))

                    feed_dict = {resp: response_flat[iChnl, :], stim_flat: whitenoise_flat[:, :chunkSize], sta_res: sta1[iChnl, :]}
                    _, sta1[iChnl, :] = sess.run([mat_op, add_op], feed_dict=feed_dict)
                    del feed_dict

                    feed_dict = {resp: response_flat[iChnl, :], stim_flat: whitenoise_flat[:, chunkSize:2*chunkSize], sta_res: sta2[iChnl, :]} 
                    _, sta2[iChnl, :] = sess.run([mat_op, add_op], feed_dict=feed_dict)
                    del feed_dict

                    feed_dict = {resp: response_flat[iChnl, :], stim_flat: whitenoise_flat[:, 2*chunkSize:3*chunkSize], sta_res: sta3[iChnl, :]} 
                    _, sta3[iChnl, :] = sess.run([mat_op, add_op], feed_dict=feed_dict)
                    del feed_dict

                    feed_dict = {resp: response_flat[iChnl, :], stim_flat: whitenoise_flat[:, 3*chunkSize:], sta_res: sta4[iChnl, :]} 
                    _, sta4[iChnl, :] = sess.run([mat_op, add_op], feed_dict=feed_dict)
                    del feed_dict

                    tf.get_default_graph().finalize()

                    gc.collect()

        #sta += np.dot(response_flat, whitenoise_flat)

        if print_every:
            if batch % print_every == 0:
                print('On batch %i of %i...' %(batch, samples/batch_size))

    #print (response.shape, response_flat.shape, whitenoise_flat.shape)
    sta1 /= samples
    sta2 /= samples

    # computing the sta for conv layer
    sta1 = np.mean(sta1, axis=1)
    sta2 = np.mean(sta2, axis=1)
    new_stimShape = (stim_shape[0], stim_shape[1], stim_shape[2]/chunkNum)
    sta1 = sta1.reshape(*((true_response_shape[2],) + new_stimShape))
    sta2 = sta2.reshape(*((true_response_shape[2],) + new_stimShape))
    sta = np.concatenate((sta1, sta2), axis=2)

    return sta1, sta2


def visualize_sta(sta, sta_data, visualpath, img_name, fig_size=(8, 10), display=True, normalize=True):

    '''
    if len(sta) == 3:
        num_units = 1
    else:
        num_units = sta.shape[0]
    '''
    nt = 20
    print('sta_shape: ' + str(sta.shape))
    num_units = sta.shape[0]

    # plot space and time profiles together
    num_cols = int(np.sqrt(num_units))
    num_rows = int(np.ceil(num_units/num_cols))
    cl_max = 0
    for x in range(num_cols):
        for y in range(num_rows):
            plt_idx = y * num_cols + x + 1
            if plt_idx > num_units:
                break
            if nt > 1:
                spatial,temporal = ft.decompose(sta[plt_idx-1])
                spatial_data, temporal_data = ft.decompose(sta_data[plt_idx-1])
            else:
                spatial = sta[plt_idx-1][0]

            sp_max = max(np.max(abs(spatial)), np.max(abs(spatial_data)))
            cl_max = max(cl_max, sp_max)
            #cl_max = max(cl_max, np.max(abs(spatial)))
    colorlimit = [-cl_max, cl_max]

    for x in range(num_cols):
        for y in range(num_rows):
            plt_idx = y * num_cols + x + 1
            if plt_idx > num_units:
                break
            if nt > 1:
                spatial,temporal = ft.decompose(sta[plt_idx-1])
                spatial_data, temporal_data = ft.decompose(sta_data[plt_idx-1])

            else:
                spatial = sta[plt_idx-1][0]
                temporal = np.array([])

            plt.subplot(2, 2, 1)
            plt.title('STA spatial data', fontsize=16)
            plt.imshow(spatial_data, interpolation='nearest', cmap='seismic', clim=colorlimit)
            plt.grid('off')
            plt.xticks([])
            plt.yticks([])
            plt.axis('on')

            plt.subplot(2, 2, 3)
            plt.title('STA temporal data', fontsize=16)
            plt.plot(np.linspace(0, len(temporal_data) * 10, len(temporal_data)), temporal_data, 'k', linewidth=2)
            plt.grid('off')
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            
            plt.subplot(2, 2, 2)
            plt.title('STA spatial predict', fontsize=16)
            plt.imshow(spatial, interpolation='nearest', cmap='seismic', clim=colorlimit)
            plt.grid('off')
            plt.xticks([])
            plt.yticks([])
            plt.axis('on')

            plt.subplot(2, 2, 4)
            plt.title('STA temporal predict', fontsize=16)
            plt.plot(np.linspace(0, len(temporal) * 10, len(temporal)), temporal, 'k', linewidth=2)
            plt.grid('off')
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])

            figpath = os.path.join(visualpath, img_name, str(plt_idx) + '.png')

            plt.savefig(figpath)
            plt.close()

# get the kernel of the filters
def get_kernel(weight, k_name):

    tmp_k = weight[k_name]
    tmp_k = tmp_k[k_name]

    return tmp_k

def plot_nonlinear(output_path, alpha, beta):

    num_cell = len(alpha)
    
    nonlinearity_filepath = os.path.join(output_path, 'vis_nonlinearity')
    make_path(nonlinearity_filepath)

    x = np.arange(-10, 10, 0.1)


    for i in range(num_cell):

        #y = alpha[i] * math.log(1 + math.exp(beta[i] * x))
        f = lambda x: alpha[i] * math.log(1 + math.exp(beta[i] *x))
        f2 = np.vectorize(f)
        y = f2(x)

        fig = plt.figure(figsize=(8, 8))

        ax = plt.subplot(111)
        ax.plot(x, y)
        plt.title(r'$cell %s \alpha=%f, \beta=%f$' %(alpha[i], beta[i]))

        filename = os.path.join(nonlinearity_filepath, str(i) + '.png')
        fig.savefig(filename)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(num_cell):

        f = lambda x: alpha[i] * math.log(1 + math.exp(beta[i] *x))
        f2 = np.vectorize(f)
        y = f2(x)

        ax.plot(x, y)
        plt.show()

    plt.legend()
    filename = os.path.join(nonlinearity_filepath, 'all_cell.png')
    fig.savefig(filename)
    plt.close()

def draw_train_STA(model, data, output_path, nt):

    visualpath = os.path.join(output_path, 'test_sta')
    make_path(visualpath)

    X = data.X
    r = data.r
    y = data.y

    X = np.transpose(X, (3, 0, 1, 2))
    r_pred = model.predict_on_batch(X)
    r_pred[r_pred<0] = 0
    print('y.shape' + str(r.shape))
    print('y_pred.shape ' + str(r_pred.shape))
    
    num_cell = r_pred.shape[1]

    #print(X.shape)
    stim_shape = X.shape[1:]
    samples = X.shape[0]

    X_flat = X.reshape(samples, -1)

    sta_data = 0
    sta_pre = 0

    for iCell in range(num_cell):
        
        y_data = y[:, 0, iCell]>0
        y_data = np.array(y_data.reshape(samples, 1)>0)
        y_pre = np.array(np.random.poisson(r_pred[:, iCell])>0)

        y_flat = y_pre.reshape(samples, 1)
        sta_pre = np.dot(y_flat.T, X_flat)
        sta_data = np.dot(y_data.T, X_flat)

        sta_data /= samples
        sta_data = sta_data.reshape(stim_shape)
        sta_pre /= samples
        sta_pre = sta_pre.reshape(stim_shape)

        sta_data = np.transpose(sta_data, (2, 0, 1))
        sta_pre = np.transpose(sta_pre, (2, 0, 1))

        if nt > 1:
            spatial_data, temporal_data = ft.decompose(sta_data)
            spatial_pre, temporal_pre = ft.decompose(sta_pre)
        else:
            spatial_data = sta_data[0]
            temporal_data = np.array([])
            spatial_pre = sta_pre[0]
            temporal_pre = np.array([])

        cl = max( np.max(abs(spatial_data)), np.max(abs(spatial_pre)) )
        colorlimit = [-cl, cl]
        plt.subplot(2,2,1)
        plt.title('STA spatial data', fontsize=20)
        plt.imshow(spatial_data, interpolation='nearest', cmap='seismic', clim=colorlimit)
        plt.grid('off')
        plt.xticks([])
        plt.yticks([])
        plt.axis('on')

        plt.subplot(2,2,3)
        plt.title('STA temporal data', fontsize=20)
        plt.plot(np.linspace(0, len(temporal_data) * 10, len(temporal_data)), temporal_data, 'k', linewidth=2)
        plt.grid('off')
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])


        plt.subplot(2,2,2)
        plt.title('STA spatial predict', fontsize=20)
        plt.imshow(spatial_pre, interpolation='nearest', cmap='seismic', clim=colorlimit)
        plt.grid('off')
        plt.xticks([])
        plt.yticks([])
        plt.axis('on')

        plt.subplot(2,2,4)
        plt.title('STA temporal predict', fontsize=20)
        plt.plot(np.linspace(0, len(temporal_pre) * 10, len(temporal_pre)), temporal_pre, 'k', linewidth=2)
        plt.grid('off')
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])

        plt.savefig(visualpath+'/sta_train' + str(iCell) + '.png')
        plt.savefig(visualpath+'/sta_train'+ str(iCell) + '.eps')
        plt.close()

# visulizing the response of model comparing to data 
def test_model(output_path, history):

        #print('Testing prune model..')
        filename = os.path.join(output_path, 'architecture.json')

        json_string = open(filename,'r').read()
        #model = model_from_json(json_string)
	# loading model with custom design ParametricSoftplus layer
        model = model_from_json(json_string, custom_objects={'ParametricSoftplus': ParametricSoftplus})

        filename = os.path.join(output_path, 'weights.h5')

        #print('filename ' + filename)
        model.load_weights(filename)
        model.summary()

        val_data = load_data('test', history)

        # visualize the firing rate of model comparing to data 

        vis_fr(model, val_data, output_path) 

        weights = h5py.File(filename, 'r')
        conv1 = get_kernel(weights, 'conv1')
        conv1_k = conv1['kernel:0']

        conv1_k = np.transpose(conv1_k, [3, 2, 0, 1])
        fig = plot_filters(conv1_k)
        #figpath = os.path.join(output_path, 'conv1_kernel.png')
        figpath = os.path.join(output_path, 'conv1_kernel.eps')
        fig.savefig(figpath)

        dense_sta, sta_data = get_denseSta(model, val_data)
        sta_filepath = os.path.join(output_path, 'denseSta.npy')
        np.save(sta_filepath, dense_sta) 
        make_path(os.path.join(output_path, 'dense_sta'))
        visualize_sta(dense_sta, sta_data, output_path, 'dense_sta')
