import numpy as np
import mxnet as mx
import os
import logging

###############################
###     Model Building      ###
###############################

def conv_layer(x, nf, k):
    #convolution and activation
    x = mx.sym.Convolution(x, num_filter=nf, kernel=k)
    x = mx.sym.Activation(x, act_type='relu')
    #max pooling reduces spatial dimension by half, increasing receptive field
    x = mx.sym.Pooling(x, kernel=(2,2), stride=(2,2), pool_type='max')
    return x

def build_cnn(conv_params, num_hidden, num_classes):

    #Input and label placeholders
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')

    #build conv layers
    x = data
    for i, conv_param in enumerate(conv_params):
        x = conv_layer(x, conv_param[0], conv_param[1])

    #flatten to fully dense layer
    x = mx.sym.Flatten(x, name='flat_1')
    #hidden layer
    x = mx.sym.FullyConnected(x, num_hidden=num_hidden, name='fc_1')
    x = mx.sym.Activation(x, act_type='relu', name='relu_3')
    #dense to vector of class length
    output = mx.sym.FullyConnected(x, num_hidden=num_classes, name='fc_2')
    #categorical cross-entropy 
    loss = mx.sym.SoftmaxOutput(output, label, name='softmax')
    return loss

###############################
###     Data Loading        ###
###############################

def get_data(f_path):
    train_X = np.load(os.path.join(f_path,'train_X.npy'))
    train_Y = np.load(os.path.join(f_path,'train_Y.npy'))
    validation_X = np.load(os.path.join(f_path,'validation_X.npy'))
    validation_Y = np.load(os.path.join(f_path,'validation_Y.npy'))
    return train_X, train_Y, validation_X, validation_Y

###############################
###     Training Loop       ###
###############################

def train(channel_input_dirs, hyperparameters, hosts):
    conv_params = hyperparameters.get('conv_params', [[20, (5,5)], [50, (5,5)]])
    num_fc = hyperparameters.get('num_fc', 128)
    num_classes = hyperparameters.get('num_classes', 43)
    batch_size = hyperparameters.get('batch_size', 64)
    epochs = hyperparameters.get('epochs', 10)
    learning_rate = hyperparameters.get('learning_rate', 1E-3)
    num_gpus = hyperparameters.get('num_gpus', 0)
    # set logging
    logging.getLogger().setLevel(logging.DEBUG)
    
    if len(hosts) == 1:
        kvstore = 'device' if num_gpus > 0 else 'local'
    else:
        kvstore = 'dist_device_sync' if num_gpus > 0 else 'dist_sync'
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    
    f_path = channel_input_dirs['training']
    train_X, train_Y, validation_X, validation_Y = get_data(f_path)
        
    train_iter = train_iter = mx.io.NDArrayIter(data = train_X, label=train_Y, batch_size=batch_size, shuffle=True)
    validation_iter = mx.io.NDArrayIter(data = validation_X, label=validation_Y, batch_size=batch_size, shuffle=False)
    sym = build_cnn(conv_params, num_fc, num_classes)
    net = mx.mod.Module(sym, context=ctx)
    net.fit(train_iter,
        eval_data=validation_iter,
        initializer = mx.initializer.Xavier(magnitude=2.24),
        optimizer='adam',
        optimizer_params={'learning_rate':learning_rate},
        eval_metric='acc',
        num_epoch=epochs)
    
    return net
    
    