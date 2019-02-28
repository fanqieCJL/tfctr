# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import config

def slice(csr_data, start=0, size=-1):
    if not isinstance(csr_data[0], list):
        if size == -1 or start + size >= csr_data[0].shape[0]:
            slc_data = csr_data[0][start:]
            slc_labels = csr_data[1][start:]
        else:
            slc_data = csr_data[0][start:start + size]
            slc_labels = csr_data[1][start:start + size]
    else:
        if size == -1 or start + size >= csr_data[0][0].shape[0]:
            slc_data = []
            for d_i in csr_data[0]:
                slc_data.append(d_i[start:])
            slc_labels = csr_data[1][start:]
        else:
            slc_data = []
            for d_i in csr_data[0]:
                slc_data.append(d_i[start:start + size])
            slc_labels = csr_data[1][start:start + size]
    return csr2input(slc_data), slc_labels

def get_var(var_list, init_path=None):
    '''
        在tensorflow中初始化各种参数变量
    '''
    # if init_path is not None:
    #     load_var_map = pkl.load(open(init_path, 'rb'))
    #     print('load variable map from', init_path, load_var_map.keys())
    var_dict = dict()
    for var_name, var_shape, init_method in var_list:
        if init_method == 'zero':
            var_dict[var_name] = tf.Variable(tf.zeros(var_shape, dtype=config.DTYPE), name=var_name, dtype=config.DTYPE)
        elif init_method == 'one':
            var_dict[var_name] = tf.Variable(tf.ones(var_shape, dtype=config.DTYPE), name=var_name, dtype=config.DTYPE)
        elif init_method == 'normal':
            var_dict[var_name] = tf.Variable(tf.random_normal(var_shape, mean=0.0, stddev=config.STDDEV, dtype=config.DTYPE),
                                            name=var_name, dtype=config.DTYPE)
        elif init_method == 'tnormal':
            var_dict[var_name] = tf.Variable(tf.truncated_normal(var_shape, mean=0.0, stddev=config.STDDEV, dtype=config.DTYPE),
                                            name=var_name, dtype=config.DTYPE)
        elif init_method == 'uniform':
            var_dict[var_name] = tf.Variable(tf.random_uniform(var_shape, minval=config.MINVAL, maxval=config.MAXVAL, dtype=config.DTYPE),
                                            name=var_name, dtype=config.DTYPE)
        elif init_method == 'xavier':
            maxval = np.sqrt(6. / np.sum(var_shape))
            minval = -maxval
            var_dict[var_name] = tf.Variable(tf.random_uniform(var_shape, minval=minval, maxval=maxval, dtype=config.DTYPE),
                                            name=var_name, dtype=config.DTYPE)
        elif isinstance(init_method, int) or isinstance(init_method, float):
            var_dict[var_name] = tf.Variable(tf.ones(var_shape, dtype=config.DTYPE) * init_method, name=var_name, dtype=config.DTYPE)
    return var_dict

def activate(weights, activation_function):
    '''
        不同的激活函数选择
    '''
    if activation_function == 'sigmoid':
        return tf.nn.sigmoid(weights)
    elif activation_function == 'softmax':
        return tf.nn.softmax(weights)
    elif activation_function == 'relu':
        return tf.nn.relu(weights)
    elif activation_function == 'tanh':
        return tf.nn.tanh(weights)
    elif activation_function == 'elu':
        return tf.nn.elu(weights)
    elif activation_function == 'none':
        return weights
    else:
        return weights

def get_optimizer(opt_algo, learning_rate, loss):
    '''
        不同的优化器选择
    '''
    if opt_algo == 'adaldeta':
        return tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'adam':
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'ftrl':
        return tf.train.FtrlOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'gd':
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'padagrad':
        return tf.train.ProximalAdagradOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'pgd':
        return tf.train.ProximalGradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    else:
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)