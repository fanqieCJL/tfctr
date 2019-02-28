# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import config
import utils
import time

class FNN:
    def __init__(self, train_loader, test_loader, embed_size=10, layer_size=None, 
                       layer_act=None, layer_keeps=None, opt_algo='gd', 
                       learning_rate=0.01, epoch=10, early_stop_round=None, 
                       l2=None, random_seed=None):
        self.graph = tf.Graph()

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.embed_size = embed_size
        self.layer_size = layer_size
        self.layer_act = layer_act
        self.layer_keeps = layer_keeps
        
        self.num_fields = len(config.FIELD_SIZES)
        self.var_list = []
        for idx in range(self.num_fields):
            self.var_list.append(['embed_{}'.format(idx), [config.FIELD_SIZES[idx], self.embed_size], 'xavier'])
        
        in_size = self.num_fields * self.embed_size
        for idx in range(len(layer_size)):
            self.var_list.append(['w_{}'.format(idx), [in_size, layer_size[idx]], 'xavier'])
            self.var_list.append(['b_{}'.format(idx), [layer_size[idx]], 'zero'])
            in_size = layer_size[idx]
        
        self.var_dict = utils.get_var(self.var_list)
        
        self.opt_algo = opt_algo
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.early_stop_round = early_stop_round
        self.l2 = l2
        self.random_seed = random_seed

        self.time_scores = []
        self.train_scores = []
        self.test_scores = []

#         with self.graph.as_default():
        if self.random_seed is not None:
            tf.set_random_seed(self.random_seed)
        self.X = [tf.sparse_placeholder(config.DTYPE) for n in range(self.num_fields)]
        self.y = tf.placeholder(config.DTYPE)

        with tf.variable_scope('Dense_Real_Layer'):
            w_embed = [self.var_dict['embed_{}'.format(idx)] for idx in range(self.num_fields)]
            xw = tf.concat([tf.sparse_tensor_dense_matmul(self.X[idx], w_embed[idx]) for idx in range(self.num_fields)], 1)
            layer_out = xw

        for idx in range(len(layer_size)):
            with tf.variable_scope('Hiden_Layer_{}'.format(idx)):
                wi = self.var_dict['w_{}'.format(idx)]
                bi = self.var_dict['b_{}'.format(idx)]
                layer_out = tf.nn.dropout(
                    utils.activate(tf.matmul(layer_out, wi) + bi, self.layer_act[idx]),
                    self.layer_keeps[idx]
                    )

        layer_out = tf.squeeze(layer_out)
        self.y_preds = tf.sigmoid(layer_out)

        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=layer_out)
        )
        if self.l2 is not None:
            for idx in range(self.num_fields):
                self.loss += self.l2 * tf.nn.l2_loss(self.var_dict['embed_{}'.format(idx)])
            for idx in range(len(self.layer_size)):
                self.loss += self.l2 * tf.nn.l2_loss(self.var_dict['w_{}'.format(idx)])
                
        self.optimizer = utils.get_optimizer(self.opt_algo, self.learning_rate, self.loss)

        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)
    
    def run(self):
        self.start_time = time.time()
        for num_epoch in tqdm(range(self.epoch)):
            train_iter = self.train_loader.sparse_iter()
            train_y_hat = []
            for X, y in train_iter:
                feed_dict = dict()
                if isinstance(X, list):
                    for idx in range(len(X)):
                        feed_dict[self.X[idx]] = X[idx]
                else:
                    feed_dict[self.X] = X
                feed_dict[self.y] = y
                
                _, local_loss = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
                
                feed_dict[self.y] = None
                y_hat = self.sess.run(self.y_preds, feed_dict=feed_dict)
                train_y_hat.extend(y_hat)
            self.train_scores.append(roc_auc_score(self.train_loader.y, train_y_hat))

            if self.test_loader is not None:
                test_iter = self.test_loader.sparse_iter()
                test_y_hat = []
                for X, y in test_iter:
                    feed_dict = dict()
                    if isinstance(X, list):
                        for idx in range(len(X)):
                            feed_dict[self.X[idx]] = X[idx]
                    else:
                        feed_dict[self.X] = X
                    y_hat = self.sess.run(self.y_preds, feed_dict=feed_dict)
                    test_y_hat.extend(y_hat)
                self.test_scores.append(roc_auc_score(self.test_loader.y, test_y_hat))
            self.time_scores.append(time.time() - self.start_time)
            if self.early_stop_round is not None:
                if num_epoch > self.early_stop_round:
                    if np.argmax(self.test_scores) == num_epoch - self.early_stop_round:
                        return