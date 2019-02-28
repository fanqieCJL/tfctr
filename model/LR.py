# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import config
import utils
import time

class LR:
    def __init__(self, train_loader, test_loader, opt_algo='gd', 
                       learning_rate=0.01, epoch=10, early_stop_round=None, 
                       l2_w=0., random_seed=None):
        self.graph = tf.Graph()

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.var_list = [('w', [sum(config.FIELD_SIZES), 1], 'xavier'),
                         ('b', [1], 'zero')]

        self.opt_algo = opt_algo
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.early_stop_round = early_stop_round
        self.l2_w = l2_w
        self.random_seed = random_seed

        self.time_scores = []
        self.train_scores = []
        self.test_scores = []

        with self.graph.as_default():
            if self.random_seed is not None:
                tf.set_random_seed(self.random_seed)
            self.X = tf.sparse_placeholder(config.DTYPE)
            self.y = tf.placeholder(config.DTYPE)

            self.var_dict = utils.get_var(self.var_list)
            w = self.var_dict['w']
            b = self.var_dict['b']

            xw = tf.sparse_tensor_dense_matmul(self.X, w)
            logits = tf.reshape(tf.add(xw, b), [-1])
            self.y_preds = tf.sigmoid(logits)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=logits)
            ) + self.l2_w * tf.nn.l2_loss(w)

            self.optimizer = utils.get_optimizer(self.opt_algo, self.learning_rate, self.loss)

            self.sess = tf.Session()
            tf.global_variables_initializer().run(session=self.sess)
    
    def run(self):
        self.start_time = time.time()
        for num_epoch in tqdm(range(self.epoch)):
            train_iter = self.train_loader.sparse_iter()
            train_y_hat = []
            for X, y in train_iter:
                _, ll = self.sess.run([self.optimizer, self.loss], feed_dict={self.X: X, self.y: y})
                y_hat = self.sess.run(self.y_preds, feed_dict={self.X: X, self.y: None})
                train_y_hat.extend(y_hat)
            self.train_scores.append(roc_auc_score(self.train_loader.y, train_y_hat))

            if self.test_loader is not None:
                test_iter = self.test_loader.sparse_iter()
                test_y_hat = []
                for X, y in test_iter:
                    y_hat = self.sess.run(self.y_preds, feed_dict={self.X: X, self.y: None})
                    test_y_hat.extend(y_hat)
                self.test_scores.append(roc_auc_score(self.test_loader.y, test_y_hat))
            self.time_scores.append(time.time() - self.start_time)
            if self.early_stop_round is not None:
                if num_epoch > self.early_stop_round:
                    if np.argmax(self.test_scores) == num_epoch - self.early_stop_round:
                        break