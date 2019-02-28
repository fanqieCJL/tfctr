# -*- coding: utf-8 -*-
import math
import numpy as np
from scipy.sparse import coo_matrix
import config
import utils

class DataLoader:
    def __init__(self, file_name, batch_size=1024, field=False):
        self.file_name = file_name # 数据集的路径
        self.batch_size = batch_size # 批大小
        self.field = field # 是否按 field 输出

        self.X, self.y = self.read_data(self.file_name) # 从文件中读取数据, 输出 CSR 稀疏数组
        self.size = self.X.shape[0] # 数据的条数

    def sparse_iter(self):
        if self.field:
            for num in range(math.ceil(self.size / self.batch_size)):
                start = num * self.batch_size
                end = (num + 1) * self.batch_size
                X_on_field = self.split_X_on_field(self.X[start:end])
                yield self.csr2input(X_on_field), self.y[start:end]
        else:
            for num in range(math.ceil(self.size / self.batch_size)):
                start = num * self.batch_size
                end = (num + 1) * self.batch_size
                yield self.csr2input(self.X[start:end]), self.y[start:end]

    def normal_iter(self):
        for num in range(math.ceil(self.size / self.batch_size)):
            start = num * self.batch_size
            end = (num + 1) * self.batch_size
            yield self.X[start:end], self.y[start:end]

    def shuffle(self, data):
        X, y = data
        ind = np.arange(X.shape[0])
        for i in range(7):
            np.random.shuffle(ind)
        return X[ind], y[ind]

    def split_X_on_field(self, X, skip_empty=True):
        '''
            数据切分
        '''
        fields = []
        for i in range(len(config.FIELD_OFFSETS) - 1):
            start_ind = config.FIELD_OFFSETS[i]
            end_ind = config.FIELD_OFFSETS[i + 1]
            if skip_empty and start_ind == end_ind:
                continue
            field_i = X[:, start_ind:end_ind]
            fields.append(field_i)
        fields.append(X[:, config.FIELD_OFFSETS[-1]:])
        return fields

    
    def read_data(self, file_name):
        '''
            读取libsvm格式数据，储存成稀疏格式
            0 5:1 9:1 140858:1 445908:1 446177:1 491668:1 491700:1 491708:1
        '''
        X = []
        D = []
        y = []
        with open(file_name) as fin:
            for line in fin:
                fields = line.strip().split()
                y_i = int(fields[0])
                X_i = [int(x.split(':')[0]) for x in fields[1:]]
                D_i = [int(x.split(':')[1]) for x in fields[1:]]
                y.append(y_i)
                X.append(X_i)
                D.append(D_i)
        y = np.reshape(np.array(y), [-1])
        X = self.libsvm2coo(zip(X, D), (len(X), config.INPUT_DIM)).tocsr()
        return X, y

    def libsvm2coo(self, libsvm_data, shape):
        '''
            libsvm格式转成coo稀疏存储格式
        '''
        coo_rows = []
        coo_cols = []
        coo_data = []
        n = 0
        for x, d in libsvm_data:
            coo_rows.extend([n] * len(x))
            coo_cols.extend(x)
            coo_data.extend(d)
            n += 1
        coo_rows = np.array(coo_rows)
        coo_cols = np.array(coo_cols)
        coo_data = np.array(coo_data)
        return coo_matrix((coo_data, (coo_rows, coo_cols)), shape=shape)

    def csr2input(self, csr_mat):
        if not isinstance(csr_mat, list):
            coo_mat = csr_mat.tocoo()
            indices = np.vstack((coo_mat.row, coo_mat.col)).transpose()
            values = csr_mat.data
            shape = csr_mat.shape
            return indices, values, shape
        else:
            inputs = []
            for csr_i in csr_mat:
                inputs.append(self.csr2input(csr_i))
            return inputs

