# -*- coding: utf-8 -*-

import tensorflow as tf

FEAT_NAME = './input/featindex.txt'
TRAIN_NAME = './input/train.txt'
TEST_NAME = './input/test.txt'

DTYPE = tf.float32

FIELD_SIZES = [0] * 16
with open(FEAT_NAME) as fin:
    for line in fin:
        FIELD_SIZES[int(line.strip().split(':')[0]) - 1] += 1

FIELD_OFFSETS = [sum(FIELD_SIZES[:i]) for i in range(len(FIELD_SIZES))]
INPUT_DIM = sum(FIELD_SIZES)
OUTPUT_DIM = 1

STDDEV = 1e-3
MINVAL = -1e-3
MAXVAL = 1e-3