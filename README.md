# tfctr

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/chaideblog/tfctr/LICENSE)

## Overview

[chaideblog/tfctr](https://github.com/chaideblog/tfctr) - Simple and useful CTR models, written in Python.

This repository maintains the code of the common CTR models, implemented with tensorflow. In this repository, we collected LR, FM, Embedding+MLP, FNN, PNN. 

## Description

> Click-through rate (CTR) is the ratio of users who click on a specific link to the number of total users who view a page, email, or advertisement. It is commonly used to measure the success of an online advertising campaign for a particular website as well as the effectiveness of email campaigns.

[This blog](https://chaideblog.github.io/) is very helpful to understand CTR. (PS: The blog is written in Chinese (中文).)

## Usage

**It is important to note that the training set and test set must follow the data format of libsvm. And the description of the features should be stored in the following format.**

```
0:other 0
0:0     1
0:1     2
1:other 3
1:00    4
1:01    5
1:02    6
2:0     7
2:1     8
```

See working example:

* [LR](https://github.com/chaideblog/tfctr/blob/master/examples/LR.ipynb)
* [FM](https://github.com/chaideblog/tfctr/blob/master/examples/FM.ipynb)
* [PNN](https://github.com/chaideblog/tfctr/blob/master/examples/PNN.ipynb)

To use these models, just load dataset and set parameters. Note that:

1. Set the path of train and test dataset in `config.py`
2. Set the description of features in `config.py`
3. Select the optimal parameters of model
4. Just run

```python
from DataLoader import DataLoader
from LR import LR

train_loader = DataLoader('./input/train.txt')
test_loader = DataLoader('./input/test.txt')

lr_params = {
    'train_loader': train_loader,
    'test_loader': test_loader,
    'opt_algo': 'gd',
    'learning_rate': 0.1,
    'epoch': 500,
    'early_stop_round': 10
}

lr_model = LR(**lr_params)
lr_model.run() 
```

This repository contains 7 .py files, they are:

* config.py: the basic parameters of models.
* DataLoader.py: the generator of dataset.
* utils.py: some helpful tools.
* LR.py: LR model.
* FM.py: FM model.
* EMLP.py: Embedding+MLP model.
* FNN.py: FNN model.
* PNN.py: PNN model.

## Reference

[1] [Zhang W , Du T , Wang J . Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction[J]. 2016.](https://arxiv.org/pdf/1601.02376.pdf)

[2] [从FM推演各深度CTR预估模型(附代码)](https://www.jiqizhixin.com/articles/2018-07-16-17)
