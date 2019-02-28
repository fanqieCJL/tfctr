# tfctr

## Overview

![chaideblog/tfctr](https://github.com/chaideblog/tfctr) - Simple and useful CTR models, written in Python.

This repository maintains the code of the common CTR models, implemented with tensorflow. In this repository, we collected LR, FM, Embedding+MLP, FNN, PNN. 

## Description

> Click-through rate (CTR) is the ratio of users who click on a specific link to the number of total users who view a page, email, or advertisement. It is commonly used to measure the success of an online advertising campaign for a particular website as well as the effectiveness of email campaigns.

![This blog](https://chaideblog.github.io/) is very helpful to understand CTR. (PS: The blog is written in Chinese (中文).)

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
2:0    7
2:1    8
```