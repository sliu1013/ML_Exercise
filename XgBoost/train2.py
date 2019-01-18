#!/usr/bin/env python
#coding=utf-8
# *******************************************************************
#     Filename @  train_lda.py
#       Author @  LiuSong
#  Create date @  2019/1/17 18:05
#        Email @  
#  Description @  
#      license @ (C) Copyright 2015-2018, DevOps Corporation Limited.
# ********************************************************************
import xgboost as xgb
from sklearn.metrics import accuracy_score
import numpy as np

def train():

    my_workpath = 'data2/'
    train_label = np.genfromtxt(my_workpath + 'train_file', dtype=np.int32, delimiter=",", usecols= -1)
    train_feature= np.genfromtxt(my_workpath + 'train_file', dtype=np.int32, delimiter=",", usecols= range(150))

    test_label = np.genfromtxt(my_workpath + 'test_file', dtype=np.int32, delimiter=",", usecols=-1)
    test_feature = np.genfromtxt(my_workpath + 'test_file', dtype=np.int32, delimiter=",", usecols=range(150))

    dtrain = xgb.DMatrix(train_feature,train_label)
    dtest = xgb.DMatrix(test_feature,test_label)
    param = {'max_depth':5, 'eta':1, 'silent':0, 'objective':'binary:logistic' }
    tree_num=2
    bst=xgb.train(param,dtrain,tree_num)
    train_preds = bst.predict(dtrain)
    test_preds=bst.predict(dtest)

    train_predictions = [round(value) for value in train_preds]
    test_predictions = [round(value) for value in test_preds]

    y_train = dtrain.get_label()  # 值为输入数据的第一行
    y_test=dtest.get_label()
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    print("Train Accuary: %.2f%%" % (train_accuracy * 100.0))
    print("Test Accuary: %.2f%%" % (test_accuracy * 100.0))


if __name__ == "__main__":
    train()

