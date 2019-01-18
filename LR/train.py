#!/usr/bin/env python
#coding=utf-8
# *******************************************************************
#     Filename @  train_lda.py
#       Author @  LiuSong
#  Create date @  2019/1/8 18:12
#        Email @  
#  Description @  
#      license @ (C) Copyright 2015-2018, DevOps Corporation Limited.
# ********************************************************************
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def data_process(train_file_path,test_file_path):
    '''
    :param train_file_path:
    :param test_file_path:
    :return:
    '''
    train_df = pd.read_csv(train_file_path, encoding='utf-8', sep='\\s+', header=None, engine='python')
    train_features=train_df.iloc[:,:-1]
    train_labels=train_df.iloc[:,-1].values

    test_df = pd.read_csv(test_file_path, encoding='utf-8', sep='\\s+', header=None, engine='python')
    test_features = test_df.iloc[:, :-1]
    test_labels = test_df.iloc[:, -1].values
    return train_features,train_labels,test_features,test_labels

def train(X_train,y_train,X_test,y_test):
    '''
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    '''
    standardScalar = StandardScaler()
    standardScalar.fit(X_train)
    X_train = standardScalar.transform(X_train)
    classifier = LogisticRegression()
    classifier.fit(X_train,y_train)
    train_accu=classifier.score(X_train,y_train)
    print("train_accu:"+str(train_accu))
    X_test=standardScalar.transform(X_test)
    test_accu=classifier.score(X_test,y_test)
    print("test_accu:" + str(test_accu))


if __name__ == "__main__":
    X_train, y_train, X_test, y_test=data_process('horseColicTraining.txt','horseColicTest.txt')
    train(X_train, y_train, X_test, y_test)