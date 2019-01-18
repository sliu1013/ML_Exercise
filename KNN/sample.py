#!/usr/bin/env python
#coding=utf-8
# *******************************************************************
#     Filename @  sample.py
#       Author @  LiuSong
#  Create date @  2019/1/7 19:02
#        Email @  
#  Description @  
#      license @ (C) Copyright 2015-2018, DevOps Corporation Limited.
# ********************************************************************
import pandas as pd
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def data_processor():
    titles=['PLANE_DISTANCES','PLAY_HOURS','ICES','LABEL']
    values=pd.read_csv('datingTestSet.txt',encoding='utf-8',sep='\\s+',header=None,names=titles,engine='python')
    features_df = values.filter(items=['PLANE_DISTANCES', 'PLAY_HOURS', 'ICES'])
    label_dict={'largeDoses':0,'smallDoses':1,'didntLike':2}
    labels_df=values['LABEL'].map(label_dict)
    return features_df.values,labels_df.values


def train():
    X,y=data_processor()
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=666)
    standardScalar = StandardScaler()
    standardScalar.fit(X_train)
    X_train = standardScalar.transform(X_train)
    best_score=0
    best_neighbour=0
    for i in range(2,5,1):
        classifier=KNeighborsClassifier(n_neighbors=i)
        classifier.fit(X_train,y_train)
        score=classifier.score(X_train,y_train)
        if score>best_score:
            best_score=score
            best_neighbour=i
    return best_neighbour,best_score




if __name__ == "__main__":
    best_neighbour, best_score=train()
    print(best_neighbour)
    print(best_score)



