#!/usr/bin/env python
#coding=utf-8
# *******************************************************************
#     Filename @  train_lda.py
#       Author @  LiuSong
#  Create date @  2019/1/16 9:54
#        Email @  
#  Description @  
#      license @ (C) Copyright 2015-2018, DevOps Corporation Limited.
# ********************************************************************
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.externals import joblib



def get_stopwords(stop_words_path):
    with open(stop_words_path,encoding='utf-8') as f:
        return f.read().splitlines()

def train():
    corpus=[]
    with open('nlp_analysis.txt',encoding='utf-8') as f:
        for line in f:
            corpus.append(line)
    stpwrdlst=get_stopwords('stop_words.txt')
    cntVector = CountVectorizer(stop_words=stpwrdlst)
    cntTf = cntVector.fit_transform(corpus)

    for num_clusters in range(2,4):
        km_cluster = KMeans(n_clusters=num_clusters)
        # 返回各自文本的所被分配到的类索引
        km_cluster.fit(cntTf)
        result = km_cluster.labels_
        # 获取聚类准则的总和(RSS)
        print(km_cluster.inertia_)
        print("Predicting result: ", result)


if __name__ == "__main__":
    train()




