#!/usr/bin/env python
#coding=utf-8
# *******************************************************************
#     Filename @  nbc.py
#       Author @  LiuSong
#  Create date @  2019/1/7 19:55
#        Email @  
#  Description @  
#      license @ (C) Copyright 2015-2018, DevOps Corporation Limited.
# ********************************************************************
import os
import jieba
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


def get_stopwords(cn_file_path,en_file_path):
    '''
    :param cn_file_path:
    :param en_file_path:
    :return:
    '''
    stop_words=set()
    with open(cn_file_path,encoding='utf-8') as f:
        for line in f:
            stop_words.add(line.strip())
    with open(en_file_path,encoding='utf-8') as f:
        for line in f:
            stop_words.add(line.strip())
    return stop_words

def data_processor(folder_path,stop_words):
    '''
    :param folder_path:
    :param test_size:
    :return:
    '''
    words_list=[]
    labels_list=[]
    folder_list = os.listdir(folder_path)
    for folder in folder_list:
        new_folder_path=os.path.join(folder_path,folder)
        files = os.listdir(new_folder_path)
        for file in files:
            with open(os.path.join(new_folder_path,file),encoding='utf-8') as f:
                raw=f.read()
                word_cut = jieba.cut(raw, cut_all=False)  # 精简模式，返回一个可迭代的generator
                new_word=""
                for word in word_cut:
                    if word not in stop_words and 1 < len(word) < 5:
                        new_word+=word+' '
                words_list.append(new_word.strip())
                labels_list.append(folder)
    # TF-IDF模型
    tfidf=TfidfVectorizer()
    feature_list = tfidf.fit_transform(words_list).toarray()
    return feature_list,labels_list

def process_labels(labels_list):
    label_map={'C000008':0,'C000010':1,'C000013':2,'C000014':3,'C000016':4,'C000020':5,
               'C000022':6,'C000023':7,'C000024':8}
    labels_list=[label_map[label] for label in labels_list]
    return np.array(labels_list)

def train():
    stop_words=get_stopwords('../stopwords_cn.txt','../stopwords_en.txt')
    feature_list, labels_list=data_processor('../SogouC/Sample',stop_words)
    X,y=feature_list,process_labels(labels_list)
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=666)

    classifier=MultinomialNB().fit(X_train,y_train)
    train_accuracy = classifier.score(X_train, y_train)

    test_accuracy=classifier.score(X_test,y_test)
    return train_accuracy,test_accuracy


if __name__ == "__main__":
    train_accuracy, test_accuracy=train()
    print("train_accuracy:"+str(train_accuracy))
    print("test_accuracy:"+str(test_accuracy))



