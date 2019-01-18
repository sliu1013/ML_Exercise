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
import random
from sklearn.naive_bayes import MultinomialNB


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

def data_processor(folder_path,test_size=0.2):
    '''
    :param folder_path:
    :param test_size:
    :return:
    '''
    datas_list=[]
    labels_list=[]
    folder_list = os.listdir(folder_path)
    for folder in folder_list:
        new_folder_path=os.path.join(folder_path,folder)
        files = os.listdir(new_folder_path)
        for file in files:
            with open(os.path.join(new_folder_path,file),encoding='utf-8') as f:
                raw=f.read()
                word_cut = jieba.cut(raw, cut_all=False)  # 精简模式，返回一个可迭代的generator
                word_list = list(word_cut)  # generator转换为list
                datas_list.append(word_list)
                labels_list.append(folder)
    words_dict={}
    for word_list in datas_list:
        for word in word_list:
            if word not in words_dict:
                words_dict[word]=1
            else:
                words_dict[word]+=1
    #words_dict按频次排序
    all_words_tuple_list = sorted(words_dict.items(), key=lambda f: f[1], reverse=True)
    all_words_list, all_words_nums = zip(*all_words_tuple_list)  # 解压缩
    all_words_list = list(all_words_list)  # 转换成列表

    data_label_list = list(zip(datas_list, labels_list))  # zip压缩合并，将数据与标签对应压缩
    random.seed(666)
    random.shuffle(data_label_list)  # 将data_label_list乱序
    index=int(len(data_label_list)*test_size)
    test_data_label_list=data_label_list[:index]
    train_data_label_list=data_label_list[index+1:len(data_label_list)]
    train_data_list, train_label_list = zip(*train_data_label_list)  # 训练集解压缩
    test_data_list, test_label_list = zip(*test_data_label_list)  # 测试集解压缩
    return all_words_list,train_data_list,train_label_list,test_data_list,test_label_list


def get_feature_words(all_words_list,deleteN):
    write_to_file(all_words_list,'all_words.txt')
    stop_words=get_stopwords('../stopwords_cn.txt','../stopwords_en.txt')
    feature_words=[]
    for word in all_words_list[deleteN:len(all_words_list)]:
        if word not in stop_words and 1<len(word)<5 and not word.isdigit():
            feature_words.append(word)
    #print("feature_words_dim:"+str(len(feature_words)))
    write_to_file(feature_words, 'feature_words.txt')
    return set(feature_words)


def process_feature(all_words_list,train_data_list,train_label_list,test_data_list,test_label_list,deleteN):
    label_map={'C000008':0,'C000010':1,'C000013':2,'C000014':3,'C000016':4,'C000020':5,
               'C000022':6,'C000023':7,'C000024':8}
    #词袋模型
    def text_features(text, feature_words):  # 出现在特征集中，则置1
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features
    feature_words=get_feature_words(all_words_list,deleteN)
    X_train=[text_features(train_data,feature_words) for train_data in train_data_list]
    X_test=[text_features(test_data,feature_words) for test_data in test_data_list]
    y_train=[label_map[train_label] for train_label in train_label_list]
    y_test=[label_map[test_label] for test_label in test_label_list]
    return X_train,X_test,y_train,y_test

def train(X_train,X_test,y_train,y_test):
    classifier=MultinomialNB().fit(X_train,y_train)
    train_accuracy = classifier.score(X_train, y_train)
    test_accuracy=classifier.score(X_test,y_test)
    return train_accuracy,test_accuracy


def write_to_file(words,file_name):
    with open(file_name,'w',encoding='utf-8') as f:
        for word in words:
            f.write(word+'\n')

if __name__ == "__main__":
    all_words_list, train_data_list, train_label_list, test_data_list, test_label_list=data_processor('../SogouC/Sample')
    best_accuracy=0
    best_deleteN=0
    for deleteN in range(0,400,10):
        X_train, X_test, y_train, y_test=process_feature(all_words_list,train_data_list,train_label_list,test_data_list,test_label_list,deleteN)
        train_accuracy,test_accuracy=train(X_train,X_test,y_train,y_test)
        if test_accuracy>best_accuracy:
            best_accuracy=test_accuracy
            best_deleteN=deleteN
    print(best_accuracy)
    print(best_deleteN)

