#!/usr/bin/env python
#coding=utf-8
# *******************************************************************
#     Filename @  train_lda.py
#       Author @  LiuSong
#  Create date @  2019/1/16 19:57
#        Email @  
#  Description @  
#      license @ (C) Copyright 2015-2018, DevOps Corporation Limited.
# ********************************************************************
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


def process_categoryID(id):
    return '__label__'+str(id)


def data_process_pandas(file_path,train_file_path,test_file_path):
    titles=['ID','title','CategoryID','CategoryName']
    df = pd.read_csv(file_path, encoding='utf-8', sep='\t',header=0,names=titles, engine='python')
    df['CategoryID']=df['CategoryID'].apply(process_categoryID)
    filter_df = df.filter(items=['title', 'CategoryID'])
    train_df,test_df=train_test_split(filter_df,test_size=0.2, random_state=666)
    train_df.to_csv(train_file_path,encoding='utf-8',sep='\t',header=False,index=False)
    test_df.to_csv(test_file_path,encoding='utf-8',sep='\t',header=False,index=False)



def data_process(file_path):
    corpus=[]
    labels_list=[]
    with open(file_path,encoding='utf-8') as f:
        i=0
        for line in f:
            i=i+1
            if i==1:
                continue
            values=line.split('\t')
            if len(values)!=4:
                continue
            corpus.append(values[1])
            labels_list.append(values[2])
    # TF-IDF模型
    tfidf = TfidfVectorizer()
    feature_list = tfidf.fit_transform(corpus).toarray()
    return feature_list, labels_list

def train():
    X, y = data_process('listing-segmented.txt')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

    classifier = MultinomialNB().fit(X_train, y_train)
    train_accuracy = classifier.score(X_train, y_train)

    test_accuracy = classifier.score(X_test, y_test)
    print('train_acc:' + str(train_accuracy))
    print('test_acc:' + str(test_accuracy))





if __name__ == "__main__":
    data_process_pandas('listing-segmented.txt','train-segmented.txt','test-segmented.txt')
    #train()



