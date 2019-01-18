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
from sklearn.decomposition import LatentDirichletAllocation



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
    for n in range(1,3):
        lda = LatentDirichletAllocation(n_components=n,learning_offset=50.,random_state=0)
        docres = lda.fit_transform(cntTf)
        print(docres)
        print(lda.perplexity(cntTf))
        #print(lda.components_)

if __name__ == "__main__":
    train()




