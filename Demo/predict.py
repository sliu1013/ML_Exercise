#!/usr/bin/env python
#coding=utf-8
# *******************************************************************
#     Filename @  predict.py
#       Author @  LiuSong
#  Create date @  2019/1/17 15:43
#        Email @  
#  Description @  
#      license @ (C) Copyright 2015-2018, DevOps Corporation Limited.
# ********************************************************************
import fastText

def train():
    classifier = fastText.train_supervised('train-segmented.txt')
    texts=['趣 多多 曲奇 饼干','银鹭 花生 牛奶 200g']
    labels = classifier.predict(texts,k=3)
    print(labels)

if __name__ == "__main__":
    train()
