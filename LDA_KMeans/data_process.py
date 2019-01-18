#!/usr/bin/env python
#coding=utf-8
# *******************************************************************
#     Filename @  data_process.py
#       Author @  LiuSong
#  Create date @  2019/1/16 9:37
#        Email @  
#  Description @  
#      license @ (C) Copyright 2015-2018, DevOps Corporation Limited.
# ********************************************************************
import jieba
import os

jieba.suggest_freq('沙瑞金', True)
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('京州', True)

def analysis(file_path,analysis_path):
    result_list=[]
    with open(file_path,encoding='utf-8') as f:
        for line in f:
            result=' '.join(jieba.cut(line))
            result_list.append(result)

    if not os.path.exists(analysis_path):
        return
    with open(analysis_path,mode='w',encoding='utf-8') as f:
        for result in result_list:
            f.write(result)



if __name__ == "__main__":
    analysis('nlp_test.txt','nlp_analysis.txt')


