#coding=utf-8

import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import gensim
import re
from collections import namedtuple
from gensim.models import Doc2Vec
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from sklearn.utils import shuffle
import multiprocessing
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score

#数据输入
def load_one_file(filename):
    """
    :param filename:文本文件名，string
    :return: one_x ：string
    """
    with open(filename) as f:
        lines = f.readlines()
    one_x = " ".join(lines)
    return one_x

def load_files_from_dir(rootdir):
    """
    :param rootdir: 文件夹名,string
    :return: x,提取的文本内容，list of string
    """
    x = []
    list = os.listdir(rootdir)
    for i in range(len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            v = load_one_file(path)
            x.append(v)
    return x

def load_files(file_path):
    """
    :param file_path:文件路径,string
    :return:
    x,文本内容，list of string
    y,文本标签，list of int
    """

    attack_text_dir_path = file_path + 'Attack_Data_Master/'
    attack_text = []
    list = os.listdir(attack_text_dir_path)

    for i in range(len(list)):
        attack_text_path = os.path.join(attack_text_dir_path, list[i])+'/'  #这个文件夹多一层遍历
        print("load %s " % attack_text_path)
        attack_one_text = load_files_from_dir(attack_text_path)
        attack_text += attack_one_text

    normal_text_path = file_path + 'Training_Data_Master/'
    print("load %s " % normal_text_path)
    normal_text = load_files_from_dir(normal_text_path)

    print(len(normal_text), len(attack_text))          #len(normal_text)=833 len(attack_text)=746
    x = normal_text + attack_text         #加一起重新划分数据集
    y = [0]*len(normal_text) + [1]*len(attack_text)

    print(type(x), x)
    return x, y

#向量化
def get_features_by_wordbagtfidf(x, y):
    """
    3-gram词袋模型+tf-idf处理
    :param x:文本内容，list of string
    :param y: 文本标签，list of int
    :return:
    x_train :训练样本内容，array，(n_samples,output_dim)=(1105, 1000)
    x_test = 测试样本内容，array，(n_samples,output_dim)=(474,1000)
    y_train = 训练样本标签，array，(n_samples, )=(1105, )
    y_test = 测试样本标签，array，(n_samples, )=(474, )
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)  # 划分数据集

    vectorizer = CountVectorizer(
        decode_error='ignore',
        ngram_range=(3, 3),#3-gram,保留局部序关系
        token_pattern=r'\b\d+\b',#过滤规则，单词切分使用的正则表达式
        strip_accents='ascii',
        max_features=max_words,
        stop_words='english',
        max_df=1.0,
        min_df=1
    )
    print(vectorizer)

    x_train = vectorizer.fit_transform(x_train)#词袋模型向量化
    x_test = vectorizer.transform(x_test)

    transformer = TfidfTransformer(smooth_idf=False)#tf-idf处理
    x_train = transformer.fit_transform(x_train)
    x_test =transformer.fit_transform(x_test)

    x_train = x_train.toarray()#转矩阵
    x_test = x_test.toarray()
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print("Transform texts into word-bag&tf-idf successfully")
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    return x_train, y_train, x_test, y_test

#模型
def gnb():
    """
    acc = 0.9177
    :return: gnb模型
    """
    model = GaussianNB()

    return model

def xgb():
    """
    acc = 0.9536
    :return: 返回xgboosting模型
    """
    model = XGBClassifier(n_estimate=100, n_jobs=-1)
    return model

def mlp():
    """
    acc = 0.9346
    :return:返回mlp模型
    """
    model = MLPClassifier(hidden_layer_sizes=(5, 2), activation='relu', solver='adam',
                          shuffle=True, verbose=True)
    return model

#训练与测试
def train_test(x, y, method_type):
    """
    :param x:文本内容，list of string
    :param y: 文本标签，list of int
    :param method_type: 方法，0:gnb(); 1:xgb(); 2:mlp()
    :return:acc: 准确率
    """
    x_train, y_train, x_test, y_test = get_features_by_wordbagtfidf(x, y)
    model = gnb()
    if method_type == 0:
        model = gnb()
    if method_type == 1:
        model = xgb()
    if method_type == 2:
        model = mlp()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    return acc


if __name__=="__main__":
    file_path = '/home/hezhouyu/projects/dataset/ADFA-LD/'
    x, y = load_files(file_path)
    max_words = 1000
    output_dim = 500

    method_type = 2
    acc = train_test(x, y, method_type)
    print(acc)

