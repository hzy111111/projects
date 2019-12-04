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
import gensim
import re
from collections import namedtuple
from sklearn.utils import shuffle
import multiprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from xgboost import XGBClassifier
from sklearn import preprocessing

#数据载入
def load_alexa():
    """
    :return:
    x：list, 全部alexa正常域名文本
    """
    data = pd.read_csv(alexa_path, sep=",", header=None)
    x = [i[1] for i in data.values]
    print("load alexa.csv successfully")
    return x

def load_dga():
    """
    :return:
    x:list，全部dga生成域名文本
    """
    data = pd.read_csv(dga_path, sep="\t", header=None, skiprows=18)#前18行为注释
    x = [i[1] for i in data.values]
    print("load dga successfully")
    return x

def get_data():
    """
    :return:
    x: list, 总的文本
    y：list, 对应标签，dga文件为1
    """
    alexa_x = load_alexa()
    dga_x = load_dga()
    x = alexa_x + dga_x
    y = [0]*len(alexa_x) + [1]*len(dga_x)

    return x, y

#特征提取
def get_features_ngram(x, y):
    """
    3-gram词袋模型
    :param x:文本内容，list of string
    :param y: 文本标签，list of int
    :return:
    x_train :训练样本内容，array，(n_samples,output_dim)=(13987,500)
    x_test = 测试样本内容，array，(n_samples,output_dim)=(5995,500)
    y_train = 训练样本标签，array，(n_samples, )=(13987, )
    y_test = 测试样本标签，array，(n_samples, )=(5995, )
    """
    vectorizer = CountVectorizer(
        decode_error='ignore',
        ngram_range=(2, 4),
        token_pattern=r'\w',
        strip_accents='ascii',
        max_features=max_words,
        stop_words='english',
        max_df=1.0,#作为一个阈值，词是否当作关键词。表示词出现的次数与语料库文档数的百分比
        min_df=1
    )
    print(vectorizer)
    x = vectorizer.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)  # 划分数据集
    x_train = x_train.toarray()  # 转矩阵
    x_test = x_test.toarray()
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print("Transform texts into 3-gram successfully")
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    return x_train, y_train, x_test, y_test

def get_aeiou(text):
    """
    :param text: str, 当前文本
    :return: count：int, 文本中元音字母的个数
    """
    count = len(re.findall(r'[aeiou]', text.lower()))#lower()所有大写转小写
    return count
def get_uniq_char_num(text):
    """
    :param text:当前文本
    :return: count：不重复字母的个数
    """
    count = len(set(text))
    return count
def get_num(text):
    """
    :param text: 当前文本
    :return: count:数字的个数
    """
    count = len(re.findall(r'[1234567890]', text.lower()))
    return count

def get_feature_statistic(x, y):
    """
    :param x:文本内容，list of string
    :param y: 文本标签，list of int
    :return:
    x_train :训练样本内容，array，(n_samples,n_statistic)=(13987,4)
    x_test = 测试样本内容，array，(n_samples,)=(5995,4)
    y_train = 训练样本标签，array，(n_samples, )=(13987, )
    y_test = 测试样本标签，array，(n_samples, )=(5995, )
    """
    X = []
    for xx in x:
        xxx = [get_aeiou(xx), get_uniq_char_num(xx), get_num(xx), len(xx)]
        X.append(xxx)
    X = preprocessing.scale(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)  # 划分数据集
    #x_train = x_train.toarray()  # 转矩阵
    #x_test = x_test.toarray()
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print("Transform texts into statistic-nums successfully")
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    return x_train, y_train, x_test, y_test

def get_features(x, y):
    """
    :param x:文本内容，list of string
    :param y: 文本标签，list of int
    :return:
    x_train :训练样本内容，array
    x_test = 测试样本内容，array
    y_train = 训练样本标签，array
    y_test = 测试样本标签，array
    """
    if load_type == 0:
        x_train, y_train, x_test, y_test = get_features_ngram(x, y)
        return x_train, y_train, x_test, y_test
    if load_type == 1:
        x_train, y_train, x_test, y_test = get_feature_statistic(x, y)
        return x_train, y_train, x_test, y_test

#模型的训练与验证
def gnb():
    """
    3gramwordbag-acc = 0.9217
    statistics-acc = 0.7073
    :return: gnb模型
    """
    model = GaussianNB()
    return model

def xgb():
    """
    3gramwordbag-acc:0.916
    :return: 返回xgboosting模型
    """
    model = XGBClassifier(n_estimate=150, max_depth=9)#n_estimate:决策树个数
    return model

def mlp():
    """
    statistics-acc = 0.8529
    :return:返回mlp模型
    """
    model = MLPClassifier(hidden_layer_sizes=(5, 2), activation='relu', solver='adam',
                          shuffle=True, verbose=1)
    return model

def lstm():
    """
    3gramwordbag-acc = 0.65
    :return:返回lstm 模型
    """
    input_dim = max_words
    max_features = 100
    model = keras.models.Sequential(
        [keras.layers.Embedding(input_dim=max_features+1, output_dim=128, input_length=input_dim),
         keras.layers.LSTM(units=128, dropout=0.2, return_sequences=True),
         keras.layers.LSTM(units=128, dropout=0.2),
         keras.layers.Dense(units=128, activation="relu"),
         keras.layers.Dropout(0.2),
         keras.layers.Dense(units=1, activation="sigmoid")])
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    return model

def train(x_train, y_train):
    """
    :param x_train: 训练样本内容，array
    :param y_train: 训练样本标签，array
    :return: model：训练好的模型
    """
    if method_type == 0:
        model = gnb()
        if model_type == 0:
            model = gnb()
        if model_type == 1:
            model = xgb()
        if model_type == 2:
            model = mlp()
        model.fit(x_train, y_train)
        return model

    if method_type == 1:
        model = lstm()
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
        return model

def test(model, x_test, y_test):
    """
    :param model: 训练好的模型
    :param x_test: 测试样本内容，array
    :param y_test: 测试样本标签，array
    :return: acc:准确率，float
    """
    acc = 0.0
    if method_type == 0:
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
    if method_type == 1:
        scores = trained_model.evaluate(x_test, y_test, verbose=1)
        acc = scores[1]
    return acc

if __name__=='__main__':
    alexa_path = '/home/hezhouyu/projects/dataset/dga/alexa.csv'
    dga_path = '/home/hezhouyu/projects/dataset/dga/dga.txt'

    load_type = 0
    method_type = 1
    model_type = 2

    max_words = 500
    input_dim = 100
    batch_size = 200
    epochs = 1

    x, y = get_data()
    x_train, y_train, x_test, y_test = get_features(x, y)
    trained_model = train(x_train, y_train)
    acc = test(trained_model, x_test, y_test)
    print(acc)
