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

#数据输入
def load_data(file_path):
    """
    :param file_path: 文件路径，string
    :return:
    x:文本内容，list of string,7500
    y:文本标签，list of int,7500
    """
    x, y = [], []
    for i in range(1, 51):#读取50位用户的行为命令
        text_path = file_path + 'User{}'.format(str(i))
        label_path = file_path + 'label.txt'
        text_seq = np.loadtxt(text_path, dtype=str)
        print("Successfully load {}".format(text_path))

        text_seq = text_seq.reshape(150, 100)#100个操作命令作为一个操作序列

        for t in text_seq:
            text = " ".join(t)
            x.append(text)
        label = np.loadtxt(label_path, usecols=i-1, dtype=int).tolist()
        label = [0]*50 + label#前50个指令默认为正常操作
        y = y + label
        print("Successfully load label")

    #print(type(x), len(x), type(y), len(x))
    return x, y

#特征向量化
def get_features_by_wordbagtfidf(x, y):
    """
    3-gram词袋模型+tf-idf处理
    :param x:文本内容，list of string
    :param y: 文本标签，list of int
    :return:
    x_train :训练样本内容，array，(n_samples,output_dim)=(5250,500)
    x_test = 测试样本内容，array，(n_samples,output_dim)=(2250,500)
    y_train = 训练样本标签，array，(n_samples, )=(5250, )
    y_test = 测试样本标签，array，(n_samples, )=(2250, )
    """
    vectorizer = CountVectorizer(
        decode_error='ignore',
        ngram_range=(2, 4),
        token_pattern=r'\b\w+\b',
        strip_accents='ascii',
        max_features=max_words,
        stop_words='english',
        max_df=1.0,#作为一个阈值，词是否当作关键词。表示词出现的次数与语料库文档数的百分比
        min_df=1
    )
    print(vectorizer)

    x = vectorizer.fit_transform(x)
    transformer = TfidfTransformer(smooth_idf=False)
    x = transformer.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)  # 划分数据集
    x_train = x_train.toarray()  # 转矩阵
    x_test = x_test.toarray()
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print("Transform texts into word-bag&tf-idf successfully")
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    return x_train, y_train, x_test, y_test

def get_features_by_vocabulary(x, y):
    """
    词汇表模型向量化文本
    :param x: 所有文本内容，list
    :param y: 所有文本标签，list
    :return:
    x_train :训练样本内容，array，(n_samples,output_dim)=(5250,100)
    x_test = 测试样本内容，array，(n_samples,output_dim)=(2250,100)
    y_train = 训练样本标签，array，(n_samples, )=(5250, )
    y_test = 测试样本标签，array，(n_samples, )=(2250, )
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)  # 划分数据集
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_words)#调用Tokenizer类，初始化一个tokenizer
    tokenizer.fit_on_texts(x_train)

    x_train = tokenizer.texts_to_sequences(x_train)#向量化x_train
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=input_dim)#限制长度
    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=input_dim)

    #list转array
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    print("Transform texts into vocabulary successfully")
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test

#模型
def gnb():
    """
    wordbagtfidf-acc = 0.5124
    vocabulary-acc = 0.8796
    :return: gnb模型
    """
    model = GaussianNB()
    return model

def svm():
    """
    wordbagtfidf-acc = 0.972
    vocabulary-acc = 0.971
    :return:返回svm模型
    """
    s = SVC()
    return s

def mlp():
    """
    wordbagidf-acc = 0.9751
    :return:返回mlp模型
    """
    model = MLPClassifier(hidden_layer_sizes=(5, 2), activation='relu', solver='adam',
                          shuffle=True, verbose=True)
    return model

def cnn():
    """
    wordbagtfidf-acc =0.9667
    vocabulary-acc = 0.968
    :return: 返回cnn模型
    """
    model = keras.models.Sequential(
        [keras.layers.Embedding(input_dim=500+1, output_dim=128, input_length=100),
         keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding="valid", activation="relu", kernel_initializer="uniform"),
         keras.layers.MaxPooling1D(pool_size=2),
         keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding="valid", activation="relu", kernel_initializer="uniform"),
         keras.layers.Flatten(),
         keras.layers.Dense(units=128, activation="tanh"),
         keras.layers.Dropout(rate=0.2),
         keras.layers.Dense(units=1, activation="sigmoid")]#units=1
    )
    #编译模型
    model.summary()#输出各层的参数状况
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    return model

def lstm():
    """
    vocabulary-acc = 0.9649
    :return:返回lstm 模型
    """
    model = keras.models.Sequential(
        [keras.layers.Embedding(input_dim=max_words+1, output_dim=128, input_length=input_dim),
         keras.layers.LSTM(units=128, dropout=0.2, return_sequences=True),
         keras.layers.LSTM(units=128, dropout=0.2),
         keras.layers.Dense(units=128, activation="relu"),
         keras.layers.Dropout(0.2),
         keras.layers.Dense(units=1, activation="sigmoid")])
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    return model

#载入数据并向量化
def get_features(load_type):
    """
    :param load_type: 选择向量化的方式。0:wordbagtfidf; 1:vocabulary
    :return:
        x_train: 向量化后的所有训练样本内容，array
        x_test: 向量化后的所有测试样本内容，array
        y_train: 训练样本标签，array
        y_test: 测试样本标签，array
    """
    x, y = load_data(file_path)
    if load_type == 0:
        x_train, y_train, x_test, y_test = get_features_by_wordbagtfidf(x, y)
        return x_train, y_train, x_test, y_test
    if load_type == 1:
        x_train, y_train, x_test, y_test = get_features_by_vocabulary(x, y)
        return x_train, y_train, x_test, y_test

#训练与测试
def train(x_train, y_train, method_type, model_type, batch_size, epochs):
    """
    :param x_train:向量化后的所有训练样本内容，array，(n_samples, output_dim)
    :param y_train:训练样本标签，array, (n_smaples, )
    :param method_type:方法类型，int，0：机器学习；1：深度学习
    :param model_type:模型类型，int，00:gnb  01:svm  02:mlp  10:cnn 11:lstm
    :param batch_size:一次训练所选取的样本，int
    :param epochs:所有样本重复次数，int
    :return:model:训练好的模型
    """
    #ml or dl
    if method_type == 0:
        model = gnb()
        if model_type == 0:
            model = gnb()
        if model_type == 1:
            model = svm()
        elif model_type == 2:
            model = mlp()
        model.fit(x_train, y_train)
        return model

    elif method_type == 1:
        model = cnn()
        if model_type == 0:
            model = cnn()
        if model_type == 1:
            model = lstm()
        elif model_type == 2:
            model = dnn()
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True)
        return model

def test(x_test, y_test, trained_model, input_type):
    """
    :param x_test: 向量化后的所有测试样本内容，array，(n_samples, output_dim)
    :param y_test: 测试样本标签，array, (n_smaples, )
    :param trained_model: 训练好的模型
    :param input_type: 方法类型，int，0：机器学习；1：深度学习
    :return: accuracy:准确率
    """
    if input_type == 0:
        y_pred = trained_model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        #accuracy = model.evaluate(x_test, y_test)
        return acc
    elif input_type == 1:
        scores = trained_model.evaluate(x_test, y_test, verbose=1)
        accuracy = scores[1]
        return accuracy

if __name__=="__main__":
    file_path = '/home/hezhouyu/projects/dataset/sea/'
    max_words = 500
    input_dim = 100
    batch_size = 200
    epochs = 5

    method_type = 1   #0:'ml'  1:'dl'
    model_type =1  #00:gnb  01:svm  02:mlp  10:cnn 11:lstm
    load_type = 1 #0:wordbagtfidf 1:vocabulary
    x_train, y_train, x_test, y_test = get_features(load_type)

    print('x_train.shape = ', x_train.shape, 'y_train.shape',  y_train.shape)
    trained_model = train(x_train, y_train, method_type, model_type, batch_size, epochs)
    print(test(x_test, y_test, trained_model, method_type))
