#coding=utf-8
import tensorflow as tf
import os
import numpy as np
import gensim
import re
#import commands
import subprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import shuffle
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import accuracy_score

#数据输入
def load_one_file(filename):
    """
    :param filename:文件名，string
    :return:x:文件内容，string
    """
    one_x = ''
    with open(filename, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip('\r')
            line = line.strip('\n')
            one_x += line
    return one_x

def load_files_from_dir(rootdir):
    """
    :param rootdir:文件路径，string
    :return: x:文件内容，list of string
    """
    x = []
    for root, dir_list, file_list in os.walk(rootdir):# 每次遍历的路径名、路径下子目录列表、目录下文件列表。
        for file in file_list:
            if file.endswith(".php") or file.endswith(".txt"):
                file_path = os.path.join(root, file)
                print("The current file path: {}".format(file_path))

                one_x = load_one_file(file_path)
                x.append(one_x)
    return x

def load_one_file_opcede(filename, php_bin):
    """
    :param filename: 文件路径，string
    :param php_bin:php7.2路径 ，string
    :return:
    t:opcode解析php后的文本, string
    n_token = 当前php文件中opcodes的数量
    """
    cmd = php_bin + " -dvld.active=1 -dvld.execute=0 " + filename
    t = ""
    try:
        status, output = subprocess.getstatusoutput(cmd)
        print("successfully get the opcodes of {}".format(filename))
    #t = output
    #print(t)
        tokens = re.findall(r'\s(\b[A-Z_]+\b)\s', output)
        n_token = len(tokens)
        print("opcode count %d" % len(t))
        t = " ".join(tokens)
    except UnicodeDecodeError:
        t = ""
        n_token = 0
    return t, n_token

def load_files_from_dir_opcode(rootdir, php_bin):
    """
    :param rootdir: 文件路径，string
    :param php_bin: php2.7路径，string
    :return:
    x：解析php文件后的文本，string
    """
    min_opcode_count = 2
    x = []
    for root, dir_list, file_list in os.walk(rootdir):
        for file in file_list:
            if file.endswith('.php'):
                file_path = os.path.join(root, file)
                print("The current file path: {}".format(file_path))
                one_x, n_tokens = load_one_file_opcede(file_path, php_bin)

                if n_tokens > min_opcode_count:#当前文本中opcode数量过少则不计入
                    x.append(one_x)
                else:
                    print("load {} opcode failed".format(file_path))
    return x

#特征向量化
def get_features_by_vocabulary(x_train, x_test, y_train, y_test):
    """
    词汇表模型向量化文本
    :param x: 所有文本内容，list
    :param y: 所有文本标签，list
    :return:
    x_train :训练样本内容，array，(n_samples,output_dim)
    x_test = 测试样本内容，array，(n_samples,output_dim)
    y_train = 训练样本标签，array，(n_samples, )
    y_test = 测试样本标签，array，(n_samples, )
    """
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


def get_features(files_path):
    """
    :param files_path:文件路径，string
    :return:
    x_train :训练样本内容，array，(n_samples,output_dim)
    x_test = 测试样本内容，array，(n_samples,output_dim)
    y_train = 训练样本标签，array，(n_samples, )
    y_test = 测试样本标签，array，(n_samples, )
    """
    webshell_path = files_path + "webshell/"
    normal_path = files_path + "normal/"

    webshell_x = load_files_from_dir(webshell_path)
    normal_x = load_files_from_dir(normal_path)

    y = [1]*len(webshell_x) + [0]*len(normal_x)
    x = webshell_x + normal_x

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)  # 划分数据集
    x_train, y_train, x_test, y_test = get_features_by_vocabulary(x_train, x_test, y_train, y_test)

    return x_train, x_test, y_train, y_test

def get_features_opcode(files_path, php_bin):
    """
    :param rootdir: 文件路径，string
    :param php_bin: php2.7路径，string
    :return:
    x_train :训练样本内容，array，(n_samples,output_dim)
    x_test = 测试样本内容，array，(n_samples,output_dim)
    y_train = 训练样本标签，array，(n_samples, )
    y_test = 测试样本标签，array，(n_samples, )
    """
    webshell_path = files_path + "webshell/"
    normal_path = files_path + "normal/"

    webshell_x = load_files_from_dir_opcode(webshell_path, php_bin)
    normal_x = load_files_from_dir_opcode(normal_path, php_bin)

    y = [1]*len(webshell_x) + [0]*len(normal_x)
    x = webshell_x + normal_x

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)  # 划分数据集
    if load_type == 0:
        x_train, y_train, x_test, y_test = get_features_by_vocabulary(x_train, x_test, y_train, y_test)
    return x_train, x_test, y_train, y_test
#模型
def lstm():
    """
    vocabulary-acc = 0.8968
    opcode-acc = 0.9338
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

#训练与测试
def train(x_train, y_train):
    """
    :param x_train :训练样本内容，array，(n_samples,output_dim)
    :param y_train:训练样本标签，array，(n_samples, )
    :return:
    model:训练好的模型
    """
    model = lstm()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True)

    return model

def test(trained_model, x_test, y_test):
    """
    :param trained_model: 训练好的模型
    :param x_test:测试样本内容，array，(n_samples,output_dim)
    :param y_test: 测试样本标签，array，(n_samples, )
    :return:
    acc:准确率，float
    """
    scores = trained_model.evaluate(x_test, y_test, verbose=1)
    acc = scores[1]
    return acc

if __name__ == '__main__':
    files_path = '/home/hezhouyu/projects/dataset/web_shell/'
    php_bin = "/usr/bin/php7.2"
    max_words = 500
    input_dim = 100
    batch_size = 300
    epochs = 15

    #x_train, x_test, y_train, y_test = get_features()    使用vocabulary向量化文本
    x_train, x_test, y_train, y_test = get_features_opcode(files_path, load_type, php_bin)#使用opcode+vocabulary向量化文本
    trained_model = train(x_train, y_train)
    acc = test(trained_model, x_test, y_test)
    print(acc)

