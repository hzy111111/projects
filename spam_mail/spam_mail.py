#coding=utf-8

import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


#数据输入
def load_one_file(filename):
    """
    :param filename:文件名， str
    :return: x:一个邮件的内容，str
    将整个邮件当作一个字符串处理，所以需要过滤回车和换行
    """
    x = ""
    with open(filename, encoding='utf-8', errors='ignore') as f: #报错原因：errors为严格（strict）形式
        for line in f:
            line = line.strip('\n')
            line = line.strip('\t')
            x += line
    return  x
def load_files_from_dir(rootdir):
    """
    :param rootdir:文件路径， str
    :return: x_list:一个文件夹下的所有文本， list
    """
    x = []
    list = os.listdir(rootdir)
    for i in range(len(list)):
        file_path = os.path.join(rootdir, list[i])
        if os.path.isfile(file_path):
            v = load_one_file(file_path)
            x.append(v)
    return x

def load_all_files():
    """
    :return:
    ham:所有的正常邮件， list
    spam:所有的垃圾邮件， list
    """
    ham = []
    spam = []
    for i in range(1, 7):
        path = "/home/hezhouyu/projects/dataset/enron/enron%d/ham/" % i
        print("load %s" % path)
        ham += load_files_from_dir(path)
        path = "/home/hezhouyu/projects/dataset/enron/enron%d/spam/" % i
        print("load %s" % path)
        spam += load_files_from_dir(path)
    return ham, spam

#数据向量化
def get_features_by_wordbag():
    """
    使用词袋模型向量化样本，正常邮件标签为0，垃圾邮件标签为1
    :return:
    x:训练样本， array，(n_samples, max_features)
    y:标签， array，(n_samples,)
    """
    ham, spam = load_all_files()
    x = ham+spam
    y = [0]*(len(ham))+[1]*(len(spam))
    vectorizer = CountVectorizer(
        decode_error='ignore',
        strip_accents='ascii',
        max_features=max_features,#词袋特征个数最大值，5000
        stop_words='english',#使用内置英文停用词
        max_df=1.0,
        min_df=1)
    print(vectorizer)
    x = vectorizer.fit_transform(x)#标准化
    x = x.toarray()
    y = np.asarray(y)#y的类型转换为arrary
    return x, y

def get_features_by_tf():
    """
    使用tf-dif模型向量化样本
    :return:
    x:训练样本， array，(n_samples,max_features)
    y:标签， array，(n_samples,)
    """
    ham, spam = load_all_files()
    x = ham+spam
    y = [0]*(len(ham))+[1]*(len(spam))
    tfidf_vectorizer = TfidfVectorizer(decode_error='ignore',
                                       strip_accents='ascii',
                                       max_features=max_features,
                                       stop_word='english',
                                       binary=False)
    x = tfidf_vectorizer.fit_transform(x)
    x = x.toarray()
    y = np.asarray(y)
    return x, y

def get_features_by_vocabulary():
    """
    使用词汇表模型向量化样本
    :return:
    """
    ham, spam = load_all_files()
    x = ham+spam
    y = [0]*(len(ham))+[1]*(len(spam))
    y = np.asarray(y)
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_features)#初始化tokenizer分类器，参数设置文本的最大长度500
    tokenizer.fit_on_texts(x)

    x = tokenizer.texts_to_sequences(x)#将多个文档转换为向量形式
    x = keras.preprocessing.sequence.pad_sequences(x, maxlen=max_document_length)#将序列填充到maxlen长度

    return x, y

#模型
def gnb_train():
    """
    朴素贝叶斯
    """
    x, y = get_features_by_wordbag()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    gnb = GaussianNB()
    return gnb
    #0.9363090383332098

def mlp_sklearn_train():
    """
    :param x_train:训练样本特征， array，(n_samples, max_features)
    :param x_test: 测试样本特征，array，（13487，5000）
    :param y_train: 训练样本标签，list，(20229,)
    :param y_test: 测试 样本标签，list，（13487，）
    """
    x, y = get_features_by_wordbag()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    clf = MLPClassifier(solver='lbfgs',
                        alpha=0.00005,
                        hidden_layer_sizes=(5, 2),
                        random_state=1)
    return clf
    #0.9802773040705864

def ml_train_test(flag):
    x, y = get_features_by_wordbag()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    if flag == 1:
        model = gnb_train()
    if flag == 2:
        model = mlp_sklearn_train()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(metrics.accuracy_score(y_test, y_pred))



def cnn():
    """
    卷积神经网络
    """
    input_dim = max_features
    input_length = max_document_length
    model = keras.models.Sequential(
        [keras.layers.Embedding(input_dim=input_dim+1, output_dim=128, input_length=input_length),#input_dim为词汇表大小，最大整数index+1
         keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding="valid", activation="relu", kernel_initializer="uniform"),
         keras.layers.Conv1D(filters=128, kernel_size=4, strides=1, padding="valid", activation="relu", kernel_initializer="uniform"),
         keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding="valid", activation="relu",kernel_initializer="uniform"),
         keras.layers.MaxPooling1D(pool_size=2),
         keras.layers.Flatten(),
         keras.layers.Dense(units=128, activation="tanh"),
         keras.layers.Dropout(rate=0.8),
         keras.layers.Dense(units=1, activation="sigmoid")]#units=1
    )
    #编译模型
    model.summary()#输出各层的参数状况
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    return model

"""
def cnn_train_test():
    x, y = get_features_by_vocabulary()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    #shape分别是 （20229，100）（13487，100）（20299，）（13487，）
    model = cnn()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    scores = trained_model.evaluate(x_test, y_test)
    print(scores[1])  #loss: 0.0123 - acc: 0.9964
    return model
"""

def lstm():
    input_dim = max_features
    input_length = max_document_length
    model = keras.models.Sequential(
        [keras.layers.Embedding(input_dim=input_dim+1, output_dim=128, input_length=input_length),
         keras.layers.LSTM(units=128, dropout=0.2, return_sequences=True),
         keras.layers.LSTM(units=128, dropout=0.2),
         keras.layers.Dense(units=128, activation="relu"),
         keras.layers.Dropout(0.2),
         keras.layers.Dense(units=1, activation="sigmoid")])
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

    return model
"""
def lstm_train_test():
    x, y = get_features_by_vocabulary()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    #shape分别是 （20229，100）（13487，100）（20299，）（13487，）
    model = lstm()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    scores = trained_model.evaluate(x_test, y_test)
    print(scores[1])  #loss: 0.0209 - acc: 0.9933
    return model
"""

def dl_train_test(flag):
    x, y = get_features_by_vocabulary()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    if flag == 1:
        model = cnn()
    if flag == 2:
        model = lstm()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    scores = trained_model.evaluate(x_test, y_test)
    print(scores[1])  #loss: 0.0209 - acc: 0.9933

if __name__=='__main__':
    max_features = 5000
    max_document_length = 100

    batch_size = 500
    epochs = 5

    ml_train_test(1) #1为gnb,2为mlp
    #dl_train_test(2)  #1为cnn,2为lstm



