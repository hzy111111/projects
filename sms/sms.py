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
def load_file(file_path):
    """
    载入数据，\t可区分标签与内容
    :param file_path:str，文件路径
    :return:
    x_train, list,训练文本
    y_train,list,训练标签
    x_test,list,测试文本
    y_test,list,测试标签
    """
    x, y = [], []
    with open(file_path) as f:
        for line in f:
            line = line.strip('\n')
            label, text = line.split('\t')
            x.append(text)
            if label == 'ham':
                y.append(0)
            else:
                y.append(1)
    f.close()
    return x, y

#文本向量化
def get_features_by_wordbagtfidf(x, y):
    """
    词袋+tfidf模型向量化文本
    :param x: 所有文本内容，list
    :param y: 所有文本标签，list
    :return:
    x_train :训练样本内容，array，(n_samples,max_words)=(3344,1000)
    x_test = 测试样本内容，array，(n_samples,max_words)=(2230,1000)
    y_train = 训练样本标签，array，(n_samples, )=(3344, )
    y_test = 测试样本标签，array，(n_samples, )=(2230, )
    """
    max_features = max_words
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, shuffle=True)  # 划分数据集
    vectorzier = CountVectorizer(
        decode_error='ignore',
        strip_accents='ascii',
        max_features=max_features,
        stop_words='english',
        max_df=1.0,
        min_df=1,
        binary=True
    )
    print(vectorzier)
    x_train = vectorzier.fit_transform(x_train)#词袋模型向量化
    x_test = vectorzier.transform(x_test)

    transformer = TfidfTransformer()#tf-idf向量化
    x_train = transformer.fit_transform(x_train)
    x_test = transformer.transform(x_test)

    x_train = x_train.toarray()#转array
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
    x_train :训练样本内容，array，(n_samples,output_dim)=(3344,500)
    x_test = 测试样本内容，array，(n_samples,output_dim)=(2230,500)
    y_train = 训练样本标签，array，(n_samples, )=(3344, )
    y_test = 测试样本标签，array，(n_samples, )=(2230, )
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, shuffle=True)  # 划分数据集
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_words)#调用Tokenizer类，初始化一个tokenizer
    tokenizer.fit_on_texts(x_train)

    x_train = tokenizer.texts_to_sequences(x_train)#向量化x_train
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=output_dim)#限制长度
    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=output_dim)

    #list转array
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    print("Transform texts into vocabulary successfully")
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test

def clean_texts(texts):
    """
    清洗数据
    :param text:
    :return:
    """
    punctuation = """.,?!:;(){}[]"""

    texts = [z.lower().replace('\n', '') for z in texts]#去除\n和低频词
    texts = [z.replace('<br />', ' ') for z in texts]#去除br和‘ ’

    for p in punctuation:
        texts = [z.replace(p, '{}'.format(p)) for z in texts]
    texts = [text.split() for text in texts]

    return texts

def features_word2vec(word2vec_model, x, output_dim):
    """
    :param word2vec_model: 训练好的word2vec模型
    :param x: 输入文本，list
    :param output_dim:输出维度
    :return:
    X:向量化后的文本，array，(n_samples,output_dim)
    """
    X = []
    for text in x:
        vectors_of_one_text = np.zeros(output_dim)
        n_words = 0
        for word in text:
            try:
                vectorized_word = word2vec_model[word]#向量化word
                vectors_of_one_text += vectorized_word
                n_words += 1
            except KeyError:
                continue
        X.append(vectors_of_one_text/n_words)#一个样本append一次（该样本向量化/词数量）
    X = np.array(X, dtype='float')

    return X

def get_features_by_word2vec(x, y):
    """
    使用word2vec向量化文本
    :param x: 所有文本内容，list
    :param y: 所有文本标签，list
    :return:
    x_train :训练样本内容，array，(n_samples,output_dim)=(3344,500)
    x_test = 测试样本内容，array，(n_samples,output_dim)=(2230,500)
    y_train = 训练样本标签，array，(n_samples, )=(3344, )
    y_test = 测试样本标签，array，(n_samples, )=(2230, )
    """
    x = clean_texts(x)#清洗内容
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, shuffle=True)  # 划分数据集

    cores = multiprocessing.cpu_count()#获取CPU核数量

    word2vec_model = Word2Vec(size=output_dim, window=10, min_count=10, iter=10, workers=cores)#初始化word2vec模型
    #build and train
    word2vec_model.build_vocab(x_train)
    word2vec_model.train(x_train, total_examples=word2vec_model.corpus_count, epochs=word2vec_model.iter)
    #word2vec_model.save('')

    x_train = features_word2vec(word2vec_model, x_train, output_dim)
    x_test = features_word2vec(word2vec_model, x_test, output_dim)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    print("Transform texts into word2vec successfully")
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test

def labelizeReviews(reviews, label_type):
    """
    :param reviews:list, 待标记文本
    :param label_type:str， 当前文本标签
    :return:labelized:list,标记后文本
    """
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        labelized.append(SentimentDocument(words=v, tags=[label]))
    return labelized

def get_doc2Vecs(doc2vec_model, texts):
    """
    :param doc2vec_model:
    :param texts:
    :return:
    """
    vectors = [doc2vec_model.infer_vector(text) for text in texts]
    return np.array(vectors, dtype='float')

def get_features_by_doc2vec(x, y):
    """
    使用doc2vec向量化文本
    :param x: 所有文本内容，list
    :param y: 所有文本标签，list
    :return:
    x_train :训练样本内容，array，(n_samples,output_dim)=(3344,500)
    x_test = 测试样本内容，array，(n_samples,output_dim)=(2230,500)
    y_train = 训练样本标签，array，(n_samples, )=(3344, )
    y_test = 测试样本标签，array，(n_samples, )=(2230, )
    """
    x = clean_texts(x)#清洗内容
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, shuffle=True)  # 划分数据集
    cores = multiprocessing.cpu_count()

    x_train_labelized = labelizeReviews(x_train, 'Train')

    doc2vec_model = Doc2Vec(size=output_dim, negative=5, hs=0, min_count=2, workers=cores, iter=60)
    doc2vec_model.build_vocab(x_train_labelized)
    doc2vec_model.train(x_train_labelized, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.iter)
    doc2vec_model.save("doc2vec_bin")

    x_test = get_doc2Vecs(doc2vec_model, x_test)
    x_train = get_doc2Vecs(doc2vec_model, x_train)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    print("Transform text into doc2vec successfully")
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test

def get_features(load_type):
    """
    :param load_type: 载入类型，int
    0:词袋+tf-idf
    1:vocabulary
    2:word2vec
    3:doc2vec
    :return:
    x_train :训练样本内容，array
    x_test = 测试样本内容，array
    y_train = 训练样本标签，array
    y_test = 测试样本标签，array
    """
    x, y = load_file(file_path)
    if load_type == 0:
        x_train, y_train, x_test, y_test = get_features_by_wordbagtfidf(x, y)
        return x_train, y_train, x_test, y_test
    if load_type == 1:
        x_train, y_train, x_test, y_test = get_features_by_vocabulary(x, y)
        return x_train, y_train, x_test, y_test
    if load_type == 2:
        x_train, y_train, x_test, y_test = get_features_by_word2vec(x, y)
        return x_train, y_train, x_test, y_test
    if load_type == 3:
        x_train, y_train, x_test, y_test = get_features_by_doc2vec(x, y)
        return x_train, y_train, x_test, y_test

#模型
def gnb():
    """
    wordbagtfidf-acc=0.7825
    :return:返回贝叶斯模型
    """
    gnb = GaussianNB()
    return gnb

def dnn():
    """
    vocabulary-acc=0.8596
    doc2vec-acc=0.9466
    word2vec-acc=0.8695
    :return: 返回dnn模型
    """
    input_dim = output_dim
    model = keras.models.Sequential([
        keras.layers.Dense(units=512, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(units=1, activation='sigmoid')
    ])
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])
    print("Compile DNN model successfully")

    return model

def cnn():
    """
    wordbagtfidf-acc:0.865
    vocabulary-acc:0.9803
    :return: 返回cnn模型
    """
    input_dim = max_words#1000

    model = keras.models.Sequential(
        [keras.layers.Embedding(input_dim=input_dim+1, output_dim=128, input_length=output_dim),
         keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding="valid",
                             activation="relu", kernel_initializer="uniform"),
         keras.layers.MaxPooling1D(pool_size=2),
         keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding="valid",
                             activation="relu", kernel_initializer="uniform"),
         keras.layers.Flatten(),
         keras.layers.Dense(units=128, activation="tanh"),
         keras.layers.Dropout(rate=0.2),
         keras.layers.Dense(units=1, activation="sigmoid")]#units=1
    )
    #编译模型
    model.summary()#输出各层的参数状况
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    print("Compile CNN model successfully")

    return model

def xgb():
    """
    wordbgtfidf-acc:0.961
    doc2vec-acc:0.9565
    :return: 返回xgboosting模型
    """
    model = XGBClassifier(n_estimate=100, n_jobs=-1)
    return model

#模型的训练与验证
def train(x_train, y_train, method_type, model_type):
    """
    :param x_train:向量化后的所有训练样本内容，array，(n_samples, output_dim)
    :param y_train:训练样本标签，array, (n_smaples, )
    :param method_type:方法类型，int，0：机器学习；1：深度学习
    :param model_type:模型类型，int，00:gnb  01:xgb  10:dnn  11:cnn
    :return:model:训练好的模型
    """
    if method_type == 0:#ml
        if model_type == 0:
            model = gnb()
            model.fit(x_train, y_train)
            return model
        if model_type == 1:
            model = xgb()
            model.fit(x_train, y_train)
            return model
    if method_type == 1:#dl
        if model_type == 0:
            model = dnn()
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True)
            return model
        if model_type == 1:
            model = cnn()
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True)
            return model

def test(x_test, y_test, trained_model, method_type):
    """
    :param x_test: 向量化后的所有测试样本内容，array，(n_samples, output_dim)
    :param y_test:测试样本标签，array, (n_smaples, )
    :param trained_model: 训练好的模型
    :param method_type: 方法类型  0：machine learning 1：deep learning
    :return:
    """
    if method_type == 0:#ml
        y_pred = trained_model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        return acc
    if method_type == 1:
        scores = trained_model.evaluate(x_test, y_test, verbose=1)
        acc = scores[1]
        return acc

if __name__ == "__main__":
    file_path = "/home/hezhouyu/projects/dataset/sms/SMSSpamCollection.txt"
    SentimentDocument = namedtuple('SentimentDocument', 'words tags')
    max_words = 1000
    output_dim = 500
    batch_size = 200
    epochs = 4

    load_type = 0    # 0:词袋+tf-idf 1:vocabulary 2:word2vec 3:doc2vec
    method_type = 0  # 0：machine learning 1：deep learning
    model_type = 1   # 00:gnb  01:xgb  10:dnn  11:cnn

    x_train, y_train, x_test, y_test = get_features(load_type)
    trained_model = train(x_train, y_train, method_type, model_type)
    acc = test(x_test, y_test, trained_model, method_type)
    print(acc)
