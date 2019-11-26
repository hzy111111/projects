#coding=utf-8

import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
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
from gensim.models import Doc2Vec
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from sklearn.utils import shuffle
import multiprocessing
from sklearn.neural_network import MLPClassifier
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score


#数据输入
def load_one_file(filename):
    """
    :param filename:文件名，string
    :return:x:文件内容，string
    """
    x = ""
    with open(filename, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip('\n')
            line = line.strip('\r')
            x += line
    f.close()
    return x

def load_files_from_dir(rootdir):
    """
    :param rootdir:文件路径，string
    :return: x:文件内容，list of string
    """
    x = []
    list = os.listdir(rootdir)
    for i in range(len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            v = load_one_file(path)
            x.append(v)
    return x

def load_file(file_path):
    """
    :param file_path:文件路径，string
    :return:
    pos_text_train，训练正样本内容，list
    neg_text_train，训练负样本内容，list
    pos_text_test，测试正样本内容，list
    neg_text_test，测试负样本标签，list
    """

    pos_train_path = file_path + "train/pos/"
    neg_train_path = file_path + "train/neg/"
    pos_test_path = file_path + "test/pos/"
    neg_test_path = file_path + "test/neg/"

    print("load %s" % pos_train_path)
    pos_text_train = load_files_from_dir(pos_train_path)
    print("load %s" % neg_train_path)
    neg_text_train = load_files_from_dir(neg_train_path)
    print("load %s" % pos_test_path)
    pos_text_test = load_files_from_dir(pos_test_path)
    print("load %s" % neg_test_path)
    neg_text_test = load_files_from_dir(neg_test_path)

    return pos_text_train, neg_text_train, pos_text_test, neg_text_test

#特征提取
def get_features_by_wordbag(pos_text_train, neg_text_train, pos_text_test, neg_text_test):
    """
    :param :
    pos_text_train，训练正样本内容，list
    neg_text_train，训练负样本内容，list
    pos_text_test，测试正样本内容，list
    neg_text_test，测试负样本标签，list
    :return: 特征向量化后
    x_train, 训练样本内容，array，(n_samples, output_dim)
    y_train, 训练样本标签，array，(n_samples, )
    x_test, 测试样本内容，array，(n_samples, output_dim)
    y_test,测试样本标签，array，(n_samples, )
    """
    x_train = pos_text_train + neg_text_train
    x_test = pos_text_test + neg_text_test
    vectorizer = CountVectorizer(
        decode_error='ignore',
        strip_accents='ascii',
        max_features=output_dim,
        stop_words='english',
        max_df=1.0,
        min_df=1)
    print(vectorizer)
    x_train = vectorizer.fit_transform(x_train)
    x_train = x_train.toarray()
    vocabulary = vectorizer.vocabulary_#复用vectorizer对测试集进行词袋化
    vectorizer = CountVectorizer(
        decode_error='ignore',
        strip_accents='ascii',
        vocabulary=vocabulary,
        max_features=output_dim,
        stop_words='english',
        max_df=1.0,
        min_df=1)
    x_test = vectorizer.fit_transform(x_test)
    x_test = x_test.toarray()
    y_train = [0]*len(pos_text_train) + [1]*len(neg_text_train)
    y_train = np.asarray(y_train)
    y_test = [0]*len(pos_text_test) + [1]*len(neg_text_test)
    y_test = np.asarray(y_test)
    #打乱
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    return x_train, y_train, x_test, y_test

def get_features_by_vocabulary(pos_text_train, neg_text_train, pos_text_test, neg_text_test):
    """
    :param     pos_text_train，训练正样本内容，list
    neg_text_train，训练负样本内容，list
    pos_text_test，测试正样本内容，list
    neg_text_test，测试负样本标签，list
    :return: 特征向量化后
    x_train, 训练样本内容，array，(n_samples, output_dim)
    y_train, 训练样本标签，array，(n_samples, )
    x_test, 测试样本内容，array，(n_samples, output_dim)
    y_test,测试样本标签，array，(n_samples, )
    :return:
    """
    x_train = pos_text_train + neg_text_train
    x_test = pos_text_test + neg_text_test
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(x_train)
    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=output_dim)

    tokenizer1 = keras.preprocessing.text.Tokenizer(num_words=max_words)
    tokenizer1.fit_on_texts(x_test)
    x_test = tokenizer1.texts_to_sequences(x_test)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=output_dim)

    y_train = [0]*len(pos_text_train) + [1]*len(neg_text_train)
    y_train = np.asarray(y_train)
    y_test = [0]*len(pos_text_test) + [1]*len(neg_text_test)
    y_test = np.asarray(y_test)

    #打乱
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    return x_train, y_train, x_test, y_test

def cleanText(corpus):
    """
    :param corpus:待清洗文本, list
    :return: corpus:清洗后的文本，list
    """
    #去掉 符号，\n,<br >,低频词汇
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n', '') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]
    for c in punctuation:
        corpus = [z.replace(c, '%s' % c) for z in corpus]
    corpus = [z.split() for z in corpus]

    return corpus

def getVecsByWord2Vec(word2vec_model, corpus, output_dim):
    """
    :param model:训练好的word2vec模型
    :param corpus:输入文本，list
    :param output_dim:输出维度，int
    :return: x:向量化的文本，array，(n_samples, output_dim)
    """
    x = []
    for text in corpus:
        vectors_of_one_text = np.zeros(output_dim)
        n_words = 0
        for word in text:
            try:
                vectorized_word = word2vec_model[word]
                vectors_of_one_text += vectorized_word
                n_words += 1
            except KeyError:
                continue
        x.append(vectors_of_one_text/n_words)
    x = np.array(x, dtype='float')
    return x

def get_features_by_word2vec(pos_text_train, neg_text_train, pos_text_test, neg_text_test, output_dim):
    """
    :param output_dim:输出维度=100
    :return:
        x_train: 向量化后的所有训练样本内容，array，(n_samples, output_dim)
        x_test: 向量化后的所有测试样本内容，array，(n_samples, output_dim)
        y_train: 训练样本标签，array, (n_smaples)
        y_test: 测试样本标签，array, (n_smaples)
    """
    pos_text_train = cleanText(pos_text_train)
    neg_text_train = cleanText(neg_text_train)
    pos_text_test = cleanText(pos_text_test)
    neg_text_test = cleanText(neg_text_test)

    x_train = pos_text_train+neg_text_train
    print("The number of samples for training is {}".format(len(x_train)))

    x_test = pos_text_test+neg_text_test
    print("The number of samples for testing is {}".format(len(x_test)))

    cores = multiprocessing.cpu_count()

    if os.path.exists(word2vec_bin):
        print("Find cache file %s"%word2vec_bin)
        word2vec_model = Word2Vec.load(word2vec_bin)
    else:
        word2vec_model = Word2Vec(size=output_dim, window=5, min_count=10, iter=10, workers=cores)
        word2vec_model.build_vocab(x_train)
        word2vec_model.train(x_train, total_examples=word2vec_model.corpus_count, epochs=word2vec_model.iter)
        word2vec_model.save("word2vec_bin")

    x_test = getVecsByWord2Vec(word2vec_model, x_test, output_dim)
    x_train = getVecsByWord2Vec(word2vec_model, x_train, output_dim)
    print("Transform text into word2vec successfully")

    y_train = [0]*len(pos_text_train) + [1]*len(neg_text_train)
    y_train = np.asarray(y_train)
    y_test = [0]*len(pos_text_test) + [1]*len(neg_text_test)
    y_test = np.asarray(y_test)

    #打乱
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

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

def get_features_by_doc2vec(pos_text_train, neg_text_train, pos_text_test, neg_text_test, output_dim):
    """
    通过doc2vec模型，得到向量化后的文本矩阵
    :param output_dim: int,输出维度
    :return:
        x_train: 向量化后的所有训练样本内容，array，(n_samples, output_dim)
        x_test: 向量化后的所有测试样本内容，array，(n_samples, output_dim)
        y_train: 训练样本标签，array, (n_smaples, )
        y_test: 测试样本标签，array, (n_smaples, )
    """
    pos_text_train = cleanText(pos_text_train)
    neg_text_train = cleanText(neg_text_train)
    pos_text_test = cleanText(pos_text_test)
    neg_text_test = cleanText(neg_text_test)

    x_train = pos_text_train+neg_text_train
    print("The number of samples for training is {}".format(len(x_train)))
    x_test = pos_text_test+neg_text_test
    print("The number of samples for testing is {}".format(len(x_test)))

    x_train_labelized = labelizeReviews(x_train, 'Train')

    cores = multiprocessing.cpu_count()

    if os.path.exists(doc2vec_bin):
        print("Find cache file %s" )% doc2vec_bin
        doc2vec_model = Doc2Vec.load(doc2vec_bin)
    else:
        doc2vec_model = Doc2Vec(size=output_dim, negative=5, hs=0, min_count=2, workers=cores, iter=60)
        doc2vec_model.build_vocab(x_train_labelized)
        doc2vec_model.train(x_train_labelized, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.iter)
        doc2vec_model.save("doc2vec_bin")

    x_test = get_doc2Vecs(doc2vec_model, x_test)
    x_train = get_doc2Vecs(doc2vec_model, x_train)
    print("Transform text into doc2vec successfully")

    y_train = [0]*len(pos_text_train) + [1]*len(neg_text_train)
    y_train = np.asarray(y_train)
    y_test = [0]*len(pos_text_test) + [1]*len(neg_text_test)
    y_test = np.asarray(y_test)

    #打乱
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    return x_train, y_train, x_test, y_test

#模型
def gnb():
    """
    :return:返回朴素贝叶斯模型
    """
    gnb = GaussianNB()
    return gnb

def svm():
    """
    :return:返回svm模型
    """
    s = SVC()
    return s

def cnn():
    """
    :return: 返回cnn模型
    """
    input_dim = max_words
    input_length = output_dim
    model = keras.models.Sequential(
        [keras.layers.Embedding(input_dim=input_dim+1, output_dim=128, input_length=input_length),
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
    :return:返回lstm 模型
    """
    input_dim = max_words
    input_length = output_dim
    model = keras.models.Sequential(
        [keras.layers.Embedding(input_dim=input_dim+1, output_dim=128, input_length=input_length),
         keras.layers.LSTM(units=128, dropout=0.2, return_sequences=True),
         keras.layers.LSTM(units=128, dropout=0.2),
         keras.layers.Dense(units=128, activation="relu"),
         keras.layers.Dropout(0.2),
         keras.layers.Dense(units=1, activation="sigmoid")])
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    return model
def mlp():
    """
    :return:返回mlp模型
    """
    model = MLPClassifier(hidden_layer_sizes=(5, 2), activation='relu', solver='adam',
                          batch_size=batch_size, shuffle=True, verbose=True)
    return model
def dnn():
    """
    :return:返回dnn模型
    """
    input_dim = output_dim
    model = keras.models.Sequential([
        keras.layers.Dense(units=512, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(units=1, activation='sigmoid')
    ])

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])
    return model

#载入数据并向量化
def get_features(load_type):
    """
    :param load_type: 选择向量化的方式。0:wordbag;1:vocabulary;2:word2vec;3:doc2vec
    :return:
        x_train: 向量化后的所有训练样本内容，array，(n_samples, output_dim)
        x_test: 向量化后的所有测试样本内容，array，(n_samples, output_dim)
        y_train: 训练样本标签，array, (n_smaples, )
        y_test: 测试样本标签，array, (n_smaples, )
    """
    pos_text_train, neg_text_train, pos_text_test, neg_text_test = load_file(file_path)
    if load_type == 0:
        x_train, y_train, x_test, y_test = get_features_by_wordbag(pos_text_train, neg_text_train, pos_text_test,
                                                                    neg_text_test)
        return x_train, y_train, x_test, y_test
    if load_type == 1:
        x_train, y_train, x_test, y_test = get_features_by_vocabulary(pos_text_train, neg_text_train, pos_text_test,
                                                                    neg_text_test)
        return x_train, y_train, x_test, y_test
    if load_type == 2:
        #x_train, y_train, x_test, y_test = get_features_by_vocabulary(x_train, y_train, x_test, y_test)
        x_train, y_train, x_test, y_test = get_features_by_word2vec(pos_text_train, neg_text_train, pos_text_test,
                                                                    neg_text_test, output_dim)
        return x_train, y_train, x_test, y_test
    elif load_type == 3:
        x_train, y_train, x_test, y_test = get_features_by_doc2vec(pos_text_train, neg_text_train, pos_text_test,
                                                                   neg_text_test, output_dim)
        return x_train, y_train, x_test, y_test

#模型训练与测试
def train(x_train, y_train, method_type, model_type, batch_size, epochs):
    """
    :param x_train:向量化后的所有训练样本内容，array，(n_samples, output_dim)
    :param y_train:训练样本标签，array, (n_smaples, )
    :param method_type:方法类型，int，0：机器学习；1：深度学习
    :param model_type:模型类型，int，00:gnb  01:svm  02:mlp  10:cnn 11:lstm 12:dnn
    :param batch_size:一次训练所选取的样本，int
    :param epochs:所有样本重复次数，int
    :return:model:训练好的模型
    """
    #ml or dl
    if method_type == 0:
        if model_type == 0:
            model = gnb()
            model.fit(x_train, y_train)

        if model_type == 1:
            model = svm()
            model.fit(x_train, y_train)

        elif model_type == 2:
            model = mlp()
            model.fit(x_train, y_train)
    elif method_type == 1:
        if model_type == 0:
            model = cnn()
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
        if model_type == 1:
            model = lstm()
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
        elif model_type == 2:
            model = dnn()
            
    return model

def test(x_test, y_test, trained_model, input_type):
    """
    :param x_test: 向量化后的所有测试样本内容，array，(n_samples, output_dim)
    :param y_test: 测试样本标签，array, (n_smaples, )
    :param trained_model: 训练好的模型
    :param input_type: 方法类型，int，0：机器学习；1：深度学习
    :return:
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

if __name__ == "__main__":
    file_path = '/home/hezhouyu/projects/dataset/imdb/aclimdb/'
    word2vec_bin = "word2ver.bin"
    doc2vec_bin = "doc2vec.bin"
    SentimentDocument = namedtuple('SentimentDocument', 'words tags')

    max_words = 500
    output_dim = 100
    batch_size = 500
    epochs = 4

    method_type = 0   #0:'ml'  1:'dl'
    model_type = 0  #00:gnb  01:svm  02:mlp  10:cnn 11:lstm 12:dnn
    load_type = 0 #0:wordbag 1:recabulary 2:word2Vec 3:doc2Vec
    x_train, y_train, x_test, y_test = get_features(load_type)
    print('x_train.shape = ', x_train.shape, 'y_train.shape',  y_train.shape)
    trained_model = train(x_train, y_train, method_type, model_type, batch_size, epochs)

    print(test(x_test, y_test, trained_model, method_type))

