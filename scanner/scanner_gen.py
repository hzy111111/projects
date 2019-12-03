# -*- coding:utf-8 -*-
import os
import pickle
from six.moves import urllib
import tflearn
from tflearn.data_utils import *
import keras

def get_data(maxlen, char_idx, xss_data_file):
    """
    :param maxlen:单个句子的长度
    :param char_idx:char字典， dic
    :param xss_data_file:输入文件，txt
    :return:
    char_idx:建立字符对应数字的转换表，生成字典
    X:每个句子提取maxlen长度的文本进行字符转换，array, (n_samples, maxlen, len(char_idx))
    Y:目标词，array, (n_samples, len(char_idx))
    """
    X, Y, char_idx = \
        textfile_to_semi_redundant_sequences(xss_data_file, seq_maxlen=maxlen, redun_step=3,
                                             pre_defined_char_idx=char_idx)
    print(X.shape, Y.shape, len(char_idx), type(X))
    print("Get data successfully")

    return X, Y, char_idx

def lstm():
    """
    :return: lstm 模型
    """
    model = keras.Sequential([
        keras.layers.LSTM(units=128, input_shape=(maxlen, len(char_idx)), dropout=0.2, return_sequences=True),
        keras.layers.LSTM(units=128, dropout=0.2),
        keras.layers.Dense(units=128, activation="relu"),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(units=len(char_idx), activation="softmax")
    ])
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    return model

def trian(X, Y):
    """
    :param X:每个句子提取maxlen长度的文本进行字符转换，array, (n_samples, maxlen, len(char_idx))
    :param Y:目标词，array, (n_samples, len(char_idx))
    :return: 训练好的模型
    """
    model = lstm()
    model.fit(X, Y, epochs=1, batch_size=128, verbose=1)
    return model

def choose_char(preds, temperature, char_idx):
    """
    :param preds: array (1, len(char_idx))模型输出的预测向量
    :param temperature: float, 控制采样过程中的随机性
    :param char_idx: dict,建立字符对应数字的转换表，生成字典
    :return:
    next_char：生成目标词
    """
    preds = np.array(preds).astype('float64')
    preds = np.log(preds) / temperature
    preds_exp = np.exp(preds)
    preds = preds_exp / np.sum(preds_exp)

    probas = np.random.multinomial(n=10, pvals=preds, size=1)#随机采样，实验10次，返回[1，len(preds)],表示落在个点次数
    value = np.argmax(probas)#返回值最大的点的索引
    next_char = [k for k, v in char_idx.items() if v == value][0]#根据value找到对应的key

    return next_char

if __name__=="__main__":
    maxlen = 25
    char_idx = None
    temperatue = 1.0
    xss_data_file = "/home/hezhouyu/projects/dataset/scan_data/xss.txt"

    X, Y, char_idx = get_data(maxlen, char_idx, xss_data_file)
    trained_model = trian(X, Y)
    seed_text = '<IMG SRC=x onbeforeunload="alert(String.fromCharCode(88,83,83))">'#作为初始随机句子
    seed_x = seed_text[4:4+25]#从初始句子中随机抽取句子向量长度为maxlen
    sequence = np.zeros((1, maxlen, len(char_idx)))#初始化一个句子向量

    for char_index, char in enumerate(seed_x):
        sequence[0, char_index, char_idx[char]] = 1
    print("new sequence char2num successfully")

    preds = trained_model.predict(sequence, verbose=1)[0]
    seed_y = choose_char(preds, temperatue, char_idx)
    sequence = seed_x + seed_y
    sequence = sequence[1:]
    print(seed_x, sequence, seed_y)

