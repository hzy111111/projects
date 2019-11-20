#coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

#数据输入
train_path = '/home/hezhouyu/projects/dataset/industry_timeseries/timeseries_train_data/'#修改
test_path = '/home/hezhouyu/projects/dataset/industry_timeseries/timeseries_predict_data/'
data = []
name = []
dfll = pd.read_csv(train_path+'1.csv', header=None,names=['years','month','day','maxC','minC','avgC','avgH','target'])
testll = pd.read_csv(test_path+'1.csv', header=None,names=['years','month','day','maxC','minC','avgC','avgH'])
data = np.array(dfll.ix[:, 3:8])
test = np.array(testll.ix[:, 3:8])
print(np.array(data).shape, np.array(test).shape)
"""
原始数据：‘2016,9,1,31.900000,20.400000,26.237500,65.500000’
np.array(data).shape=(578,5)
np.array(test).shape=(91,4)
"""



#设置常量
rnn_unit = 10 #隐层数量
input_size = 4
output_size = 1
lr = 0.001
epochs = 100
weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}
def get_train_data(batch_size=60, time_step=20, train_begin=0, train_end=len(data)):
    """
    获取训练集
    :param batch_size:int,60
    :param time_step:int,20
    :param train_begin:int,0
    :param train_end:int, 578
    :return batch_index:list train_x:list，(558, 20, 4) train_y:list，(558, 20, 1)
    578-20 因为time_step=20
    """
    batch_index = []
    data_train = data[train_begin:train_end]
    normalized_train_data = (
        data_train-np.mean(data_train, axis=0))/np.std(data_train, axis=0)#axis=0，输出矩阵为1行  <class 'numpy.ndarray'> (578, 5)
    train_x,train_y = [],[]
    for i in range(len(normalized_train_data)-time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i+time_step,:4]
        y = normalized_train_data[i:i+time_step, 4, np.newaxis]#np.newaxis把shape=(4,)转换成shape=(,4)
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append(len(normalized_train_data)-time_step)
    return batch_index, train_x, train_y

def get_test_data(time_step=20,data=data,test_begin=0):
    """
    获取测试集
    :param time_step:
    :param data:
    :param test_begin:
    :return:mean, std, test_x, test_y
    <class 'numpy.ndarray'> (5,) <class 'numpy.ndarray'> (5,) <class 'list'> (29,) <class 'list'> (578,)
    """
    data_test = data[test_begin:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test-mean)/std
    size = (len(normalized_test_data)+time_step-1)//time_step#(578+20-1)//20
    test_x, test_y = [], []
    for i in range(size-1):
        x = normalized_test_data[i*time_step:(i+1)*time_step, :4]
        y = normalized_test_data[i*time_step:(i+1)*time_step, 4]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:, :4]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:, 4]).tolist())#extended list
    return mean, std, test_x, test_y

#网络结构
def lstm(X):
    """
    :param X:
    :return:  pred,Tensor("sec_lstm/Shape_2:0", shape=(3,), dtype=int32)
     final_states, Tensor("sec_lstm/Shape_3:0", shape=(3,), dtype=int32)
    """
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X,[-1,input_size])
    input_rnn = tf.matmul(input,w_in)+b_in
    input_rnn = tf.reshape(input_rnn,[-1,time_step,rnn_unit])#输入
    cell = tf.contrib.rnn.BasicLSTMCell(rnn_unit)#调用
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    #print(tf.shape(output_rnn), tf.shape(final_states))
    output = tf.reshape(output_rnn,[-1,rnn_unit])
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out)+b_out
    return pred, final_states

#训练模型
def train_lstm(batch_size=60, time_step=20, epochs=epochs, train_begin=0, train_end=len(data)):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(X)
    loss = tf.reduce_mean(
        tf.square(tf.reshape(pred, [-1])-tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            for step in range(len(batch_index)-1):
                _, loss_ = sess.run([train_op, loss], feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],
                                                                Y:train_y[batch_index[step]:batch_index[step+1]]})
            if (i+1)%50==0:
                print("Numbers of epochs:", i+1, "loss", loss_)
                print("Save:", saver.save(sess, 'lstm_model/model.ckpt'))
        print("Finished")

#预测
def prediction(time_step=20):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    mean, std, test_x, test_y = get_test_data(time_step, test_begin=0)
    with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
        pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('lstm_model')
        saver.restore(sess, module_file)
        test_predict=[]
        for step in range(len(test_x)-1):
            prob=sess.run(pred,feed_dict={X:[test_x[step]]})
            predict=prob.reshape((-1))
            test_predict.extend(predict)
        test_y=np.array(test_y)*std[4]+mean[4]
        test_predict=np.array(test_predict)*std[4]+mean[4]
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)]))  #mean absolute error
        print("The MAE of this predict:",acc)
        #画图表示结果
        plt.figure(figsize=(24,8))
        plt.plot(list(range(len(test_predict))), test_predict, color='b',label = 'prediction')
        plt.plot(list(range(len(test_y))), test_y,  color='r',label = 'origin')
        plt.legend(fontsize=24)
        plt.show()
train_lstm()
prediction()


