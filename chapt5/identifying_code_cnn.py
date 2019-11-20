import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
#数据输入
K.set_image_dim_ordering('th') # 设置图像的维度顺序（‘tf’或‘th’）
# 当前的维度顺序如果为'th'，则输入图片数据时的顺序为：channels,rows,cols，否则:rows,cols,channel
mnist = input_data.read_data_sets('/home/hezhouyu/projects/MNIST_data', one_hot = True)
(x_train, y_train) = (mnist.train.images, mnist.train.labels)
(x_test, y_test) = (mnist.test.images, mnist.test.labels)
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')
#print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
#训练集（55000，1，28，28）, 测试集（10000，1，28，28）

#设置常量
num_classes = y_test.shape[1]#类别数为10
epochs=10

#网络结构
def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#训练模型
def train_model():
    model = cnn_model()
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=2)
    #verbose : 进度表示方式。0表示不显示数据，1表示显示进度条，2表示用只显示一个数据。
    #model.summary()

#预测
def prediction():
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))

if __name__ == "__main__":
    train_model()
    prediction()
#loss: 0.0156 - acc: 0.9950 - val_loss: 0.0364 - val_acc: 0.9882 Baseline Error: 1.18%

