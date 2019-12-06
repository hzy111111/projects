#coding=utf-8
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_data():
    path = "/home/hezhouyu/projects/dataset/creditcard/creditcard.csv"
    data = pd.read_csv(path)
    print("load data successfully")

    return data

def normalized(data):
    data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    data = data.drop(['Time', 'Amount'], axis=1)#axis=0按行删除，1按列删除
    #y = data['Class']#得到标签
    #features = data.drop(['Class'], axis=1).columns
    #x = data[features]
    print("data normalized")
    return data

def undefsampling(data):
    len_fraud = len(data[data.Class == 1])#黑样本数量
    fraud_idx = np.array(data[data.Class == 1].index)#黑样本索引值

    normal_idx = np.array(data[data.Class == 0].index)
    random_choice_index = np.random.choice(normal_idx, size=len_fraud, replace=False)#repalce:有无放回

    x_index = np.concatenate([fraud_idx, random_choice_index])
    data = data.drop(['Class'], axis=1)
    x = data.iloc[x_index, :]
    y = [1]*len_fraud + [0]*len_fraud
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, shuffle=True)

    return x_train, x_test, y_train, y_test

def oversamping(data):
    y = data['Class'].values#得到标签
    data = data.drop(['Class'], axis=1)
    x = data.values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, shuffle=True)

    x_train, y_train = SMOTE().fit_sample(x_train, y_train)

    return x_train, x_test, y_train, y_test

def gnb():
    """
    undersample-acc = 0.8984
    :return: 朴素贝叶斯模型
    """
    model = GaussianNB()
    return model

def xgb():
    """
    undersample-acc:0.9238
    oversample-acc:
    :return: 返回xgboosting模型
    """
    model = XGBClassifier(n_estimate=150, max_depth=9)#n_estimate:决策树个数
    return model

def mlp():
    """
    undersample-acc = 0.934
    :return:mlp模型
    """
    model = MLPClassifier(hidden_layer_sizes=(5, 2), activation='relu', solver='adam',
                          shuffle=True, verbose=True)
    return model

if __name__ == "__main__":
    data = load_data()
    data = normalized(data)

    #x_train, x_test, y_train, y_test = undefsampling(data)
    x_train, x_test, y_train, y_test = oversamping(data)
    model = gnb()

    trained_model = model.fit(x_train, y_train)
    y_pred = trained_model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    clf = classification_report(y_test, y_pred)
    print(acc, '\n', clf)
