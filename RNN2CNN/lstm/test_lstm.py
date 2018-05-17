# encoding: utf-8
from utils import *
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import sys
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
import copy


def sin(x, T=100):
    return np.sin(2.0 * np.pi * x / T)

#sin波にノイズを乗せた状態の時系列データを合成する関数
def toy_problem(T=100, ampl=0.05):
    x = np.arange(0, 2 * T + 1) #関数のXの範囲を指定
    noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x))
    return sin(x) + noise

#
def make_dataset(low_data, n_prev=100):

    data, target = [], []
    maxlen = 3

    for i in range(len(low_data)-maxlen):
        data.append(  low_data[i:i + maxlen])
        target.append(low_data[i + maxlen])

    re_data   = np.array(data).reshape(len(data), maxlen, 1)
    re_target = np.array(target).reshape(len(data), 1)

    return re_data, re_target

f = toy_problem()
#print(f)
#g is re_data, h is re_target
#gを見て予測，hが正解
g, h = make_dataset(f)
print(g.shape)
print("#")
future_test = g[175].T

# 1つの学習データの時間の長さ -> 25
time_length = future_test.shape[1]
# 未来の予測データを保存していく変数
future_result = np.empty((0))

length_of_sequence = g.shape[1]
in_out_neurons = 1
n_hidden = 128

# モデル構築
model = Sequential()
model.add(LSTM(n_hidden, batch_input_shape=(None, length_of_sequence, in_out_neurons), return_sequences=False))
model.add(Dense(in_out_neurons))
model.add(Activation("linear"))
optimizer = Adam(lr=0.001)
model.compile(loss="mean_squared_error", optimizer=optimizer)
model_layer_dict, _, _ = init_coverage_tables(model,model,model)
#print(model_layer_dict)
print(model.summary())

layer_name, index = neuron_to_cover(model_layer_dict)
#print(layer_name, index)
# 学習
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=2)
model.fit(g, h,
          batch_size=300,
          epochs=1,
          validation_split=0.1,
          callbacks=[early_stopping]
          )

# 予測
predicted = model.predict(g)
#print(predicted[0])
a = copy.deepcopy(g)
g[1] = predicted[:3]
print("#")
for _ in range(g.shape[0]-190):
    _a = arange(3).reshape(1,3,1)
    _a[0]=g[_]
    #print(_a)
    update_coverage(_a, model, model_layer_dict,0.9)
    print(_,len(model_layer_dict), neuron_covered(model_layer_dict)[2])


# 予測(スクラッチ)
weights = model.get_weights()


obj=weights
print(len(obj))


w1=obj[0]
w2=obj[1]
w3=obj[2]
w4=obj[3]
w5=obj[4]

#pare_weights = np.concatenate((w1, w2, w3, w4, w5), axis=0)
pare_weights = np.concatenate((w1, w2), axis=0)
#pare_weights = np.concatenate((pare_weights, w3), axis=0)
print("#")
print(pare_weights.shape)
print(w3.shape)
print("===========")
for _ in range(len(obj)): print(obj[_].shape)
print("===========")

hl = n_hidden

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def activate(x):
    x[0:hl]      = sigmoid(x[0:hl])      #i
    x[hl:hl*2]   = sigmoid(x[hl:hl*2])   #a
    x[hl*2:hl*3] = sigmoid(x[hl*2:hl*3]) #f
    x[hl*3:hl*4] = sigmoid(x[hl*3:hl*4]) #o
    return x

def cactivate(c):
    return sigmoid(c)

x1 = np.array(g[0,0,:])
x2 = np.array(g[0,1,:])
x3 = np.array(g[0,2,:])
#print(x1, x2,x3)
h1 = np.zeros(hl)
c1 = np.zeros(hl)

o1 = x1.dot(w1)+h1.dot(w2)+w3
o1 = activate(o1)

c1 = o1[0:hl]*o1[hl:hl*2] + o1[hl*2:hl*3]*c1
#c1 = o1[0:128]*o1[128:256] + c1

h2 = o1[hl*3:hl*4]*cactivate(c1)

#2個目
o2 =  x2.dot(w1)+h2.dot(w2)+w3
o2 = activate(o2)

c2 = o2[0:hl]*o2[hl:hl*2] + o2[hl*2:hl*3]*c1
#c2 = o2[0:128]*o2[128:256] + c1

h3 = o2[hl*3:hl*4]*cactivate(c2)

#3個目
o3 = x3.dot(w1)+h3.dot(w2)+w3
o3 = activate(o3)

c3 = o3[0:hl]*o3[hl:hl*2] + o3[hl*2:hl*3]*c2
#c3 = o3[0:128]*o3[128:256] + c2

h4 = o3[hl*3:hl*4]*cactivate(c3)

y = h4.dot(w4)+w5

y = np.exp(y)/np.sum(np.exp(y))
print(y)
print("#")
plt.plot(y,'b-')
#plt.show()



# 未来予想
for step2 in range(100):

    test_data = np.reshape(future_test, (1, time_length, 1))
    #print(test_data)
    batch_predict = model.predict(test_data)

    future_test = np.delete(future_test, 0)
    future_test = np.append(future_test, batch_predict)

    future_result = np.append(future_result, batch_predict)


# sin波をプロット
plt.figure()
plt.plot(range(25,len(predicted)+25),predicted, color="r", label="predict")
plt.plot(range(0, len(f)), f, color="b", label="row")
plt.plot(range(0+len(f), len(future_result)+len(f)), future_result, color="g", label="future")
plt.legend()
#plt.show()
