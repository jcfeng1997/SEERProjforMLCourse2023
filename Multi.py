#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 19:11:17 2021

@author: nadou
"""
from sklearn import preprocessing
from keras import models
from keras import layers
from keras.utils import to_categorical
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("new_seer_data.csv")
data = np.array(data)
y = data[:,7]
x = data[:,[1,2,3,4,5,6,8,9,10,11,12,13,14]]
x = preprocessing.scale(x)

x_train = x[np.arange(0,80679)]
x_test = x[np.arange(80679,115255)]
y_train = y[np.arange(0,80679)]
y_test = y[np.arange(80679,115255)]
newy_train = to_categorical(y_train)
newy_test = to_categorical(y_test)

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(13,))) 
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))  
model.add(layers.Dense(8, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, newy_train, epochs=20, batch_size=512, validation_data=(x_test, newy_test))
results = model.evaluate(x_test, newy_test)



plt.plot(history.epoch, history.history.get('loss'))


plt.plot(history.epoch, history.history.get('accuracy'))

model.summary()

history_dict = history.history 
loss_values = history_dict['loss'] 
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss') 
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss') 
plt.legend()
plt.show()

plt.clf()
acc = history_dict['accuracy'] 
val_acc = history_dict['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc') 
plt.plot(epochs, val_acc, 'b', label='Validation acc') 
plt.title('Training and validation accuracy') 
plt.xlabel('Epochs')
plt.ylabel('Accuracy') 
plt.legend()
plt.show()
