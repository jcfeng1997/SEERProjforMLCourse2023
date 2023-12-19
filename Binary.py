#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 22:53:20 2021

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



data = pd.read_csv("binary.csv")
data = np.array(data)
y = data[:,13]
x = data[:,[1,2,3,4,5,6,7,8,9,10,11,12,14,15]]
x = preprocessing.scale(x)

x_train = x[np.arange(0,16304)]
x_test = x[np.arange(16304,23290)]
y_train = y[np.arange(0,16304)]
y_test = y[np.arange(16304,23290)]
newy_train = to_categorical(y_train)
newy_test = to_categorical(y_test)

model = models.Sequential()
model.add(layers.Dense(4, activation='relu', input_shape=(14,)))
model.add(layers.Dense(4, activation='relu'))  
model.add(layers.Dense(2, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, newy_train, epochs=20, batch_size=512, validation_data=(x_test, newy_test))
results = model.evaluate(x_test, newy_test)

model.summary()

plt.plot(history.epoch, history.history.get('loss'))


plt.plot(history.epoch, history.history.get('accuracy'))

results

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
