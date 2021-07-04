! pip install pywavelets
! pip install tensorflow

! wget 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip'
! unzip 'UCI HAR Dataset.zip'

import numpy as np
import matplotlib.pyplot as plt 
import pywt
import pandas as pd 
from sklearn.model_selection import train_test_split
import os
from tensorflow import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import History
from keras.layers import BatchNormalization
from keras.layers import Dropout

activities = ['WALKING', 'WALKING_UPSTAIRS', \
              'WALKING_DOWNSTAIRS', 'SITTING', \
              'STANDING', 'LAYING']
train_signals, test_signals = [], []
for input_file in os.listdir('UCI HAR Dataset/train/Inertial Signals/'):
        signal = pd.read_csv('UCI HAR Dataset/train/Inertial Signals/'+input_file, delim_whitespace=True, header=None)
        train_signals.append(signal)
train_signals = np.transpose(np.array(train_signals), (1, 2, 0))
y_train = pd.read_csv('UCI HAR Dataset/train/y_train.txt', header=None)
for input_file in os.listdir('UCI HAR Dataset/test/Inertial Signals/'):
        signal = pd.read_csv('UCI HAR Dataset/test/Inertial Signals/'+input_file, delim_whitespace=True, header=None)
        test_signals.append(signal)
test_signals = np.transpose(np.array(test_signals), (1, 2, 0))
y_test = pd.read_csv('UCI HAR Dataset/test/y_test.txt', header=None)

train_signals, test_signals, y_train, y_test = train_test_split(np.concatenate((train_signals, test_signals)),\
        np.concatenate((y_train, y_test)), test_size=2947, random_state=42)

t0 = 0
HZ = 50
dt= 1/HZ
N = np.shape(train_signals)[1]
time = np.arange(0, N) * dt + t0
scales = np.arange(1, N)
plt.figure()
data_number = 0
for i in range(9):
    signal = train_signals[data_number,:,i]
    [coefficients, frequencies] = pywt.cwt(signal, scales, 'morl', dt)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    contourlevels = np.log2(levels)
    fig, ax = plt.subplots()
    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both')
    ax.set_title('Wavelet Transform (Power Spectrum) of signal', fontsize=20)
    ax.set_ylabel('Period', fontsize=18)
    ax.set_xlabel('Time', fontsize=18)
    yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], -1)
    cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    plt.show()

scales = range(1,N)
waveletname = 'morl'
train_size = np.shape(train_signals)[0] # 5000 on colab
test_size= np.shape(test_signals)[0] # 500 on colab
n_comp = np.shape(train_signals)[2]
train_data_cwt = np.ndarray(shape=(train_size, N-1, N-1, n_comp))
for ii in range(0,train_size):
    if ii % 1000 == 0:
        print(ii)
    for jj in range(0,n_comp):
        signal = train_signals[ii, :, jj]
        coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        coeff_ = coeff[:,:N-1]
        train_data_cwt[ii, :, :, jj] = coeff_
test_data_cwt = np.ndarray(shape=(test_size, N-1, N-1, n_comp))
for ii in range(0,test_size):
    if ii % 100 == 0:
        print(ii)
    for jj in range(0,n_comp):
        signal = test_signals[ii, :, jj]
        coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        coeff_ = coeff[:,:N-1]
        test_data_cwt[ii, :, :, jj] = coeff_

x_train = train_data_cwt
x_test = test_data_cwt
y_train = y_train[:train_size]
y_test = y_test[:test_size]

history = History()
img_x = N-1
img_y = N-1
img_z = n_comp
input_shape = (img_x, img_y, img_z)
batch_size = 16
num_classes = np.unique(y_train)[-1]+1
epochs = 10
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential([
    Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(64, (5, 5), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (5, 5), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    BatchNormalization(),
    Dense(1000, activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    Dense(num_classes, activation='softmax')
])
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose = 1,
          validation_split = 0.2,
          callbacks=[history],
          use_multiprocessing=True)

train_score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))
test_score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))
