! pip install pywavelets
! pip install tensorflow

! wget 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip'
! unzip 'UCI HAR Dataset.zip'

import numpy as np
import matplotlib.pyplot as plt 
import pywt
import pandas as pd 
from sklearn.decomposition import PCA 
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

activities = ['WALKING', 'WALKING_UPSTAIRS', \
              'WALKING_DOWNSTAIRS', 'SITTING', \
              'STANDING', 'LAYING']

X_train = pd.read_csv('UCI HAR Dataset/train/X_train.txt', delim_whitespace=True, header=None)
y_train = pd.read_csv('UCI HAR Dataset/train/y_train.txt', header=None)
X_test = pd.read_csv('UCI HAR Dataset/test/X_test.txt', delim_whitespace=True, header=None)
y_test = pd.read_csv('UCI HAR Dataset/test/y_test.txt', header=None)

scales = 100
pc = 10
wavelet = 'morl'

features = np.empty((0,scales*pc))
pca = PCA(n_components = pc)
for i in range(np.shape(X_train)[0]):
  coef, freqs = pywt.cwt(X_train[i,:], np.arange(1,scales+1), wavelet)
  features = np.vstack([features, pca.fit_transform(coef).flatten()])

print('{:4.2} percent of variance is reserved after PCA'.\
    format(np.sum(100*pca.explained_variance_ratio_)))

model = Pipeline([('scaler', StandardScaler()),\
                  ('poly', PolynomialFeatures(2)),\
                  ('ml', MLPClassifier(hidden_layer_sizes=8, solver='sgd', random_state=42))])
model.fit(features, np.ravel(y_train))
print('traning accuracy: {:4.2}'.format(model.score(features, np.ravel(y_train))))

features_ = np.empty((0,scales*pc))
for i in range(np.shape(X_test)[0]):
  coef_, freqs_ = pywt.cwt(X_test[i,:], np.arange(1,scales+1), wavelet)
  features_ = np.vstack([features_, pca.transform(coef_).flatten()])

print('test accuracy: {:4.2}'.format(model.score(features_, np.ravel(y_test))))