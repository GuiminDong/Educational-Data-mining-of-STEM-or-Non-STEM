# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 09:55:14 2017

@author: dongg
"""
from __future__ import print_function
print(__doc__)
import pandas as pd
import numpy as np
import csv
from sklearn.decomposition import PCA
from sklearn import decomposition
from string import ascii_letters
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation
from sklearn.model_selection import GridSearchCV
from time import time
from operator import itemgetter
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing
from sklearn.feature_selection import RFECV
import sklearn
import sklearn.linear_model as lm
import sklearn.grid_search as gs
from sklearn.metrics import classification_report,confusion_matrix

#SVM

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.model_selection import GridSearchCV
data=pd.read_csv('newnewdata.csv')
y=data['isSTEM'].to_frame()
X=data.drop(['isSTEM'], 1)
#Random Forest Classifier
#train_ver2, test_ver2, labels=input_data[0:343], input_data[343:], complete_label[0:343].isSTEM.tolist()
train_ver2, test_ver2, labels=X[0:343], X[343:], y[0:343]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_ver2, y, test_size=0.3, random_state=42)
train_ver2, test_ver2, labels=X[0:343], X[343:], y[0:343]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_ver2, labels, test_size=0.3, random_state=42)
parameters = [{'kernel': ['linear'],
               'gamma': [0.01, 0.1, 1, 10],
                'C': [10, 100, 1000]}
              ]
print("# Tuning hyper-parameters")
print()

clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)
clf.fit(X_train, y_train['isSTEM'].values)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on training set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#Compute confusion matrix
cnf_matrix = confusion_matrix(y[392:], predictions)
np.set_printoptions(precision=2)

#Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Non-STEM', 'STEM'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Non-STEM', 'STEM'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()
print ("Accuracy Score:")
print (accuracy_score(y[392:], predictions))
