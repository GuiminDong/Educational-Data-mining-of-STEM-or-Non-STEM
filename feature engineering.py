# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 21:57:51 2017

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
from sklearn.grid_search import GridSearchCV
from time import time
from operator import itemgetter
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import preprocessing

data=pd.read_csv('newnewdata.csv')
y=data['isSTEM'].to_frame()
X=data.drop(['isSTEM'], 1)
names=list(X)
#feature selection
ranks={}
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFE, f_regression
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
np.random.seed(36)
def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))
#random forest
rf = RandomForestClassifier(min_samples_split = 60, 
                             max_leaf_nodes = 10, 
                             n_estimators = 55, 
                             max_depth = 5,
                             min_samples_leaf = 10,
                             min_weight_fraction_leaf= 0.2)
rf.fit(X, y['isSTEM'].values)
ranks["RF"] = rank_to_dict(rf.feature_importances_, names)

#logistic regression
logReg = LogisticRegression()
logReg.fit(X, y['isSTEM'].values)
ranks["logReg"] = rank_to_dict(np.abs(logReg.coef_[0]), names)

#LDA

lda = LinearDiscriminantAnalysis()
lda.fit(X, y['isSTEM'].values)
ranks["lda"] = rank_to_dict(np.abs(lda.coef_[0]), names)


#support vector machine
svc = svm.SVC(kernel='linear', gamma=1, C=10).fit(X, y['isSTEM'].values)
ranks["svs"] = rank_to_dict(np.abs(svc.coef_[0]), names)


#RFECV
selector = RFECV(estimator=logReg, cv=5,
 scoring='mean_squared_error')
selector.fit(X, y['isSTEM'].values)
print ("Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), selector.ranking_), list(X))))
