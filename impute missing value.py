# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 09:48:21 2017

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
data=pd.read_csv('newnewdata.csv')
y=data['isSTEM'].to_frame()
X=data.drop(['isSTEM'], 1)
dropped_X=data.drop(['isSTEM', ], 1)
names=list(X)

#impute missing mcas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
X=input_data
missed_mcas=X.loc[X['MCAS'] == -999]
complete_mcas=X.loc[X['MCAS'] != -999]
mcas=complete_mcas['MCAS']
train_mcas=complete_mcas.drop('MCAS', 1)
selected_train_mcas=complete_mcas.drop(['MCAS', 'responseIsChosen', 'timeSinceSkill', 'totalTimeByPercentCorrectForskill', 'totalFrTimeOnSkill', 'sumTimePerSkill', 'NumActions','frTimeTakenOnScaffolding', 'timeTaken', 'totalFrSkillOpportunities', 'frPast5HelpRequest', 'totalFrAttempted'], 1)
x_train, x_cv, y_train, y_cv = train_test_split(selected_train_mcas,mcas, test_size =0.3)
ridgeReg = Ridge(alpha=0.001, normalize=True)
ridgeReg.fit(x_train,y_train)
pred = ridgeReg.predict(x_cv)
score=ridgeReg.score(x_cv,y_cv)
def pretty_print_linear(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst, key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name) for coef, name in lst)
impute_input = X.drop(X[X.MCAS != -999].index)
impute_input=impute_input.drop(['MCAS','responseIsChosen', 'timeSinceSkill', 'totalTimeByPercentCorrectForskill', 'totalFrTimeOnSkill', 'sumTimePerSkill', 'NumActions','frTimeTakenOnScaffolding', 'timeTaken', 'totalFrSkillOpportunities', 'frPast5HelpRequest', 'totalFrAttempted'], 1)
imputed_mcas=ridgeReg.predict(impute_input)
X.loc[X['MCAS'] == -999, 'MCAS'] = imputed_mcas
newnewdata=pd.concat([X,labels], axis=1)
newnewdata.to_csv('newnewdata.csv', encoding='utf-8', index=False)