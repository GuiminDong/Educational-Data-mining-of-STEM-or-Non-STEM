# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 09:50:00 2017

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
#Random Forest Classifier
#train_ver2, test_ver2, labels=input_data[0:343], input_data[343:], complete_label[0:343].isSTEM.tolist()
train_ver2, test_ver2, labels=X[0:343], X[343:], y[0:343]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_ver2, y, test_size=0.3, random_state=42)
grid_times = {}
clf = RandomForestClassifier(random_state = 84)

for number in np.arange(2, 600, 50):
    
    param = np.arange(1,number,10)
    param_grid = {"n_estimators": param,
                  "criterion": ["gini", "entropy"]}
    
    grid_search = GridSearchCV(clf, param_grid = param_grid)
    
    t0 = time()
    grid_search.fit(X_train[features], y_train)
    compute_time = time() - t0
    grid_times[len(grid_search.grid_scores_)] = time() - t0
    
grid_times = pd.DataFrame.from_dict(grid_times, orient = 'index')

# calculate the time to run a GridSearchCV for multiple numbers of parameter permutations.
grid_times = {0: { 2: 0.034411907196044922,
                  12: 1.5366179943084717,
                  22: 5.0431020259857178,
                  32: 11.378448963165283,
                  42: 20.211128950119019,
                  52: 30.040457010269165,
                  62: 39.442277908325195,
                  72: 56.834053993225098,
                  82: 67.847633838653564,
                  92: 91.005517959594727,
                  102: 111.2420859336853,
                  112: 135.75759792327881}}
final = pd.DataFrame.from_dict(grid_times)
final = final.sort_index()
plt.plot(final.index.values, final[0])
plt.xlabel('Number of Parameter Permutations')
plt.ylabel('Time (sec)')
plt.title('Time vs. Number of Parameter Permutations of GridSearchCV')

# function takes a RF parameter and a ranger and produces a plot and dataframe of CV scores for parameter values
def evaluate_param(parameter, num_range, index):
    grid_search = GridSearchCV(clf, param_grid = {parameter: num_range})
    grid_search.fit(X_train, y_train['isSTEM'].values)
    
    df = {}
    for i, score in enumerate(grid_search.grid_scores_):
        df[score[0][parameter]] = score[1]
       
    
    df = pd.DataFrame.from_dict(df, orient='index')
    df.reset_index(level=0, inplace=True)
    df = df.sort_values(by='index')
 
    plt.subplot(3,2,index)
    plot = plt.plot(df['index'], df[0])
    plt.title(parameter)
    return plot, df

# parameters and ranges to plot
param_grid = {"n_estimators": np.arange(2, 300, 2),
              "max_depth": np.arange(1, 28, 1),
              "min_samples_split": np.arange(2,150,1),
              "min_samples_leaf": np.arange(1,60,1),
              "max_leaf_nodes": np.arange(2,60,1),
              "min_weight_fraction_leaf": np.arange(0.1,0.4, 0.1)}
index = 1
plt.figure(figsize=(16,12))
for parameter, param_range in dict.items(param_grid):   
    evaluate_param(parameter, param_range, index)
    index += 1


# Utility function to report best scores
def report(grid_scores, n_top):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.4f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

# parameters for GridSearchCV
param_grid2 = {"n_estimators": [55, 60, 65, 90, 95, 120, 125],
              "max_depth": [5, 10],
              "min_samples_split": [7, 8, 9, 10, 11, 12],
              "min_samples_leaf": [10,11,12,13,14,15],
              "max_leaf_nodes": [25, 30],
              "min_weight_fraction_leaf": [0.2]}

grid_search = GridSearchCV(clf, param_grid=param_grid2)
grid_search.fit(X_train, y_train['isSTEM'].values)

report(grid_search.grid_scores_, 4)



#clf model
clf = RandomForestClassifier(min_samples_split = 60, 
                             max_leaf_nodes = 10, 
                             n_estimators = 55, 
                             max_depth = 5,
                             min_samples_leaf = 10,
                             min_weight_fraction_leaf= 0.2)
clf.fit(train_ver2, labels['isSTEM'].values)


# importance feature
features = list(input_data)
importances = clf.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(15,14))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features) ## removed [indices]
plt.xlabel('Relative Importance')
plt.show()

#rf show
def visualize_tree(tree, feature_names):

    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    
i_tree = 0
for tree_in_forest in clf.estimators_:
    visualize_tree(tree_in_forest, list(input_data))
    i_tree = i_tree + 1