# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 10:01:28 2017

@author: dongg
"""

#MLP
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.core import Dropout, Activation
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
from keras.constraints import max_norm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

data=pd.read_csv('newnewdata.csv')
y=data['isSTEM'].to_frame()
y=pd.DataFrame.as_matrix(y)
y=y.reshape((492, ))
y_bool=y.astype('bool')
import itertools
from sklearn import svm
X=pd.read_csv('norm.csv')
X=pd.DataFrame.as_matrix(X)
stand_X=pd.read_csv('standarized_X.csv')
#keras mlp
def create_model(init_mode='normal', activation='relu',dropout_rate=0.1, neurons=70):
    model = Sequential()
    model.add(Dense(35, input_dim=35,init=init_mode, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, init=init_mode, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    optimizer =SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
seed = 7
np.random.seed(seed)

model = KerasClassifier(build_fn=create_model, verbose=0)

batch_size = [10]
epochs = [5]

init_mode = ['normal']
activation = ['relu']
dropout_rate = [0.01]
neurons = [70]
optimizer =SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  
model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=10, verbose=0)
param_grid = dict(neurons=neurons, dropout_rate=dropout_rate, init_mode=init_mode, activation=activation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-11, cv=10)
grid_result = grid.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%0.3f (+/-%0.03f) for %r" % (mean, stdev, param))

#scikit learn mlp
mlp=MLPClassifier()
param_grid_linear = {
        "activation": ['relu', 'logistic', 'tanh'],
        "hidden_layer_sizes": [(35, 80, 1)],
        "solver": ['sgd', 'adam'], 
        "learning_rate": ['invscaling', 'adaptive', 'constant'],
        "max_iter": [800, 1000],
        "momentum": [0.3, 0.5, 0.8],
        }

X_train, X_test, y_train, y_test = train_test_split(stand_X, y, test_size=0.30, random_state=7)
CV_mlp = GridSearchCV(estimator=mlp, param_grid=param_grid_linear, cv= 10)

CV_mlp.fit(X_train, y_train)
print("Best parameters set found on development set:")
print()
print(CV_mlp.best_params_)
print()
print("Grid scores on training set:")
print()
means = CV_mlp.cv_results_['mean_test_score']
stds = CV_mlp.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, CV_mlp.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))



X_train, X_test, y_train, y_test = train_test_split(stand_X[0:350], y[0:350], test_size=0.30, random_state=7)
mlp = MLPClassifier(activation='relu', solver = 'sgd', learning_rate = 'adaptive', max_iter = 1000, hidden_layer_sizes=(35,70,1))
mlp.fit(stand_X[0:392], y[0:392])
predictions = mlp.predict(stand_X[392:])
print(confusion_matrix(y[392:],predictions))
print (classification_report(y[392:],predictions))
print (accuracy_score(y[392:], predictions))

def plot_coefficients(classifier, feature_names, top_features=20):
 coef = classifier.coef_.ravel()
 top_positive_coefficients = np.argsort(coef)[-top_features:]
 top_negative_coefficients = np.argsort(coef)[:top_features]
 top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
 # create plot
 plt.figure(figsize=(15, 5))
 colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
 plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
 feature_names = np.array(feature_names)
 plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
 plt.show()


print ("plotting")
mlp.fit(X, y)
plot_coefficients(mlp, list(X))



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