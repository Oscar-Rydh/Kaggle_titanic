# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Preprocess data
# Load data
dataset = pd.DataFrame.from_csv(path='train.csv')

# Fixing NAN on embarked with the most common value
most_used_embarked = dataset['Embarked'].value_counts().index[0]
dataset.loc[dataset['Embarked'].isnull(), 'Embarked'] = most_used_embarked

# Loading values
X = dataset.iloc[:, [1, 3,4,5,6, 8,10]].values
y = dataset[['Survived']]
# Remove missing values
from sklearn.preprocessing import Imputer
# Replacing missing age with average
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# Onehot Encoding embarked with dummies
dumies = pd.get_dummies(X[:, -1])
X = X[:, 0:6]
X = np.concatenate((X, dumies), axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = None)

# Feature Scaling on X train
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Converting to numpy arrays due to some wierd error
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

import keras
# Initializes the ANN
from keras.models import Sequential
# Used to create the network
from keras.layers import Dense
# Dropout regularization to reduce overfitting when the variance is high, or if the trainset acc is much better then testset
from keras.layers import Dropout

# Initilizing a nural network for classification
classifier = Sequential()

# Adding input layer and first hidden layer
# Choosing the number of nodes in a hidden layer is an art and is perfected via experimentation
# For the none artistic choose N = (inputnodes + outputnodes) / 2
# Units = number of nodes in layer
classifier.add(Dense(units=10, activation='relu', kernel_initializer='uniform', input_dim=9))
# Dropout start with rate 0.1 then 0.2 etc upto 0.4
classifier.add(Dropout(rate=0.1))

# Adding a second hidden layer.
classifier.add(Dense(units=5, activation='relu', kernel_initializer='uniform'))
classifier.add(Dropout(rate=0.1))

# Adding a second hidden layer.
classifier.add(Dense(units=3, activation='relu', kernel_initializer='uniform'))
classifier.add(Dropout(rate=0.1))

# Adding the output layer. Units = 1 since its classifying, activation = sigmoid since we want the probability for the outcome
# Activation function must be softmax if we have more classes
classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))

# Compile the ANN (Apply stochastic gradiant descent or something else) 
# Refer udemy ANN step8 for info
# Optimizer: The algorithm to update weights (train)
# Loss: lossfunction of defining and calculating the cost
# Metrics: Criteria for evaluating the NN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'])

# Fitting (train) the ANN to our trainingset
classifier.fit(X_train, y_train, batch_size=32, epochs=500)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Confusion matrix need true or false values, not probabilities
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Evaluating the NN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# K-fold cross validation expects a function to build the NN
def build_classifier():
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    classifier = Sequential()
    
    classifier.add(Dense(units=10, activation='relu', kernel_initializer='uniform', input_dim=9))
    classifier.add(Dropout(rate=0.1))
    
    classifier.add(Dense(units=5, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dropout(rate=0.1))
    
    classifier.add(Dense(units=3, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dropout(rate=0.1))
    
    classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))
    
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'])
    return classifier

# Same constructor as for the Sequential fit function
k_fold_classifier = KerasClassifier(build_fn=build_classifier, batch_size=32, epochs=500)
# cv: nbr of folds
# n_jobs: number of cpus used
accuracies = cross_val_score(estimator=k_fold_classifier, X = X_train, y=y_train, cv=10, n_jobs=-1)
mean = accuracies.mean()
variance = accuracies.std()

# MEAN: 0.74620500849632765
# variance: 0.13132454887951542
# Trying to improve the network

# Improving the ANN with grid search
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Grid search expects a function to build the NN
# Tuning paramaters that already exists must be passed as arguments to the build
def build_classifier(optimizer, loss, unit_1, unit_2, unit_3, node_2, node_3):
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    classifier = Sequential()
    
    classifier.add(Dense(units=unit_1, activation='relu', kernel_initializer='uniform', input_dim=9))
    classifier.add(Dropout(rate=0.1))
    if (node_2):
        classifier.add(Dense(units=unit_2, activation='relu', kernel_initializer='uniform'))
        classifier.add(Dropout(rate=0.1))
    if (node_3):
        classifier.add(Dense(units=unit_3, activation='relu', kernel_initializer='uniform'))
        classifier.add(Dropout(rate=0.1))
    
    classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))
    
    classifier.compile(optimizer=optimizer, loss=loss, metrics= ['accuracy'])
    return classifier

# Same constructor as for the Sequential fit function
grid_search_classifier = KerasClassifier(build_fn=build_classifier)

parameters = {'batch_size': [25, 32, 64, 128],
              'nb_epoch': [50, 100, 500],
              'optimizer': ['adam', 'rmsprop'],
              'loss': ['binary_crossentropy', 'categorical_crossentropy']
              'unit_1': [5, 8, 16],
              'unit_2': [5, 8, 16],
              'unit_3': [5, 8, 16],
              'node_2': [True, False],
              'node_3': [True, False]
              }


# Initialize the grid search
grid_search = GridSearchCV(estimator=grid_search_classifier, param_grid=parameters, n_jobs=-1, cv=10, verbose=1, scoring = 'accuracy')

best_paramerters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print ('best_parameter: ', best_paramerters)
print ('best_accuracy:', best_accuracy)