#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 20:55:55 2017

@author: chenhua
"""

# =============================================================================
# 1. Supervised learning: K nearest neighbors
# =============================================================================

import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# load package stored dataset iris
iris = datasets.load_iris()
type(iris)

# provide key values of the iris object
iris.keys()
print(iris.keys())

# check column names
iris.feature_names

type(iris.data), type(iris.target)
iris.data.shape


# =============================================================================
# Train/test split 
# =============================================================================

X_iris = iris.data
y_iris = iris.target


df_iris = pd.DataFrame(X_iris, columns = iris.feature_names)
df_iris.head()

X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size = 0.3,
                                                    random_state = 21, # Seed setting
                                                    stratify = y_iris) # Stratify the split
                                                                       # according to the labels
                                                                       # so that they are distributed
                                                                       # in the train and test sets
                                                                       # as they are in the original dataset.

# =============================================================================
# Observe optimal number of neighbors 
# =============================================================================

# Setup array to store train and test accuracy
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    
    #Setup a k-NN classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors = neighbors[i])
    
    #Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set 
    train_accuracy[i] = knn.score(X_train, y_train)
    
    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)
    
# Genenrate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()





