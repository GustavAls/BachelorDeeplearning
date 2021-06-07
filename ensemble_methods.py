import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg as lng
import scipy.io as io
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate
from sklearn.model_selection import GridSearchCV
import os


X = data['X']
y = data['y'].ravel()
N, P = X.shape
# Try to experiment with max_samples, max_features, number of modles, and other models
n_estimators = range(5, 101)
max_depth = range(1, 11)

# We do an outer loop over max_depth here ourselves because we cannot include in the CV paramgrid.
# Notice this is not a "proper" way to select the best max_depth but for the purpose of vizuallizing behaviour it should do

parameter = {'n_estimators': [i for i in n_estimators]}
test_acc = np.zeros((len(n_estimators), len(max_depth)))
for i in max_depth:
    # Create and fit an AdaBoosted decision tree
    boost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=i))

    # Fit the grid search model
    boost_grid = GridSearchCV(boost, parameter)

    # Fit the grid search model
    boost_grid.fit(X, y)

    test_acc[:, i - 1] = boost_grid.cv_results_['mean_test_score']

fig, ax = plt.subplots(figsize=(15,15))

ax.plot(n_estimators, test_acc)
ax.set_xlabel('Maximum tree depth')
ax.set_ylabel('Mean test accuracy')
ax.legend(['MaxDepth=1','MaxDepth=2','MaxDepth=3','MaxDepth=4','MaxDepth=5','MaxDepth=6','MaxDepth=7','MaxDepth=8','MaxDepth=9','MaxDepth=10'])
plt.show()
# Try to change the learning rate and add it to param_grid