
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import re


filename = 'data/iris_train.csv'
df = pd.read_csv(filename, index_col = 0)
labels = df['labels']
df = df.drop(['labels'], axis = 1)
X_train, y_train = df, labels


def cross_validate(X_train, y_train):
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1500, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt', 'log2']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 7, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [False, True]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    classifier = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = classifier, param_distributions = random_grid, 
                                   n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = 3)

    rf_random.fit(X_train, y_train)
    print rf_random.best_estimator_


#cross_validate(df, labels)


classifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=80, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=4, min_samples_split=7,
            min_weight_fraction_leaf=0.0, n_estimators=488, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
# classifier = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
#             max_depth=90, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=5,
#             min_weight_fraction_leaf=0.0, n_estimators=922, n_jobs=2,
#             oob_score=False, random_state=None, verbose=0,
#             warm_start=False)


X_train, X_test,y_train, y_test = train_test_split(df, labels, test_size = 0.3)


#sm = SMOTE(random_state = 42)
#X_train, y_train = sm.fit_sample(X_train, y_train)
print classification_report(y_train, y_train)


pred = classifier.predict(X_test)
print (classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))


classifier.fit(X_train, y_train)
filename = 'data/iris_test.csv'
test_df = pd.read_csv(filename, index_col = 0)
pred_test = classifier.predict(test_df)


test_df['pred'] = pred_test


final_df = test_df['pred']


final_df.to_csv('iris_final.csv')

######################
    

