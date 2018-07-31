# Script for aiSaturdays challeng by Nairobi Women in DS/ML

import pandas as pd;
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from  sklearn.linear_model import LogisticRegression
from  sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

#########
# 1st part of assignment. Predict type of plant based on
# sepal_length, sepal_width, petal_length, petal_width
#########

names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'labels']

iris_train = pd.read_csv("data/iris_train.csv", header = 1, names = names)
iris_test = pd.read_csv("data/iris_test.csv", header = 1, names = names[:-1])

print(iris_train.head())
print(iris_test.head())


# Look at descriptives
print(iris_train.describe())
print(iris_test.describe())

# Split train data in X and y
X_train = iris_train.iloc[:, 0:4]
y_train = iris_train.iloc[:, -1]

# scale data for classification

scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
print('Standardized features\n')
print(str(X_train[:4]))
X_test = scaler.transform(iris_test)
print('Standardized test features\n')
print(str(X_test[:4]))


# Baseline logistic regression model
# perform 5 fold cross validation since training data is small to
# get an estimate of accuracy
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
scores = cross_val_score(lr_model, X_train, y_train, cv=5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# --> LR Accuracy: 0.90 (+/- 0.09)

# svm_model
# perform 5 fold cross validation since training data is small to
# get an estimate of accuracy
svm_model = LinearSVC(C=10)
scores = cross_val_score(svm_model, X_train, y_train, cv=5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# --> SVM Accuracy: 0.95 (+/- 0.12)

# Based on CV, select as SVM as final model

svm_model.fit(X_train,y_train)
y_pred = pd.DataFrame({"4" : svm_model.predict(X_test)})

print(y_pred)


y_pred.to_csv('submissions/iris_test_labelled.csv')
