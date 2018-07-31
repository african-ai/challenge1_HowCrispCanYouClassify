# Script for aiSaturdays challeng by Nairobi Women in DS/ML

import pandas as pd;
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from  sklearn.linear_model import LogisticRegression
from  sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier



#########
# 2nd part of assignment. to identify a digit/number given
# optical data from handwritten digits
#########

digits_train = pd.read_csv("data/digits_train.csv")
digits_test = pd.read_csv("data/digits_test.csv")

digits_train = digits_train.drop("Unnamed: 0", axis=1)
digits_test = digits_test.drop("Unnamed: 0", axis=1)

print(digits_train.columns)

print(digits_train.head())
print(digits_test.head())


# Descriptives
# print(digits_train.describe())
# print(digits_test.describe())

print(digits_train.shape)
print(digits_test.shape)


# Split digits_train in X and y
X = digits_train.iloc[:, 0:64]
print (X.head())
y = digits_train.iloc[:, -1]

# Split digits_train in training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)


X_test = digits_test

# Baseline Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_val)
print('The accuracy of the Logistic Regression is', accuracy_score(y_pred, y_val))


knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_val)
print ('The accuracy of the KNN is', accuracy_score(y_pred, y_val))


svm_model = LinearSVC(C=10)
svm_model.fit(X_train,y_train)
y_pred = svm_model.predict(X_val)
print('The accuracy of the Support Vector Machine is', accuracy_score(y_pred, y_val))

svm_model13 = LinearSVC(C=0.1)
svm_model13.fit(X_train,y_train)
y_pred = svm_model13.predict(X_val)
print('The accuracy of the Support Vector Machine is', accuracy_score(y_pred, y_val))

# Problem seems to be very linear. Choose linear regression for final prediction
y_pred = pd.DataFrame({"labels" : lr_model.predict(X_test)})

y_pred.to_csv('submissions/digit_test_labelled.csv', index_label = "id")
