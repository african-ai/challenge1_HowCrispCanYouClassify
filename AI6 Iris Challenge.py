
# coding: utf-8

# In[154]:


import os
import numpy as np #linear algebra
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('pylab', 'inline')


# In[155]:


#sklearn imports
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[156]:


#Set the working directiry
os.chdir('C:\\Users\path-to-working-directory') #Provide the path to your working folder here
#just to confirm our working directory
print(os.getcwd())

#Reading data from CSV file
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df_train = pd.read_csv("./data/iris_train.csv",index_col=0)
df_test = pd.read_csv("./data/iris_test.csv",index_col=0)

#check if the data was loaded
print('There are {} samples in the training set and {} samples in the test set.'.format(df_train.shape[0], df_test.shape[0]))


# In[157]:


#Exploring the data
print(df_train.head(10))


# In[158]:


print(df_test.head(10))


# In[159]:


print(df_train.shape)
print(df_test.shape)


# In[160]:


df_train.hist(bins=20)
plt.show()


# In[161]:


#Let’s now take a look at the number of instances (rows) that belong to each class. We can view this as an absolute count.

print(df_train.groupby('labels').size())


# In[162]:


#Defining data and label
X_learning = df_train.iloc[:, 0:4]
Y_learning = df_train.iloc[:, 4]
X_testing = df_test.iloc[:,0:4]
print(X_testing.head(5))


# In[163]:


#Split data into training and test datasets (training will be based on 70% of data)
validation_size = 0.20
seed = 7
scoring = 'accuracy'
x_learning_temp, x_test, y_learning_temp, y_test = model_selection.train_test_split(X_learning, Y_learning, test_size=validation_size, random_state=seed) 
#test_size: if integer, number of examples into test dataset; if between 0.0 and 1.0, means proportion
print('There are {} samples in the training set and {} samples in the test set'.format(X_train.shape[0], X_test.shape[0]))

#print(X_train)
#print(X_test)
#print(y_train)
#print(y_test)


# In[164]:


#Model 1 SVC (Support Vector Classification)


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('RF', RandomForestClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, x_learning_temp, y_learning_temp, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


#svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
#knn = KNeighborsClassifier(n_neighbors = 7, p = 2, metric='minkowski')
#decision_tree = tree.DecisionTreeClassifier(criterion='gini')
#random_forest = RandomForestClassifier()

#svm.fit(x_learning, y_learning)
#print('The accuracy of the SVM classifier on training data is {:.2f}'.format(svm.score(x_learning, y_learning)))
#print('The accuracy of the SVM classifier on test data is {:.2f}'.format(svm.score(x_test, y_test)))


# In[167]:


# Make predictions on validation dataset
knn = KNeighborsClassifier(n_neighbors = 7, p = 2, metric='minkowski')
random_forest=RandomForestClassifier()
svm = SVC()
knn.fit(x_learning_temp, y_learning_temp)
predictions = knn.predict(x_test)
print(predictions)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# In[169]:


#Final Test Dat
knn = KNeighborsClassifier(n_neighbors = 7, p = 2, metric='minkowski')
random_forest=RandomForestClassifier()
svm = SVC()
knn.fit(X_learning, Y_learning)
predictions = knn.predict(X_testing)
print(predictions)
dataset = pd.DataFrame({'4':predictions})
print(dataset.head())
dataset.to_csv('./submissions/iris_test_labelled.csv')

