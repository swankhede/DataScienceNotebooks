# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:06:09 2020

@author: SwanKhede
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('heart.csv')


x = data.iloc[:,:-1].values
y = data.iloc[:,[-1]]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#using svm
from sklearn import svm
cls = svm.SVC(kernel='linear')
cls.fit(x_train,y_train.values.ravel())
pred = cls.predict(x_test)
print(pred)


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
nb_pred=nb.predict(x_test)

#using knn


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X_train,y_train)
k_pred = clf.predict(X_test)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(x_train,y_train)
tree_pred = tree.predict(x_test)



from sklearn.metrics import accuracy_score
svm = accuracy_score(y_test, pred)
tree = accuracy_score(y_test, tree_pred)
knn = accuracy_score(y_test, k_pred)
nbg = accuracy_score(y_test, nb_pred)

sns.heatmap(data.corr(),annot=True)


sns.pairplot(data,hue='target')
