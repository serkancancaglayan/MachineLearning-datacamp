from sklearn import datasets
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

plt.style.use('ggplot')

iris = datasets.load_iris()

x = iris.data
y = iris.target 


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 21, stratify = y)


#larger n_neighbors = smoother decision boundary = less complex model 
#smaller n_neighbors = more accuracy = complex model
knn = KNeighborsClassifier(n_neighbors = 8)
knn.fit(X_train, y_train)

y_predictions = knn.predict(X_test)
score = knn.score(X_test, y_test)
print(score)
