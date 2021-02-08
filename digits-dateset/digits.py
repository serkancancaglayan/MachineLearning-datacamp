from sklearn import datasets
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


digits = datasets.load_digits()

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 21, stratify = y)

knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(X_train, y_train)

acccuracy = knn.score(X_test, y_test)
print(acccuracy)

prediction = knn.predict([digits.data[0]])
print(prediction)
plt.imshow(digits.images[0], cmap = plt.cm.gray_r, interpolation = 'nearest')
plt.show()
	
