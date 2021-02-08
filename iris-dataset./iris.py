from sklearn import datasets
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier


plt.style.use('ggplot')

iris = datasets.load_iris()

x = iris.data
y = iris.target 
df = pd.DataFrame(x, columns = iris.feature_names)
print(df.head())

knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(x, y)


new = np.array([[5.6, 2.8, 3.9, 1.1]])
prediction = knn.predict(new)

print(prediction)
