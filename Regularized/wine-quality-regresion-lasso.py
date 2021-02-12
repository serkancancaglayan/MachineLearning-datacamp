#dataset : https://archive.ics.uci.edu/ml/datasets/wine+quality
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split


wine_data = pd.read_csv('winequailty-red.csv', sep = ';')
print(wine_data.columns.tolist())

X = wine_data.drop('quality', axis = 1).values
y = wine_data['quality'].values

names = wine_data.drop('quality', axis = 1).columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

lasso = Lasso(alpha = 0.1)
lasso_coef = lasso.fit(X, y).coef_

_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation = 60)
_ = plt.ylabel('Coefficients')
plt.show()
