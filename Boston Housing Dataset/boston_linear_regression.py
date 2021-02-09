import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

boston_data = pd.read_csv('BostonHousing.csv')

X = boston_data.drop('medv', axis = 1).values
y = boston_data['medv'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

#comparing predictions and actual targets
print(y_pred)
print(y_test)

score = reg.score(X_test, y_test)
print(score)