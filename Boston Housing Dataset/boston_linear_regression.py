import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
boston_data = pd.read_csv('BostonHousing.csv')

X = boston_data.drop('medv', axis = 1).values
y = boston_data['medv'].values


X_rooms = X[:,5]

y = y.reshape(-1, 1)
X_rooms = X_rooms.reshape(-1, 1)

reg = LinearRegression()
reg.fit(X_rooms, y)

prediction_shape = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1, 1)
plt.scatter(X_rooms, y, color = 'blue')
plt.title('House Price ')
plt.ylabel('Price 10k $')
plt.xlabel('Rooms')
plt.plot(prediction_shape, reg.predict(prediction_shape), color = 'black', linewidth = 3)
plt.show()
