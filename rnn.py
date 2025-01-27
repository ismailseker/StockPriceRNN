import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataTrain = pd.read_csv("train.csv")

print(dataTrain.isnull().sum())

train = dataTrain.loc[:,['Open']].values

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
trainScaled = scaler.fit_transform(train)

plt.plot(trainScaled)
plt.show()

xTrain = []
yTrain = []
timesteps = 50

for i in range(timesteps,1258):
    xTrain.append(trainScaled[i-timesteps:i,0])
    yTrain.append(trainScaled[i,0])
xTrain,yTrain = np.array(xTrain),np.array(yTrain)

xTrain = np.reshape(xTrain, (xTrain.shape[0],xTrain.shape[1],1))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

regressor = Sequential()

regressor.add(LSTM(units=50,activation='tanh',return_sequences = True, input_shape=(xTrain.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50,activation='tanh',return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50,activation='tanh',return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(xTrain,yTrain,epochs = 100, batch_size = 32)

dataTest = pd.read_csv("test.csv")

print(dataTest.isnull().sum())

realStockPrice = dataTest.iloc[:,1:2].values

dataTotal = pd.concat((dataTrain['Open'], dataTrain['Open']), axis = 0)
inputs = dataTotal[len(dataTotal) - len(dataTest) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

plt.plot(realStockPrice, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

