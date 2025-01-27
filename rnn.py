import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataTrain = pd.read_csv("train.csv")
# dataTest = pd.read_csv("test.csv")


print(dataTrain.isnull().sum())
# print(dataTest.isnull().sum())

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