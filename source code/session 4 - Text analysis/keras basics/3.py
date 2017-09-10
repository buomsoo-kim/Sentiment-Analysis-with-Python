# LSTM with Keras (airline passenger prediction)
import numpy as np
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

np.random.seed(7)

def create_dataset(dataset, look_back = 1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back -1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

df = pd.read_csv('international-airline-passengers.csv', usecols = [1], engine='python', skipfooter=3)

data = df.values.astype('float32')
scaler = MinMaxScaler(feature_range = (0,1))
data = scaler.fit_transform(data)

train_size = int(len(data) * 0.67)
train_data , test_data = data[:train_size, :], data[train_size:, :]

look_back = 1
X_train, Y_train = create_dataset(train_data, look_back)
X_test, Y_test = create_dataset(test_data, look_back)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
#print(X_train.shape)
#print(X_test.shape)

model = Sequential()
model.add(LSTM(4, input_shape = (None, look_back)))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.fit(X_train, Y_train, epochs = 100, batch_size = 1, verbose = 2)

train_prediction = scaler.inverse_transform(model.predict(X_train))
test_prediction = scaler.inverse_transform(model.predict(X_test))
Y_train = scaler.inverse_transform([Y_train])
Y_test = scaler.inverse_transform([Y_test])

train_score = math.sqrt(mean_squared_error(Y_train[0], train_prediction[:, 0]))
print('Train Score: {} RMSE'.format(train_score))

test_score = math.sqrt(mean_squared_error(Y_test[0], test_prediction[:, 0]))
print('Test Score: {} RMSE'.format(test_score))