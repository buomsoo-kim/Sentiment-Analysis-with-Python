from keras.models import Sequential
from keras.layers import Dense
import numpy as np

np.random.seed(7)

data = np.loadtxt('pima-indians-diabetes.data.txt', delimiter = ',')

X = data[:, 0:-1]
Y = data[:, -1]

model = Sequential()
model.add(Dense(12, input_dim = 8, kernel_initializer = 'uniform', activation = 'sigmoid'))
model.add(Dense(8, kernel_initializer = 'uniform', activation = 'sigmoid'))
model.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])
model.fit(X, Y, epochs = 150, batch_size = 10)

scores = model.evaluate(X,Y)
print(model.metrics_names[1], scores[1]*100)    # printing the accuracy of the prediction