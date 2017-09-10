from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np

np.random.seed(7)

def create_model():
    model = Sequential()
    model.add((Dense(12, input_dim = 8, kernel_initializer = 'he_normal', activation = 'relu')))
    model.add((Dense(8, kernel_initializer = 'he_normal', activation = 'relu')))
    model.add(Dense(1, kernel_initializer = 'he_normal', activation = 'sigmoid'))
    
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

data = np.loadtxt('pima-indians-diabetes.data.txt', delimiter = ',')
X = data[:, :-1]
Y = data[:, -1]

model = KerasClassifier(build_fn = create_model, epochs = 150, batch_size = 10, verbose = 0)
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 7)
results = cross_val_score(model, X, Y, cv = kfold)

print(results)
print(results.mean())