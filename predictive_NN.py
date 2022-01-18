import numpy as np
from tensorflow.keras import Sequential, layers
from sklearn.model_selection import train_test_split
from data import get_in_out_simple_predictive_NN


# Parameters
learning_rate = 1.59e-7      # unused
momentum = 0.95              # unused
nb_epochs = 10
batch_size = 128
test_ratio = 0.3
nb_neurone_layer_1 = 256
nb_neurone_layer_2 = 256
nb_neurone_layer_3 = 64

#Load data
print("Data loading ...")
inputs, outputs = get_in_out_simple_predictive_NN()
n, p = inputs.shape
nb_inputs = 17
nb_outputs = 5

X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=test_ratio, random_state=1)
print("Data loaded !")


mod = Sequential()
mod.add(layers.Dense(nb_neurone_layer_1, activation='relu', input_shape=(nb_inputs,)))
mod.add(layers.Dense(nb_neurone_layer_2, activation='relu'))
mod.add(layers.Dropout(0.5))
mod.add(layers.Dense(nb_neurone_layer_3, activation='relu'))
mod.add(layers.Dense(nb_outputs, activation='linear'))
mod.compile(loss='mse')
print("Model created !")

print("Training of the model ...")
hist = mod.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs)
print("Model trained !")
mod.save('model/.')
print("Model saved !")