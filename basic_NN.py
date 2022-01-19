from random import randint

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, layers
from tensorflow.keras import models

from data import get_in_out_basic_NN

# Parameters
learning_rate = 1.59e-7  # unused
momentum = 0.95  # unused
nb_epochs = 10
batch_size = 128
test_ratio = 0.3
nb_neurone_layer_1 = 256
nb_neurone_layer_2 = 256
nb_neurone_layer_3 = 64

def generate_basic_NN():
    """
    Generate the basic NN which predict the inverse kinematic

    :return: None
    """

    # Load data
    inputs, outputs = get_in_out_basic_NN()
    n, p = inputs.shape
    nb_inputs = 7
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

    print("Training of the models ...")
    mod.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs)
    print("Model trained !")
    mod.save('models/basic_NN')
    print("Model saved !")


def test_basic_NN(n=10):
    """
    Apply the basic NN to n random valid configurations

    :param n: number of tests
    :return: None
    """
    basic_NN = models.load_model("models/basic_NN")
    inputs, outputs = get_in_out_basic_NN()
    n, p = inputs.shape

    for i in range(10):
        ind = randint(0, n)
        output = outputs[ind]
        input = inputs[ind]
        print("Expected values :", output)
        computed_output = basic_NN.predict(np.reshape(input, (1, 7)))[0]
        print("Computed values :", computed_output)
