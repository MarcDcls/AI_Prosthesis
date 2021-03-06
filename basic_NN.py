import numpy as np
import sklearn.metrics
from data import get_in_out_basic_NN
from random import randint
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, layers, optimizers
from tensorflow.keras import models
import matplotlib.pyplot as plt

# Parameters
learning_rate = 1.59e-7
momentum = 0.95
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

    mod = Sequential()
    mod.add(layers.Dense(nb_neurone_layer_1, activation='relu', input_shape=(nb_inputs,)))
    mod.add(layers.Dense(nb_neurone_layer_2, activation='relu'))
    mod.add(layers.Dropout(0.5))
    mod.add(layers.Dense(nb_neurone_layer_3, activation='relu'))
    mod.add(layers.Dense(nb_outputs, activation='linear'))
    mod.compile(optimizer=optimizers.RMSprop(lr=learning_rate, momentum=momentum), loss='mse')
    print("Model created !")

    print("Training of the models ...")
    hist = mod.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(X_test, y_test))
    print("Model trained !")
    print("Model saved !")

    loss_values = hist.history['loss']
    val_loss_values = hist.history['val_loss']
    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, label='Training Loss')
    plt.plot(epochs, val_loss_values, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


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
