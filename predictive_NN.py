import numpy as np
from data import get_in_out_simple_predictive_NN
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, layers
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


def generate_predictive_NN():
    """
    Generate the predictive NN which predict the next angles of the arm based on the current ones

    :return: None
    """
    # Load data
    inputs, outputs = get_in_out_simple_predictive_NN()
    n, p = inputs.shape
    nb_inputs = 12  # 17
    nb_outputs = 5

    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=test_ratio)

    mod = Sequential()
    mod.add(layers.Dense(nb_neurone_layer_1, activation='relu', input_shape=(nb_inputs,)))
    mod.add(layers.Dense(nb_neurone_layer_2, activation='relu'))
    mod.add(layers.Dropout(0.5))
    mod.add(layers.Dense(nb_neurone_layer_3, activation='relu'))
    mod.add(layers.Dense(nb_outputs, activation='linear'))
    mod.compile(loss='mse')
    print("Model created !")

    print("Training of the model ...")
    hist = mod.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(X_test, y_test))
    print("Model trained !")
    mod.save('models/predictive_NN')
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
