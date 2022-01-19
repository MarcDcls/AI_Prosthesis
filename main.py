from tensorflow.keras import models
from random import randint
from data import get_in_out_basic_NN
import numpy as np

basic_NN = models.load_model("models/basic_NN")
inputs, outputs = get_in_out_basic_NN()
n, p = inputs.shape

for i in range(10):
    ind = randint(0, n)
    output = outputs[ind]
    input = inputs[ind]
    print("Expected values :", output)
    computed_output = basic_NN(np.reshape(input, (1, 7)))
    print("Computed values :", computed_output)


