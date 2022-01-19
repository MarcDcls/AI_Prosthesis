import numpy as np
from math import *
from data import get_current_target_naive_seq, load_data_seq_shoulder
from tensorflow.keras import models

def interpolate(posHandStart, posHandEnd, step):
    return np.linspace(posHandStart, posHandEnd, step)


def generate_seq():
    basic_NN = models.load_model("models/basic_NN")
    shoulders = load_data_seq_shoulder()
    firsts, lasts, nb_pos = get_current_target_naive_seq()
    hands = []
    for i in range(len(nb_pos)):
        interpolations = interpolate(firsts[i, 1:], lasts[i, :], nb_pos[i])
        for j in range(len(interpolations)):
            hands.append(interpolations[j])
    inputs = np.concatenate((shoulders, hands), axis=1)
    print(inputs.shape)
    
    outputs = []
    for hand in hands:
        input = np.concatenate(())
        outputs.append(basic_NN(np.reshape(hand, (1, 7)[:, 1:])).numpy())

generate_seq()