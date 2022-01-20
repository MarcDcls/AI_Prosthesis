import numpy as np
from tensorflow.keras import models

from data import get_first_and_last_hands_seq, load_data_seq_shoulder, get_first_and_last_hands_and_shoulders_seq


def interpolate(posHandStart, posHandEnd, step):
    """
    Interpolate successive positions and orientations of the and between 2 configurations

    :param posHandStart: initial configuration
    :param posHandEnd: final configuration
    :param step: number of positions
    :return: sequence of configurations of the hand
    """
    return np.linspace(posHandStart, posHandEnd, step)


def generate_naive_seq(n):
    """
    Generate angles of the arm with the basic NN on the naive sequence

    :param n: number of entry to process
    :return: outputs (angles of the arm)
    """
    basic_NN = models.load_model("models/basic_NN")
    shoulders = load_data_seq_shoulder()[:n, :]
    firsts, lasts, nb_pos = get_first_and_last_hands_seq(n)
    hands = []
    for i in range(len(nb_pos)):
        interpolations = interpolate(firsts[i, :], lasts[i, :], nb_pos[i])
        for j in range(len(interpolations)):
            hands.append(interpolations[j])
    inputs = np.concatenate((shoulders, hands), axis=1)
    print("Predicting sequence ...")
    outputs = basic_NN.predict(inputs)
    print("Sequence predicted !")
    return outputs, hands

def generate_interpolated_seq(n):
    basic_NN = models.load_model("models/basic_NN")
    firsts, lasts, nb_pos = get_first_and_last_hands_and_shoulders_seq(n)
    output_firsts = basic_NN.predict(firsts)
    output_lasts = basic_NN.predict(lasts)
    outputs = []
    for i in range(len(nb_pos)):
        interpolations = interpolate(output_firsts[i, :], output_lasts[i, :], nb_pos[i])
        for j in range(len(interpolations)):
            outputs.append(interpolations[j])
    return np.array(outputs)