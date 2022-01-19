import numpy as np
from tensorflow.keras import models

from data import get_current_target_naive_seq, load_data_seq_shoulder


def interpolate(posHandStart, posHandEnd, step):
    """
    Interpolate successive positions and orientations of the and between 2 configurations

    :param posHandStart: initial configuration
    :param posHandEnd: final configuration
    :param step: number of positions
    :return: sequence of configurations of the hand
    """
    return np.linspace(posHandStart, posHandEnd, step)


def generate_naive_seq():
    """
    Generate angles of the arm with the basic NN on the naive sequence

    :return: outputs (angles of the arm)
    """
    basic_NN = models.load_model("models/basic_NN")
    shoulders = load_data_seq_shoulder()
    firsts, lasts, nb_pos = get_current_target_naive_seq()
    hands = []
    for i in range(len(nb_pos)):
        interpolations = interpolate(firsts[i, :], lasts[i, :], nb_pos[i])
        for j in range(len(interpolations)):
            hands.append(interpolations[j])
    inputs = np.concatenate((shoulders, hands), axis=1)
    print("Predicting sequence ...")
    outputs = basic_NN.predict(inputs)
    print("Sequence predicted !")
    return outputs