import numpy as np
from data import load_data
from tensorflow.keras import models


def generate_smart_seq(n):
    """
    Generate angles of the arm with the predictive NN during the sequence based on the last real configuration of the
    arm for each prediction

    :param n: number of entry to process
    :return: outputs (angles of the arm)
    """
    predictive_NN = models.load_model("models/predictive_NN")
    data = load_data()

    result_bras = []
    for i in range(n):
        computed_output = predictive_NN.predict(np.reshape(data[i, :12], (1, 12)))[0]
        result_bras.append(computed_output)
    result_bras = np.array(result_bras)
    return result_bras


def generate_smart_generative_seq(n):
    """
    Generate angles of the arm with the predictive NN during the sequence by generating the prediction based only on
    the first configuration of the arm

    :param n: number of entry to process
    :return: outputs (angles of the arm)
    """
    predictive_NN = models.load_model("models/predictive_NN")
    data = load_data()

    result_bras = []
    computed_output = data[0, 2:7]
    for i in range(n):
        inputs = np.concatenate((data[i, :2], computed_output, data[i, 7:12]), axis=0)
        computed_output = predictive_NN.predict(np.reshape(inputs, (1, 12)))[0]
        result_bras.append(computed_output)
    result_bras = np.array(result_bras)
    return result_bras
