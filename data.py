import numpy as np


def load_data():
    """
    Load data with the format : [shPitch, shRoll, armYaw, elbPitch, forearmYaw, wriPitch, wriRoll, tgtPosX, tgtPosY,
    tgtPosZ, tgtPitch, tgtRoll, handRemapPosX, handRemapPosY, handRemapPosZ, handRemapPitch, handRemapRoll]

    :return: data
    """
    data = np.genfromtxt('data/corpus_students_only_validated_targets.csv', delimiter=',',
                         usecols=(4, 5, 6, 7, 8, 9, 10, 29, 30, 31, 32, 33, 53, 54, 55, 56, 57))[1:, :]
    return data


def get_in_out_basic_NN():
    """
    Split data into inputs and outputs for the basic NN with the format :

    inputs = [shPitch, shRoll, handRemapPosX, handRemapPosY, handRemapPosZ, handRemapPitch, handRemapRoll]

    outputs = [armYaw, elbPitch, forearmYaw, wriPitch, wriRoll]

    :return: inputs, outputs
    """
    data = load_data()
    inputs = np.concatenate((data[:, :2], data[:, 12:]), axis=1)
    outputs = data[:, 2:7]
    return inputs, outputs

def get_in_out_simple_predictive_NN():
    """
    Split data into inputs (of size 17) and outputs for the predictive NN with the format :

    inputs = [shPitch(t), shRoll(t), handRemapPosX(t), handRemapPosY(t), handRemapPosZ(t), handRemapPitch(t),
    handRemapRoll(t), tgtPosX(t), tgtPosY(t), tgtPosZ(t), tgtPitch(t), tgtRoll(t)]

    outputs = [armYaw(t+1), elbPitch(t+1), forearmYaw(t+1), wriPitch(t+1), wriRoll(t+1)]

    :return: inputs, outputs
    """
    data = load_data()
    inputs = []
    outputs = []
    return inputs, outputs