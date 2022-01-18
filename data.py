import numpy as np


def load_data():
    """
    Load data and split them into inputs and outputs with the format :

    inputs = [shPitch, shRoll, handRemapPosX, handRemapPosY, handRemapPosZ, handRemapPitch, handRemapRoll]

    outputs = [armYaw, elbPitch, forearmYaw, wriPitch, wriRoll]

    :return: inputs, outputs
    """
    data = np.genfromtxt('data/corpus_students_only_validated_targets.csv', delimiter=',',
                         usecols=(4, 5, 6, 7, 8, 9, 10, 53, 54, 55, 56, 57))[1:, :]
    inputs = np.concatenate((data[:, :2], data[:, 7:]), axis=1)
    outputs = data[:, 2:7]
    return inputs, outputs
