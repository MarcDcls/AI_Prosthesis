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

def load_data_seq():
    """
    Load data with the format : [tgt_number, shPitch, shRoll, handRemapPosX, handRemapPosY, handRemapPosZ,
    handRemapPitch, handRemapRoll]

    :return: data
    """
    data = np.genfromtxt('data/corpus_students_only_validated_targets.csv', delimiter=',',
                         usecols=(2, 4, 5, 53, 54, 55, 56, 57))[1:, :]
    return data 

def get_current_target_naive_seq():
    data = load_data_seq()
    lastTgtN = 0
    current = []
    target = []
    counts = []
    count = 0
    for i in range(len(data)):
        current.append(data[i,:])
        if data[i, 0] != lastTgtN:
            target.append(data[i-1,:])
            lastTgtN = data[i, 0]
            counts.append(count)
            count = 0
        count = count + 1
    current = np.array(current)
    target = np.array(target)
    counts = np.array(counts)
    return current, target, counts

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
    for i in range(len(data)-1):
        inputs.append(data[i])
        outputs.append(data[i+1,2:7])
    
    return np.array(inputs), np.array(outputs)