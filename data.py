import numpy as np


def load_data():
    """
    Load data with the format : [shPitch, shRoll, armYaw, elbPitch, forearmYaw, wriPitch, wriRoll, tgtPosX, tgtPosY,
    tgtPosZ, tgtPitch, tgtRoll, handRemapPosX, handRemapPosY, handRemapPosZ, handRemapPitch, handRemapRoll]

    :return: data
    """
    print("Loading general data...")
    data = np.genfromtxt('data/corpus_students_only_validated_targets.csv', delimiter=',',
                         usecols=(4, 5, 6, 7, 8, 9, 10, 29, 30, 31, 32, 33, 53, 54, 55, 56, 57))[1:, :]
    print("General data loaded !")
    return data


def load_data_seq_hand():
    """
    Load data with the format : [tgt_number, handRemapPosX, handRemapPosY, handRemapPosZ, handRemapPitch, handRemapRoll]

    :return: data
    """
    print("Loading hand data...")
    data = np.genfromtxt('data/corpus_students_only_validated_targets.csv', delimiter=',',
                         usecols=(2, 53, 54, 55, 56, 57))[1:, :]
    print("Hand data loaded !")
    return data


def load_data_seq_shoulder():
    """
    Load data with the format : [shPitch, shRoll]

    :return: data
    """
    print("Loading shoulder data...")
    data = np.genfromtxt('data/corpus_students_only_validated_targets.csv', delimiter=',',
                         usecols=(4, 5))[1:, :]
    print("Shoulder data loaded !")
    return data


def get_current_target_naive_seq(n):
    """
    Prepare the data for the interpolation of the hand trajectory

    :param n: number of entry to process
    :return: list of the hand first configurations, list of the hand last configurations, list of the number of
    configurations between the first and the last configurations
    """
    data = load_data_seq_hand()
    print("Data processing ...")
    lastTgtN = data[0, 0]
    firsts = [data[0, 1:]]
    lasts = []
    nb_pos = []
    count = 1
    for i in range(1, n):
        if data[i, 0] != lastTgtN:
            lasts.append(data[i - 1, 1:])
            firsts.append(data[i, 1:])
            lastTgtN = data[i, 0]
            nb_pos.append(count)
            count = 1
        else:
            count = count + 1
    lasts.append(data[-1, 1:])
    nb_pos.append(count)
    print("Data processed !")
    firsts = np.array(firsts)
    lasts = np.array(lasts)
    nb_pos = np.array(nb_pos)
    return firsts, lasts, nb_pos


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
    for i in range(len(data) - 1):
        inputs.append(data[i])
        outputs.append(data[i + 1, 2:7])

    return np.array(inputs), np.array(outputs)


def format_data(prediction, path):
    """
    Format the predicted data for Unity and save it in CSV

    :param global_data: global data (from load_data()))
    :param prediction: predicted angles of the arm
    :param path: saving location
    :return: None
    """
    print("Formatting data...")
    n, p = prediction.shape
    data = np.genfromtxt('data/corpus_students_only_validated_targets.csv', delimiter=',',
                         usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 59, 60, 61, 62, 63, 64))[1:n+1, :]
    sh_predicted = data[:n, 4:6]
    formatted_data = np.concatenate((data, sh_predicted, prediction), axis=1)
    np.savetxt(path, formatted_data, delimiter=',')
    print("Data formatted and saved !")
    return data

def format_data_from_csv(prediction_csv, path):

     """
     Format the predicted data from a csv file for Unity and save it in CSV

    :param global_data: global data (from load_data()))
    :param prediction: predicted angles of the arm in a csv file
    :param path: saving location
    :return: None
    """
     print("Formatting data...")
     prediction = np.genfromtxt(prediction_csv, delimiter=',')
     #print(prediction)
     n, p = prediction.shape
     data = np.genfromtxt('data/corpus_students_only_validated_targets.csv', delimiter=',',
                         usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 59, 60, 61, 62, 63, 64))[1:n+1, :]
     sh_predicted = data[:n, 4:6]
     formatted_data = np.concatenate((data, sh_predicted, prediction), axis=1)
     np.savetxt(path, formatted_data, delimiter=',')
     print("Data formatted and saved !")
     return data

format_data_from_csv('results_bras.csv', 'formated_data.csv')
